import logging
import queue
import threading

from voice_app.audio_capture import AudioCapture
from voice_app.config import AppConfig
from voice_app.history import append_message, load_history
from voice_app.llm import LLMClient
from voice_app.state_machine import AppEvent, State, StateMachine
from voice_app.stt import STTClient
from voice_app.tts import TTSEngine
from voice_app.vad import VADDetector

logger = logging.getLogger(__name__)


class Orchestrator:
    """Coordinate all background workers and drive the conversation lifecycle."""

    def __init__(self, config: AppConfig, debug: bool = False) -> None:
        self.config = config
        self.debug = debug
        self.event_queue: queue.Queue = queue.Queue()
        self.state_machine = StateMachine()
        self._gen_id = 0
        self._shutdown_event = threading.Event()
        self._vad_thread: threading.Thread | None = None
        self._pending_response_by_gen: dict[int, str] = {}

        self._vad_audio_queue: queue.Queue = queue.Queue(maxsize=200)
        # Larger buffer for STT so brief backpressure does not drop audio before Deepgram.
        self._stt_audio_queue: queue.Queue = queue.Queue(maxsize=2000)

        self.audio_capture = AudioCapture(
            sample_rate=config.audio.sample_rate,
            channels=config.audio.channels,
        )
        self.audio_capture.add_consumer(self._vad_audio_queue)
        self.audio_capture.add_consumer(self._stt_audio_queue)

        self.vad = VADDetector(
            sample_rate=config.audio.sample_rate,
            aggressiveness=config.vad.aggressiveness,
            ring_buffer_size=config.vad.ring_buffer_size,
            speech_start_frames=config.vad.speech_start_frames,
        )

        self.tts = TTSEngine(
            rate=config.tts.rate,
            volume=config.tts.volume,
            event_queue=self.event_queue,
        )
        self.llm = LLMClient(
            api_key=config.openai_api_key,
            config=config.llm,
            event_queue=self.event_queue,
            tts_engine=self.tts,
        )
        self.stt = STTClient(
            api_key=config.deepgram_api_key,
            config=config.deepgram,
            audio_queue=self._stt_audio_queue,
            event_queue=self.event_queue,
            sample_rate=config.audio.sample_rate,
            channels=config.audio.channels,
        )
        self._echo_suppress_timer: threading.Timer | None = None

        self._messages = load_history(config.history.file)

    def _cancel_echo_suppress_timer(self) -> None:
        if self._echo_suppress_timer is not None:
            self._echo_suppress_timer.cancel()
            self._echo_suppress_timer = None

    def _set_stt_echo_suppression(self, active: bool) -> None:
        self.stt.set_echo_suppression(active)

    def _schedule_stt_echo_release(self) -> None:
        """Re-open mic to Deepgram after TTS plus a short tail (room echo)."""

        self._cancel_echo_suppress_timer()
        delay_s = max(0.0, self.config.tts.echo_suppress_tail_ms / 1000.0)

        def _release() -> None:
            self._set_stt_echo_suppression(False)
            self._echo_suppress_timer = None

        timer = threading.Timer(delay_s, _release)
        timer.daemon = True
        self._echo_suppress_timer = timer
        timer.start()

    def run(self) -> None:
        """Start worker threads and block in the main event loop."""

        self.tts.start()
        self.llm.start()
        self.audio_capture.start()
        self.stt.start()

        self._vad_thread = threading.Thread(
            target=self._vad_loop,
            daemon=True,
            name="VADLoop",
        )
        self._vad_thread.start()

        self._display_status("Ready — start talking! (Ctrl-C to quit)")

        try:
            self._event_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self._shutdown()

    def _vad_loop(self) -> None:
        """Convert raw VAD detections into high-level speech events."""

        while not self._shutdown_event.is_set():
            try:
                frame = self._vad_audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            speech_detected = self.vad.process_frame(frame)
            current_state = self.state_machine.state

            if current_state == State.IDLE and speech_detected:
                self.event_queue.put({"type": "SPEECH_DETECTED"})
            elif current_state == State.PROCESSING and speech_detected:
                # User can cancel before TTS starts (no speaker echo yet).
                self.event_queue.put({"type": "INTERRUPT"})
            elif (
                current_state == State.SPEAKING
                and speech_detected
                and self.config.vad.barge_in_while_speaking
            ):
                # Optional barge-in during playback (needs headphones without echo).
                self.event_queue.put({"type": "INTERRUPT"})

    def _event_loop(self) -> None:
        """Consume all worker events and drive the state machine on the main thread."""

        while True:
            try:
                event = self.event_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            event_type = event.get("type")
            logger.debug("Event: %s | State: %s", event_type, self.state_machine.state)

            if event_type == "SPEECH_DETECTED":
                self.state_machine.transition(AppEvent.SPEECH_DETECTED)
                self._display_status("Listening...")

            elif event_type == "UTTERANCE_COMPLETE":
                if self.state_machine.state != State.LISTENING:
                    logger.debug(
                        "Ignoring UTTERANCE_COMPLETE (state=%s)",
                        self.state_machine.state,
                    )
                    continue
                text = event["text"]
                self.state_machine.transition(AppEvent.UTTERANCE_COMPLETE)
                self._display_user(text)
                self._handle_utterance(text)

            elif event_type == "FIRST_TTS_CHUNK":
                # Half-duplex STT: mute mic to Deepgram while TTS plays (unless barge-in + headphones).
                if not self.config.vad.barge_in_while_speaking:
                    self._cancel_echo_suppress_timer()
                    self._set_stt_echo_suppression(True)
                self.state_machine.transition(AppEvent.FIRST_TTS_CHUNK)
                self.vad.reset()
                self._display_status("Speaking...")

            elif event_type == "LLM_RESPONSE_READY":
                if event.get("gen_id") == self._gen_id:
                    self._pending_response_by_gen[self._gen_id] = event.get("response", "")

            elif event_type == "TTS_COMPLETE":
                if event.get("gen_id") == self._gen_id:
                    self.state_machine.transition(AppEvent.TTS_COMPLETE)
                    response_text = self._pending_response_by_gen.pop(self._gen_id, "")
                    if response_text:
                        self._messages = append_message(
                            self.config.history.file,
                            "assistant",
                            response_text,
                            self.config.history.max_messages_in_context,
                        )
                    self.vad.reset()
                    if not self.config.vad.barge_in_while_speaking:
                        self._schedule_stt_echo_release()
                    self._display_status("Ready")

            elif event_type == "INTERRUPT":
                if self.state_machine.state in (State.SPEAKING, State.PROCESSING):
                    interrupted_gen = self._gen_id
                    self._gen_id += 1
                    self._pending_response_by_gen.pop(interrupted_gen, None)
                    self.tts.interrupt()
                    self.llm.cancel()
                    self.vad.reset()
                    self._cancel_echo_suppress_timer()
                    self._set_stt_echo_suppression(False)
                    self.state_machine.transition(AppEvent.INTERRUPT)
                    self._display_status("Interrupted — listening...")

            elif event_type == "TRANSCRIPT_INTERIM":
                if self.debug:
                    logger.debug("[STT partial] %s", event.get("text", ""))

            elif event_type == "ERROR":
                logger.error("Error event: %s", event.get("error"))
                print(f"  Error: {event.get('error')}")
                self._cancel_echo_suppress_timer()
                self._set_stt_echo_suppression(False)
                self.state_machine.transition(AppEvent.ERROR)
                self._display_status("Ready (after error)")

            elif event_type == "SHUTDOWN":
                break

    def _handle_utterance(self, text: str) -> None:
        """Persist user input, prepare the prompt, and submit a new LLM request."""

        self._messages = append_message(
            self.config.history.file,
            "user",
            text,
            self.config.history.max_messages_in_context,
        )

        self._gen_id += 1
        self.tts.set_generation(self._gen_id)

        messages = [{"role": "system", "content": self.config.system_prompt}]
        messages.extend(self._messages[-self.config.history.max_messages_in_context :])
        self.llm.submit(messages, self._gen_id)

    def _display_status(self, status: str) -> None:
        print(f"  [{status}]")

    def _display_user(self, text: str) -> None:
        print(f"\n[STT] {text}")

    def _shutdown(self) -> None:
        """Stop all workers in reverse start order and exit cleanly."""

        self._cancel_echo_suppress_timer()
        self._set_stt_echo_suppression(False)
        self._shutdown_event.set()
        if self._vad_thread is not None:
            self._vad_thread.join(timeout=1)
        self.stt.stop()
        self.audio_capture.stop()
        self.llm.stop()
        self.tts.stop()
        print("\nGoodbye!")
