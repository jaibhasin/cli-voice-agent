"""
voice_app/orchestrator.py — Coordinate all background workers and drive the
conversation lifecycle via a central event queue + state machine.

AEC vs half-duplex path
=======================
When config.aec.enabled is True:
  • SpeakerReferenceBuffer and AECProcessor are created and wired into
    TTSEngine (speaker reference tap) and AudioCapture (mic cleanup).
  • STT receives echo-free audio → set_echo_suppression / silence injection
    are disabled.
  • Residual echo is filtered with a transcript-aware guard:
    – VAD hits during SPEAKING are treated as provisional only.
    – STT interim/final text that matches the assistant's own response is dropped.
    – Non-echo STT text still triggers barge-in and normal listening.
  • All _echo_suppress_timer / _set_stt_echo_suppression / _schedule_stt_echo_release
    call sites are skipped.

When config.aec.enabled is False (default):
  • Original half-duplex behaviour is preserved exactly:
    – FIRST_TTS_CHUNK mutes the STT mic path.
    – TTS_COMPLETE schedules re-opening the mic path after echo_suppress_tail_ms.
    – VAD INTERRUPT in SPEAKING state requires barge_in_while_speaking = True.
"""

import logging
import queue
import threading

from voice_app.audio_capture import AudioCapture
from voice_app.config import AppConfig
from voice_app.echo_guard import EchoTranscriptGuard
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
        self._echo_guard = EchoTranscriptGuard(
            cooldown_ms=config.aec.residual_echo_guard_ms,
            interrupted_cooldown_ms=config.aec.interrupted_echo_guard_ms,
        )

        self._vad_audio_queue: queue.Queue = queue.Queue(maxsize=200)
        # Larger buffer for STT so brief backpressure does not drop audio before Deepgram.
        self._stt_audio_queue: queue.Queue = queue.Queue(maxsize=2000)

        # ------------------------------------------------------------------
        # AEC setup — must happen before constructing TTSEngine / AudioCapture
        # so that the shared SpeakerReferenceBuffer is available to both.
        # ------------------------------------------------------------------
        _ref_buffer = None    # SpeakerReferenceBuffer | None
        _aec_processor = None  # AECProcessor | None

        if config.aec.enabled:
            # Local import so the module loads cleanly when speexdsp is absent
            # and aec.enabled is False (the common case).
            from voice_app.aec import AECProcessor, SpeakerReferenceBuffer

            # 320 samples = one 20 ms frame at 16 kHz (matches FRAME_SIZE in aec.py)
            _ref_buffer = SpeakerReferenceBuffer(
                maxlen=config.aec.ref_buffer_frames, frame_size=320
            )
            _aec_processor = AECProcessor(
                config=config.aec,
                sample_rate=config.audio.sample_rate,
                frame_size=320,
                ref_buffer=_ref_buffer,
            )

        # ------------------------------------------------------------------
        # Audio capture — with optional AEC processor
        # ------------------------------------------------------------------
        self.audio_capture = AudioCapture(
            sample_rate=config.audio.sample_rate,
            channels=config.audio.channels,
            aec_processor=_aec_processor,
        )
        self.audio_capture.add_consumer(self._vad_audio_queue)
        self.audio_capture.add_consumer(self._stt_audio_queue)

        self.vad = VADDetector(
            sample_rate=config.audio.sample_rate,
            aggressiveness=config.vad.aggressiveness,
            ring_buffer_size=config.vad.ring_buffer_size,
            speech_start_frames=config.vad.speech_start_frames,
        )

        # ------------------------------------------------------------------
        # TTS — uses local piper-tts subprocess; receives ref_buffer for AEC
        # ------------------------------------------------------------------
        self.tts = TTSEngine(
            rate=config.tts.rate,
            volume=config.tts.volume,
            event_queue=self.event_queue,
            model_path=config.tts.model_path,
            piper_bin=config.tts.piper_bin,
            ref_buffer=_ref_buffer,
        )

        self.llm = LLMClient(
            api_key=config.openai_api_key,
            config=config.llm,
            event_queue=self.event_queue,
            tts_engine=self.tts,
        )

        # ------------------------------------------------------------------
        # STT — with aec_enabled flag; when True, silence injection is a no-op
        # ------------------------------------------------------------------
        self.stt = STTClient(
            api_key=config.deepgram_api_key,
            config=config.deepgram,
            audio_queue=self._stt_audio_queue,
            event_queue=self.event_queue,
            sample_rate=config.audio.sample_rate,
            channels=config.audio.channels,
            aec_enabled=config.aec.enabled,
        )

        # Half-duplex echo-suppress timer (only used when aec.enabled is False)
        self._echo_suppress_timer: threading.Timer | None = None

        self._messages = load_history(config.history.file)

    # -----------------------------------------------------------------------
    # Half-duplex echo suppression helpers
    # (only called when config.aec.enabled is False)
    # -----------------------------------------------------------------------

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

    # -----------------------------------------------------------------------
    # Main run loop
    # -----------------------------------------------------------------------

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
        """
        Convert raw VAD detections into high-level speech events.

        AEC path:
          In SPEAKING state, raw VAD hits are treated as provisional only.
          Residual speaker leakage can still trip the detector, so barge-in is
          confirmed by a non-echo transcript on the main thread.

        Half-duplex path:
          In SPEAKING state, INTERRUPT requires barge_in_while_speaking = True
          (safe only with headphones, since there is no speaker signal removal).
        """
        while not self._shutdown_event.is_set():
            try:
                frame = self._vad_audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            speech_detected = self.vad.process_frame(frame)
            current_state = self.state_machine.state

            if current_state == State.IDLE and speech_detected:
                if self.config.aec.enabled and self._echo_guard.should_gate_vad():
                    continue
                self.event_queue.put({"type": "SPEECH_DETECTED"})

            elif current_state == State.PROCESSING and speech_detected:
                # User can cancel before TTS starts (no speaker echo yet)
                self.event_queue.put({"type": "INTERRUPT"})

            elif current_state == State.SPEAKING and speech_detected:
                # With AEC enabled, wait for transcript evidence before barge-in.
                # Half-duplex still uses the explicit barge-in flag.
                if not self.config.aec.enabled and self.config.vad.barge_in_while_speaking:
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

            elif event_type == "ASSISTANT_RESPONSE_CHUNK":
                if event.get("gen_id") == self._gen_id:
                    self._echo_guard.note_tts_chunk(
                        gen_id=self._gen_id,
                        text=event.get("text", ""),
                    )

            elif event_type == "UTTERANCE_COMPLETE":
                text = event["text"]
                if self._is_probable_echo(text):
                    self._discard_echo_listening_state()
                    logger.info("Ignoring probable echo utterance: %s", text)
                    continue

                if self.state_machine.state == State.SPEAKING and self.config.aec.enabled:
                    self._interrupt_current_turn()

                if self.state_machine.state == State.IDLE:
                    self.state_machine.transition(AppEvent.SPEECH_DETECTED)
                    self._display_status("Listening...")

                if self.state_machine.state != State.LISTENING:
                    logger.debug(
                        "Ignoring UTTERANCE_COMPLETE (state=%s)",
                        self.state_machine.state,
                    )
                    continue
                self.state_machine.transition(AppEvent.UTTERANCE_COMPLETE)
                self._display_user(text)
                self._handle_utterance(text)

            elif event_type == "FIRST_TTS_CHUNK":
                # Half-duplex: mute mic to STT while TTS plays.
                # Skipped when AEC is active — AEC handles echo removal directly.
                if not self.config.aec.enabled and not self.config.vad.barge_in_while_speaking:
                    self._cancel_echo_suppress_timer()
                    self._set_stt_echo_suppression(True)
                self.state_machine.transition(AppEvent.FIRST_TTS_CHUNK)
                self.vad.reset()
                self._display_status("Speaking...")

            elif event_type == "LLM_RESPONSE_READY":
                if event.get("gen_id") == self._gen_id:
                    response_text = event.get("response", "")
                    self._pending_response_by_gen[self._gen_id] = response_text
                    self._echo_guard.note_response_ready(self._gen_id, response_text)

            elif event_type == "TTS_COMPLETE":
                if event.get("gen_id") == self._gen_id:
                    self._echo_guard.note_tts_complete(self._gen_id)
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
                    # Schedule echo release only on half-duplex path
                    if not self.config.aec.enabled and not self.config.vad.barge_in_while_speaking:
                        self._schedule_stt_echo_release()
                    self._display_status("Ready")

            elif event_type == "INTERRUPT":
                self._interrupt_current_turn()

            elif event_type == "TRANSCRIPT_INTERIM":
                text = event.get("text", "")
                if self._is_probable_echo(text):
                    logger.debug("Ignoring probable echo interim transcript: %s", text)
                    continue

                if self.state_machine.state == State.SPEAKING and self.config.aec.enabled:
                    self._interrupt_current_turn()
                elif self.state_machine.state == State.IDLE:
                    self.state_machine.transition(AppEvent.SPEECH_DETECTED)
                    self._display_status("Listening...")

                if self.debug:
                    logger.debug("[STT partial] %s", text)

            elif event_type == "ERROR":
                logger.error("Error event: %s", event.get("error"))
                print(f"  Error: {event.get('error')}")
                if not self.config.aec.enabled:
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
        self._echo_guard.start_generation(self._gen_id)
        self.tts.set_generation(self._gen_id)

        messages = [{"role": "system", "content": self.config.system_prompt}]
        messages.extend(self._messages[-self.config.history.max_messages_in_context :])
        self.llm.submit(messages, self._gen_id)

    def _interrupt_current_turn(self) -> None:
        if self.state_machine.state not in (State.SPEAKING, State.PROCESSING):
            return

        interrupted_gen = self._gen_id
        self._gen_id += 1
        self._pending_response_by_gen.pop(interrupted_gen, None)
        self._echo_guard.note_interrupt(interrupted_gen)
        self.tts.interrupt()
        self.llm.cancel()
        self.vad.reset()
        if not self.config.aec.enabled:
            self._cancel_echo_suppress_timer()
            self._set_stt_echo_suppression(False)
        self.state_machine.transition(AppEvent.INTERRUPT)
        self._display_status("Interrupted — listening...")

    def _is_probable_echo(self, text: str) -> bool:
        if not self.config.aec.enabled:
            return False

        return self._echo_guard.is_probable_echo(
            text,
            speaking_active=self.state_machine.state == State.SPEAKING,
        )

    def _discard_echo_listening_state(self) -> None:
        if self.state_machine.state == State.LISTENING:
            self.state_machine.state = State.IDLE
            self.vad.reset()
            self._display_status("Ready")

    def _display_status(self, status: str) -> None:
        print(f"  [{status}]")

    def _display_user(self, text: str) -> None:
        print(f"\n[STT] {text}")

    def _shutdown(self) -> None:
        """Stop all workers in reverse start order and exit cleanly."""
        # Release half-duplex STT suppression (no-op on AEC path)
        if not self.config.aec.enabled:
            self._cancel_echo_suppress_timer()
            self._set_stt_echo_suppression(False)

        self._shutdown_event.set()
        if self._vad_thread is not None:
            self._vad_thread.join(timeout=1)
        self.stt.stop()
        self.audio_capture.stop()
        self.llm.stop()
        self.tts.stop()
        self._echo_guard.clear()
        print("\nGoodbye!")
