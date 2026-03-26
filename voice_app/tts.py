"""
voice_app/tts.py — Deepgram TTS WebSocket + PyAudio output pipeline.

Architecture
============

  LLM worker
    │  speak(text, gen_id) / finish(gen_id)
    ▼
  _tts_queue  ──────────────────────────────────────┐
                                                     │
  _run_tts_ws() worker thread                        │
    │  send_text(chunk)  →  Deepgram TTS WS          │
    │  flush()  at turn end (≤20/60 s limit)         │
    │                                                 │
    │  on_audio(pcm_bytes)  [Deepgram SDK thread]    │
    ▼                                                 │
  _playback_queue  (bounded, ≤100 frames ≈ 2 s)     │
    │                                                 │
    ▼  _pyaudio_callback()  [real-time, must not block]
  PyAudio output stream → Speakers
    │  (if AEC enabled)
    ▼
  SpeakerReferenceBuffer.push(monotonic_ns, pcm)
    │  consumed by AECProcessor in AudioCapture
    ▼
  echo-cancelled mic frames → VAD + STT

TTS_COMPLETE event
------------------
Emitted when Deepgram fires the Flushed event, i.e. all synthesised PCM for
the turn has been sent to _playback_queue.  A few frames may still be playing
at that moment; echo_suppress_tail_ms (non-AEC path) covers the residual tail.

Public API (identical to the original pyttsx3/say implementation)
-----------------------------------------------------------------
  speak(text, gen_id)     — queue a sentence chunk
  finish(gen_id)          — signal end of assistant turn → triggers Flush
  interrupt()             — stop speech and clear queues/ref-buffer immediately
  set_generation(gen_id)  — start a new generation, clear interrupt flag
  start()                 — open PyAudio stream + launch worker thread
  stop()                  — shut down all components
"""

import logging
import queue
import threading
import time

import pyaudio

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Frame constants — must match audio_capture.py and aec.py
# ---------------------------------------------------------------------------

# 20 ms frame at 16 kHz
FRAME_SIZE = 320
# linear16 mono: 2 bytes per sample → 640 bytes per frame
FRAME_BYTES = FRAME_SIZE * 2
# Bounded playback queue: 100 frames ≈ 2 s of audio.
# Keeps memory finite; lets Deepgram WS listener block on a full queue (TCP backpressure)
# instead of growing the queue unboundedly.
PLAYBACK_QUEUE_MAXSIZE = 100


class TTSEngine:
    """
    Text-to-speech engine backed by a persistent Deepgram TTS WebSocket and
    a PyAudio output stream with a real-time callback.

    When no api_key is supplied (test environments), the engine falls back to a
    silent no-op loop that still emits TTS_COMPLETE so the orchestrator state
    machine advances normally.
    """

    # Sentinel placed in _tts_queue to mark the end of an assistant turn.
    _FINISH_MARKER = object()

    def __init__(
        self,
        rate: int,
        volume: float,
        event_queue: "queue.Queue | None" = None,
        *,
        api_key: str = "",
        voice: str = "aura-2-thalia-en",
        ref_buffer=None,
    ) -> None:
        """
        Parameters
        ----------
        rate:        Speech rate (not used by Deepgram TTS; kept for API compat).
        volume:      Playback volume (not used by Deepgram TTS; kept for API compat).
        event_queue: Destination for {"type": "TTS_COMPLETE", "gen_id": N} events.
        api_key:     Deepgram API key.  Empty string triggers silent/no-op mode.
        voice:       Deepgram TTS voice model (e.g. "aura-2-thalia-en").
        ref_buffer:  SpeakerReferenceBuffer shared with the AEC pipeline.
                     None when aec.enabled is False.
        """
        self.rate = rate
        self.volume = volume
        self.event_queue = event_queue
        self._api_key = api_key
        self._voice = voice
        self._ref_buffer = ref_buffer  # SpeakerReferenceBuffer | None

        # Queue of (text | _FINISH_MARKER, gen_id) items produced by the LLM worker
        self._tts_queue: queue.Queue = queue.Queue()
        # Bounded PCM frame queue consumed by the PyAudio output callback
        self._playback_queue: queue.Queue = queue.Queue(maxsize=PLAYBACK_QUEUE_MAXSIZE)

        self._interrupt_flag = threading.Event()
        self._stop_event = threading.Event()
        self._current_gen_id: int = -1
        # Tracks the gen_id active when Flush was sent; read in the Flushed callback.
        self._flushed_gen_id: int = -1

        self._thread: threading.Thread | None = None
        self._pa: pyaudio.PyAudio | None = None
        self._output_stream = None
        # Current Deepgram TTS websocket connection (used by interrupt() to send Clear).
        self._ws_conn = None

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def start(self) -> None:
        """
        Open the PyAudio output stream (callback-driven) and launch the TTS
        worker thread that maintains the Deepgram WebSocket.
        """
        self._stop_event.clear()

        # Open a non-blocking output stream.  PyAudio calls _pyaudio_callback
        # every FRAME_SIZE samples to fetch the next audio chunk.
        self._pa = pyaudio.PyAudio()
        self._output_stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16_000,
            output=True,
            frames_per_buffer=FRAME_SIZE,
            stream_callback=self._pyaudio_callback,
        )
        self._output_stream.start_stream()

        self._thread = threading.Thread(
            target=self._run_tts_ws,
            daemon=True,
            name="TTSEngine",
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal all workers to stop and release audio resources."""
        self._stop_event.set()
        self._interrupt_flag.set()

        if self._thread is not None:
            self._thread.join(timeout=3)

        if self._output_stream is not None:
            try:
                self._output_stream.stop_stream()
                self._output_stream.close()
            except Exception:
                pass
            self._output_stream = None

        if self._pa is not None:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None

    def set_generation(self, gen_id: int) -> None:
        """
        Advance to a new LLM generation.

        Clears the interrupt flag so the next turn's chunks are processed
        normally.  Called by the orchestrator before submitting a new LLM request.
        """
        self._current_gen_id = gen_id
        self._interrupt_flag.clear()

    def speak(self, text: str, gen_id: int) -> None:
        """Queue a sentence chunk for speech synthesis."""
        self._tts_queue.put((text, gen_id))

    def finish(self, gen_id: int) -> None:
        """
        Signal that no more text chunks are coming for this generation.
        The worker thread will respond by sending Flush to Deepgram, which
        finalises synthesis and ultimately triggers the TTS_COMPLETE event.
        """
        self._tts_queue.put((self._FINISH_MARKER, gen_id))

    def interrupt(self) -> None:
        """
        Stop current speech immediately.

        Step-by-step:
          1. Set _interrupt_flag so the WS worker and audio callback skip
             further work for the current generation.
          2. Drain _tts_queue (no more chunks will be sent to Deepgram).
          3. Drain _playback_queue (speaker goes silent on next callback tick).
          4. Clear the AEC reference buffer (stale reference frames from the
             interrupted turn must not pollute the next turn's echo cancellation).
          5. Send Clear to the Deepgram TTS WS to abort in-progress synthesis
             and prevent additional PCM arriving after the interrupt.
        """
        self._interrupt_flag.set()
        self._drain_tts_queue()
        self._drain_playback_queue()

        if self._ref_buffer is not None:
            self._ref_buffer.clear()

        conn = self._ws_conn
        if conn is not None:
            try:
                conn.clear()
            except Exception:
                pass

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _drain_tts_queue(self) -> None:
        """Discard all pending (text, gen_id) items from the TTS queue."""
        while not self._tts_queue.empty():
            try:
                self._tts_queue.get_nowait()
            except queue.Empty:
                break

    def _drain_playback_queue(self) -> None:
        """Discard all buffered PCM frames — silences the speaker immediately."""
        while not self._playback_queue.empty():
            try:
                self._playback_queue.get_nowait()
            except queue.Empty:
                break

    def _pyaudio_callback(self, in_data, frame_count, time_info, status):
        """
        Real-time PyAudio output callback.  MUST NEVER BLOCK.

        Pops one FRAME_BYTES (640-byte / 320-sample linear16) chunk from
        _playback_queue.  On underrun (queue empty), returns a silence frame
        so the output stream never stalls waiting for data.

        AEC reference tap
        -----------------
        If a SpeakerReferenceBuffer is wired (aec.enabled = True), every frame
        that leaves the callback is pushed into the buffer with its
        time.monotonic_ns() timestamp.  The AECProcessor in AudioCapture then
        looks up the frame that was playing at the moment a mic frame was captured.
        Both sides use the same clock, enabling time-aligned echo cancellation.
        """
        try:
            frame = self._playback_queue.get_nowait()
        except queue.Empty:
            # Silence on underrun — this is the expected no-op during silence
            frame = bytes(FRAME_BYTES)

        # Feed AEC reference buffer (only when aec.enabled = True)
        if self._ref_buffer is not None:
            self._ref_buffer.push(time.monotonic_ns(), frame)

        return (frame, pyaudio.paContinue)

    # -----------------------------------------------------------------------
    # TTS WebSocket worker
    # -----------------------------------------------------------------------

    def _run_tts_ws(self) -> None:
        """
        Main TTS worker thread entry point.

        Behaviour:
          • No api_key → fall back to _run_silent_loop (for test environments).
          • Otherwise → maintain a persistent Deepgram TTS WebSocket session.
            On transient errors the session is recreated with a short backoff
            (reconnect on protocol ambiguity, as advised by the plan).
        """
        if not self._api_key:
            self._run_silent_loop()
            return

        while not self._stop_event.is_set():
            try:
                self._run_ws_session()
            except Exception as exc:
                logger.error("[TTS] WebSocket session error: %s", exc)
                if not self._stop_event.is_set():
                    # Brief backoff before reconnect to avoid tight error loops
                    time.sleep(0.5)

    def _run_ws_session(self) -> None:
        """
        One Deepgram TTS WebSocket session lifecycle: connect → stream → close.

        The session is kept alive across multiple assistant turns (one session
        per conversation).  It is torn down only when the worker exits or an
        unrecoverable error forces recreation.

        Flush semantics
        ---------------
        Deepgram imposes a hard limit of 20 Flush messages per 60 seconds.
        To respect this:
          • send_text() is called once per sentence chunk with no Flush in between.
          • flush() is called exactly once per assistant turn, when _FINISH_MARKER
            arrives from the LLM worker.
          • clear() is called by interrupt() to abort in-progress synthesis.

        TTS_COMPLETE timing
        -------------------
        Deepgram fires a Flushed event after all PCM for the flushed turn has
        been sent.  We emit TTS_COMPLETE at that moment.  A few frames may still
        be draining from _playback_queue at that time; echo_suppress_tail_ms
        (non-AEC path) covers the residual speaker tail.
        """
        # Import here so the module loads cleanly even if deepgram-sdk is absent.
        from deepgram import DeepgramClient, SpeakWebSocketEvents, SpeakWebSocketOptions

        dg = DeepgramClient(api_key=self._api_key)
        conn = dg.speak.websocket.v("1")

        # ------------------------------------------------------------------
        # Event callbacks — fired by the Deepgram SDK internal listener thread
        # ------------------------------------------------------------------

        def on_audio(data, **kwargs):
            """
            Buffer incoming raw PCM from Deepgram.
            Discards frames if we are in an interrupted state so stale audio
            from a Cleared turn does not leak into the next turn's playback.
            """
            if self._interrupt_flag.is_set():
                return  # discard; the turn was interrupted before this arrived
            self._tts_ws_on_audio(data)

        def on_flushed(flushed, **kwargs):
            """
            Deepgram signals that all PCM for the flushed turn has been sent.
            Emit TTS_COMPLETE unless the turn was interrupted after Flush.
            """
            gen = self._flushed_gen_id
            if not self._interrupt_flag.is_set() and self.event_queue is not None:
                self.event_queue.put({"type": "TTS_COMPLETE", "gen_id": gen})

        def on_error(error, **kwargs):
            logger.error("[TTS WS] error: %s", error)

        conn.on(SpeakWebSocketEvents.AudioData, on_audio)
        conn.on(SpeakWebSocketEvents.Flushed, on_flushed)
        conn.on(SpeakWebSocketEvents.Error, on_error)

        # Linear16 mono 16 kHz — matches the entire audio pipeline
        options = SpeakWebSocketOptions(
            model=self._voice,
            encoding="linear16",
            sample_rate=16_000,
        )
        if not conn.start(options):
            raise RuntimeError("Deepgram TTS WebSocket failed to start")

        self._ws_conn = conn
        try:
            while not self._stop_event.is_set():
                try:
                    item = self._tts_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                text, gen_id = item

                # Discard items from a superseded generation (after interrupt)
                if gen_id != self._current_gen_id:
                    continue

                if text is self._FINISH_MARKER:
                    # End of assistant turn — flush Deepgram TTS to finalise synthesis.
                    # We save gen_id before the async Flushed callback fires.
                    if not self._interrupt_flag.is_set():
                        self._flushed_gen_id = gen_id
                        try:
                            conn.flush()
                        except Exception as exc:
                            logger.warning("[TTS] flush() failed: %s", exc)
                            # Emit TTS_COMPLETE directly so the state machine doesn't stall
                            if self.event_queue is not None:
                                self.event_queue.put({"type": "TTS_COMPLETE", "gen_id": gen_id})
                    continue

                if self._interrupt_flag.is_set():
                    continue

                logger.info("[TTS] %s", text)
                try:
                    conn.send_text(text)
                except Exception as exc:
                    logger.error("[TTS] send_text failed: %s", exc)
                    raise  # propagate to force session recreation via _run_tts_ws

        finally:
            self._ws_conn = None
            try:
                conn.finish()
            except Exception:
                pass

    def _tts_ws_on_audio(self, data: bytes) -> None:
        """
        Slice Deepgram PCM audio into FRAME_BYTES chunks and push to _playback_queue.

        Partial final chunks are zero-padded to FRAME_BYTES so the PyAudio
        callback always pops exact-size frames.

        _playback_queue is bounded (PLAYBACK_QUEUE_MAXSIZE).  When it is full
        we block here inside the Deepgram SDK listener thread — this is
        intentional TCP backpressure rather than unbounded memory growth.
        Overshooting the bound logs a warning and drops the frame.
        """
        for offset in range(0, len(data), FRAME_BYTES):
            chunk = data[offset : offset + FRAME_BYTES]
            # Pad the last partial frame (common at utterance boundaries)
            if len(chunk) < FRAME_BYTES:
                chunk = chunk + bytes(FRAME_BYTES - len(chunk))
            try:
                self._playback_queue.put(chunk, block=True, timeout=1.0)
            except queue.Full:
                logger.warning("[TTS] playback queue full; dropping audio frame")

    def _run_silent_loop(self) -> None:
        """
        No-op fallback for test environments without a Deepgram API key.

        Processes _tts_queue items normally (respects gen_id, interrupt flag)
        and emits TTS_COMPLETE on _FINISH_MARKER so the orchestrator state
        machine progresses exactly as it does in production.
        """
        while not self._stop_event.is_set():
            try:
                item = self._tts_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            text, gen_id = item
            if gen_id != self._current_gen_id:
                continue
            if text is self._FINISH_MARKER:
                if not self._interrupt_flag.is_set() and self.event_queue is not None:
                    self.event_queue.put({"type": "TTS_COMPLETE", "gen_id": gen_id})
