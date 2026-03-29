"""
voice_app/tts.py — piper-tts local synthesis + PyAudio output pipeline.

Architecture
============

  LLM worker
    │  speak(text, gen_id) / finish(gen_id)
    ▼
  _tts_queue  ──────────────────────────────────────────┐
                                                         │
  _run_tts_worker() worker thread                        │
    │  accumulates text chunks per turn                  │
    │  on _FINISH_MARKER: calls _synthesise(text)        │
    │    → spawns:  piper --model … --output_raw         │
    │    → reads stdout (raw linear16 PCM, 16 kHz mono)  │
    │    → slices into FRAME_BYTES chunks                 │
    ▼                                                     │
  _playback_queue  (bounded, ≤100 frames ≈ 2 s)          │
    │                                                     │
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
Emitted as soon as all synthesised PCM for the turn has been pushed to
_playback_queue (i.e. the piper subprocess has exited and all its output has
been queued).  A few frames may still be playing at that moment;
echo_suppress_tail_ms (non-AEC path) covers the residual tail.

Public API (identical to the Deepgram implementation)
------------------------------------------------------
  speak(text, gen_id)     — queue a sentence chunk
  finish(gen_id)          — signal end of assistant turn → triggers synthesis
  interrupt()             — stop speech and clear queues/ref-buffer immediately
  set_generation(gen_id)  — start a new generation, clear interrupt flag
  start()                 — open PyAudio stream + launch worker thread
  stop()                  — shut down all components

Configuration
-------------
  model_path:  Absolute path to a piper .onnx voice model.
               Defaults to $PIPER_MODEL env-var, then raises RuntimeError at
               synthesis time if still not set.
  piper_bin:   Path to the piper executable (default: "piper").
"""

import logging
import os
import queue
import subprocess
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
PLAYBACK_QUEUE_MAXSIZE = 100


class TTSEngine:
    """
    Text-to-speech engine backed by a local piper-tts subprocess and a
    PyAudio output stream with a real-time callback.

    piper is invoked once per assistant turn with the full concatenated text
    of that turn.  Its stdout (raw linear16 PCM, 16 kHz mono) is read in a
    tight loop, sliced into FRAME_BYTES chunks, and pushed to _playback_queue.

    When no model_path is supplied (test environments), the engine falls back
    to a silent no-op loop that still emits TTS_COMPLETE so the orchestrator
    state machine advances normally.
    """

    # Sentinel placed in _tts_queue to mark the end of an assistant turn.
    _FINISH_MARKER = object()

    def __init__(
        self,
        rate: int,
        volume: float,
        event_queue: "queue.Queue | None" = None,
        *,
        model_path: str = "",
        piper_bin: str = "piper",
        ref_buffer=None,
    ) -> None:
        """
        Parameters
        ----------
        rate:        Speech rate (not used by piper; kept for API compat).
        volume:      Playback volume (not used by piper-tts; kept for API compat).
        event_queue: Destination for {"type": "TTS_COMPLETE", "gen_id": N} events.
        model_path:  Path to the piper .onnx voice model file.
                     Falls back to the PIPER_MODEL environment variable.
                     Empty string triggers silent/no-op mode.
        piper_bin:   Name or path of the piper executable.
        ref_buffer:  SpeakerReferenceBuffer shared with the AEC pipeline.
                     None when aec.enabled is False.
        """
        self.rate = rate
        self.volume = volume
        self.event_queue = event_queue
        self._model_path = model_path or os.environ.get("PIPER_MODEL", "")
        self._piper_bin = piper_bin
        self._ref_buffer = ref_buffer  # SpeakerReferenceBuffer | None

        # Queue of (text | _FINISH_MARKER, gen_id) items from the LLM worker
        self._tts_queue: queue.Queue = queue.Queue()
        # Bounded PCM frame queue consumed by PyAudio output callback
        self._playback_queue: queue.Queue = queue.Queue(maxsize=PLAYBACK_QUEUE_MAXSIZE)

        self._interrupt_flag = threading.Event()
        self._stop_event = threading.Event()
        self._current_gen_id: int = -1

        self._thread: threading.Thread | None = None
        self._pa: pyaudio.PyAudio | None = None
        self._output_stream = None
        self._played_frames = 0
        self._silence_frames = 0
        self._synth_frames = 0
        self._last_diag_log_ns = 0

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def start(self) -> None:
        """
        Open the PyAudio output stream (callback-driven) and launch the TTS
        worker thread.
        """
        self._stop_event.clear()

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
            target=self._run_tts_worker,
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
        The worker thread will synthesise the accumulated text with piper
        and emit TTS_COMPLETE when done.
        """
        self._tts_queue.put((self._FINISH_MARKER, gen_id))

    def interrupt(self) -> None:
        """
        Stop current speech immediately.

        Step-by-step:
          1. Set _interrupt_flag so the worker and audio callback skip
             further work for the current generation.
          2. Drain _tts_queue (no more chunks will be synthesised).
          3. Drain _playback_queue (speaker goes silent on next callback tick).
          4. Clear the AEC reference buffer (stale reference frames from the
             interrupted turn must not pollute the next turn's echo cancellation).
        """
        self._interrupt_flag.set()
        self._drain_tts_queue()
        self._drain_playback_queue()

        if self._ref_buffer is not None:
            self._ref_buffer.clear()

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
            frame = bytes(FRAME_BYTES)
            self._silence_frames += 1

        if self._ref_buffer is not None:
            self._ref_buffer.push(time.monotonic_ns(), frame)
        self._played_frames += 1
        self._maybe_log_playback_diag()

        return (frame, pyaudio.paContinue)

    def _maybe_log_playback_diag(self) -> None:
        now_ns = time.monotonic_ns()
        if now_ns - self._last_diag_log_ns < 2_000_000_000:
            return
        self._last_diag_log_ns = now_ns
        logger.info(
            "[TTS_DIAG] played_frames=%d synth_frames=%d silence_frames=%d playback_q=%d",
            self._played_frames,
            self._synth_frames,
            self._silence_frames,
            self._playback_queue.qsize(),
        )

    # -----------------------------------------------------------------------
    # TTS worker
    # -----------------------------------------------------------------------

    def _run_tts_worker(self) -> None:
        """
        Main TTS worker thread entry point.

        No model_path → fall back to _run_silent_loop (for test environments).
        Otherwise, accumulate text chunks per turn and synthesise with piper
        when _FINISH_MARKER arrives.
        """
        if not self._model_path:
            self._run_silent_loop()
            return

        # Accumulate sentence chunks until the end of each turn
        pending_text: list[str] = []
        pending_gen_id: int = -1

        while not self._stop_event.is_set():
            try:
                item = self._tts_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            text, gen_id = item

            # Discard items from a superseded generation (after interrupt)
            if gen_id != self._current_gen_id:
                pending_text.clear()
                pending_gen_id = -1
                continue

            if text is self._FINISH_MARKER:
                if not self._interrupt_flag.is_set() and pending_text:
                    full_text = " ".join(pending_text)
                    logger.info("[TTS] synthesising %d chars for gen_id=%d", len(full_text), gen_id)
                    self._synthesise(full_text, gen_id)
                pending_text.clear()
                pending_gen_id = -1
                continue

            if self._interrupt_flag.is_set():
                continue

            logger.info("[TTS] chunk: %s", text)
            pending_text.append(text)
            pending_gen_id = gen_id

    def _synthesise(self, text: str, gen_id: int) -> None:
        """
        Run piper on *text* and push the resulting PCM into _playback_queue.
        Emits TTS_COMPLETE when all frames have been queued (or on error).

        piper is invoked with:
            piper --model <model_path> --output_raw
        stdin  = UTF-8 text
        stdout = raw linear16 PCM, 16 kHz, mono (piper default output rate
                 matches the model; ensure the .onnx model is 16 kHz).
        """
        cmd = [self._piper_bin, "--model", self._model_path, "--output_raw"]
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            logger.error("[TTS] piper executable not found: %s", self._piper_bin)
            self._emit_complete(gen_id)
            return
        except Exception as exc:
            logger.error("[TTS] failed to launch piper: %s", exc)
            self._emit_complete(gen_id)
            return

        try:
            # Write text to piper's stdin and close it so piper knows it is done
            stdin_bytes = (text + "\n").encode("utf-8")
            proc.stdin.write(stdin_bytes)
            proc.stdin.close()

            # Stream PCM from piper's stdout into _playback_queue
            while not self._interrupt_flag.is_set():
                chunk = proc.stdout.read(FRAME_BYTES)
                if not chunk:
                    break  # piper finished

                # Pad partial final frame
                if len(chunk) < FRAME_BYTES:
                    chunk = chunk + bytes(FRAME_BYTES - len(chunk))

                try:
                    self._playback_queue.put(chunk, block=True, timeout=1.0)
                    self._synth_frames += 1
                except queue.Full:
                    logger.warning("[TTS] playback queue full; dropping audio frame")

            # If interrupted, kill piper so it doesn't keep writing
            if self._interrupt_flag.is_set():
                try:
                    proc.kill()
                except Exception:
                    pass
                return  # do NOT emit TTS_COMPLETE for an interrupted turn

        except Exception as exc:
            logger.error("[TTS] error reading piper output: %s", exc)
            try:
                proc.kill()
            except Exception:
                pass
        finally:
            try:
                proc.wait(timeout=2)
            except Exception:
                pass
            stderr_out = proc.stderr.read()
            if stderr_out:
                logger.debug("[TTS piper stderr] %s", stderr_out.decode(errors="replace").strip())

        self._emit_complete(gen_id)

    def _emit_complete(self, gen_id: int) -> None:
        """Emit TTS_COMPLETE if we are not interrupted and have an event queue."""
        if not self._interrupt_flag.is_set() and self.event_queue is not None:
            self.event_queue.put({"type": "TTS_COMPLETE", "gen_id": gen_id})

    def _run_silent_loop(self) -> None:
        """
        No-op fallback for test environments without a piper model.

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
