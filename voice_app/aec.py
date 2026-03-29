"""
voice_app/aec.py — Acoustic Echo Cancellation (AEC) primitives.

Architecture
============
The AEC pipeline has three participants that must share a common clock
(time.monotonic_ns):

  1. TTSEngine._pyaudio_callback  →  SpeakerReferenceBuffer.push(ts, pcm)
     Real-time PyAudio output callback records every frame that goes to
     the speaker, stamped with monotonic_ns at the moment it leaves the
     callback.

  2. AudioCapture._capture_loop   →  AECProcessor.process(mic_pcm, mic_ts)
     For each mic frame captured, the capture loop stamps it with
     monotonic_ns, then asks AECProcessor to cancel the echo using the
     reference frame that was playing at approximately that timestamp.

  3. AECProcessor                 →  SpeakerReferenceBuffer.get_frame_at(ts)
     Looks up the closest buffered reference frame and feeds it together
     with the mic frame to speexdsp.EchoCanceller.

Frame constants
===============
All audio in this pipeline is 16-bit signed little-endian mono at 16 kHz
(Deepgram's "linear16" encoding):
  FRAME_SIZE  = 320 samples  (20 ms per frame)
  FRAME_BYTES = 640 bytes    (2 bytes × 320 samples)
"""

import collections
import threading

# ---------------------------------------------------------------------------
# Public frame constants — imported by audio_capture.py and tts.py
# ---------------------------------------------------------------------------
FRAME_SIZE = 320        # samples per 20 ms frame at 16 kHz
FRAME_BYTES = FRAME_SIZE * 2  # linear16 mono: 2 bytes/sample

# A reference frame is considered stale if it is more than 6 frame durations
# (≈ 120 ms) away from the mic timestamp. This easily covers typical OS/hardware
# buffer latency (40-80ms on macOS) so we don't accidentally drop valid reference frames.
_MAX_REF_AGE_NS = int(6 * FRAME_SIZE / 16_000 * 1_000_000_000)  # ~120 000 000 ns


class SpeakerReferenceBuffer:
    """
    Thread-safe, timestamp-indexed ring buffer of speaker PCM reference frames.

    Usage pattern
    -------------
    Producer (PyAudio output callback, real-time):
        ref_buf.push(time.monotonic_ns(), pcm_frame)

    Consumer (AudioCapture capture loop, near-real-time):
        ref_frame = ref_buf.get_frame_at(mic_timestamp_ns)

    When the buffer is empty (agent is silent) or the closest match is too old,
    get_frame_at() returns a zero-filled silence frame — the correct "nothing
    to cancel" input for speexdsp.

    Parameters
    ----------
    maxlen:     Ring-buffer depth.  200 ≈ 4 s at 20 ms/frame.
    frame_size: Samples per frame (must match the audio pipeline).
    """

    def __init__(self, maxlen: int = 200, frame_size: int = FRAME_SIZE) -> None:
        self._frame_size = frame_size
        # Silence frame: zero-filled linear16 mono
        self._silence = bytes(frame_size * 2)
        # deque auto-discards the oldest entry when maxlen is reached
        self._buf: collections.deque = collections.deque(maxlen=maxlen)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Producer API
    # ------------------------------------------------------------------

    def push(self, timestamp_ns: int, frame: bytes) -> None:
        """
        Append a speaker reference frame with its playback timestamp.

        Called from the PyAudio output callback — must execute without
        blocking, hence we use a Lock (not a Condition or heavy primitive).
        """
        with self._lock:
            self._buf.append((timestamp_ns, frame))

    # ------------------------------------------------------------------
    # Consumer API
    # ------------------------------------------------------------------

    def get_frame_at(self, query_ns: int, speaker_delay_ms: int = 0) -> bytes:
        """
        Return the buffered frame whose timestamp is closest to query_ns.

        speaker_delay_ms
        ----------------
        The PyAudio output callback fires when a frame is *queued* to the
        hardware buffer, not when it physically exits the speaker.  By the time
        the microphone captures that frame the hardware + acoustic propagation
        delay has elapsed — typically 40–80 ms on macOS.

        To compensate, we subtract speaker_delay_ms from query_ns before
        searching the buffer.  This shifts the lookup *back in time* so we
        match the reference frame that was queued at the moment corresponding
        to the mic capture time, rather than the frame queued speaker_delay_ms
        *after* that moment.

        Example:
            Mic captures at t=100 ms, speaker delay = 60 ms.
            Adjusted query = 100 ms − 60 ms = 40 ms → finds the reference
            frame queued at ~40 ms, which is the one actually playing at 100 ms.

        With speaker_delay_ms = 0 (default) the behaviour is unchanged from the
        original implementation.

        Returns the silence constant when:
          • the buffer is empty (nothing has been played yet), or
          • the closest frame is more than _MAX_REF_AGE_NS from the adjusted
            query_ns (stale after an interrupt, a startup gap, or a silent turn).

        O(n) scan over the ring buffer (n ≤ 200); acceptable for 20 ms frames.
        """
        # Convert the millisecond delay to nanoseconds and shift the query
        # back in time so we find the reference frame that was *physically*
        # playing when the mic captured its frame.
        adjusted_ns = query_ns - speaker_delay_ms * 1_000_000

        with self._lock:
            if not self._buf:
                return self._silence
            # Find the frame with the minimum time distance to the adjusted query
            best_ts, best_frame = min(self._buf, key=lambda x: abs(x[0] - adjusted_ns))
            if abs(best_ts - adjusted_ns) > _MAX_REF_AGE_NS:
                return self._silence
            return best_frame

    def clear(self) -> None:
        """
        Discard all buffered frames.

        Call this on interrupt so AECProcessor does not subtract a stale
        reference (from the just-interrupted turn) from mic audio captured
        after the speaker has gone silent.
        """
        with self._lock:
            self._buf.clear()


class AECProcessor:
    """
    Thin wrapper around speexdsp.EchoCanceller that integrates with
    SpeakerReferenceBuffer for time-aligned reference lookup.

    Data flow per mic frame
    -----------------------
    1. ref = ref_buffer.get_frame_at(mic_time_ns)   — closest speaker frame
    2. cleaned = ec.process(mic_frame, ref)          — speexdsp adaptive filter

    speexdsp convergence
    --------------------
    The adaptive filter needs ~1 s of simultaneous mic + speaker activity to
    fully converge.  Before convergence the output degrades gracefully (partial
    suppression), never complete silence.

    Installation
    ------------
    Requires system libspeexdsp and the Python bindings:
      macOS:  brew install speexdsp && pip install speexdsp
      Debian: apt install libspeexdsp-dev && pip install speexdsp

    If speexdsp is not installed, AECProcessor.__init__ raises ImportError with
    install instructions.  The orchestrator never creates AECProcessor when
    aec.enabled is False, so users on the half-duplex path are unaffected.
    """

    def __init__(
        self,
        config,
        sample_rate: int = 16_000,
        frame_size: int = FRAME_SIZE,
        ref_buffer: "SpeakerReferenceBuffer | None" = None,
    ) -> None:
        """
        Parameters
        ----------
        config:      AECConfig — reads config.filter_length and
                     config.speaker_delay_ms.
        sample_rate: Audio sample rate in Hz (must be 16 000 for this pipeline).
        frame_size:  Samples per frame (must be 320 for this pipeline).
        ref_buffer:  Shared SpeakerReferenceBuffer populated by TTSEngine.
        """
        # Defer the speexdsp import so the module loads cleanly without the lib.
        try:
            import speexdsp as _speexdsp
        except ImportError as exc:
            raise ImportError(
                "speexdsp is required for AEC (aec.enabled = true).\n"
                "Install instructions:\n"
                "  macOS:  brew install speexdsp && pip install speexdsp\n"
                "  Debian: apt install libspeexdsp-dev && pip install speexdsp\n"
                "Or set aec.enabled: false in config.yaml to use the half-duplex fallback."
            ) from exc

        self._ref_buffer = ref_buffer
        # speaker_delay_ms: hardware + acoustic propagation delay between the
        # PyAudio callback timestamp and when the sound reaches the microphone.
        # Calibrate with calibrate_aec.py; 0 = let speexdsp adapt on its own.
        self._speaker_delay_ms: int = getattr(config, "speaker_delay_ms", 0)
        # Silence used when no reference buffer is wired (defensive fallback)
        self._silence = bytes(frame_size * 2)
        # EchoCanceller.create(frame_size_samples, filter_length_samples, sample_rate)
        # filter_length controls how long an acoustic tail can be cancelled;
        # 2048 samples = 128 ms at 16 kHz (covers most room echoes).
        self._ec = _speexdsp.EchoCanceller.create(
            frame_size, config.filter_length, sample_rate
        )

    def process(self, mic_frame: bytes, mic_time_ns: int) -> bytes:
        """
        Cancel speaker echo from mic_frame using a time-aligned reference.

        Parameters
        ----------
        mic_frame:    Raw 640-byte (320-sample linear16 mono) mic frame.
        mic_time_ns:  time.monotonic_ns() stamp from when the frame was read.

        Returns
        -------
        Echo-cancelled frame of the same byte length.

        Timing note
        -----------
        speaker_delay_ms is forwarded to get_frame_at() so the buffer lookup
        shifts the query back in time by the known hardware + acoustic delay.
        This ensures speexdsp receives the reference frame that was *physically
        playing* at mic_time_ns rather than the one queued at that moment.
        With speaker_delay_ms = 0 (default) there is no shift; speexdsp still
        adapts, just more slowly during the first ~1 s of overlap.
        """
        ref = (
            # Pass speaker_delay_ms so the lookup compensates for the gap
            # between when PyAudio queues a frame and when it reaches the mic.
            self._ref_buffer.get_frame_at(mic_time_ns, self._speaker_delay_ms)
            if self._ref_buffer is not None
            else self._silence
        )
        # speexdsp.EchoCanceller.process(mic, reference):
        #   mic       — raw microphone frame (contains speech + echo)
        #   reference — what the speaker was playing (the echo source)
        # Returns the echo-cancelled mic signal.
        # IMPORTANT: argument order is (mic, ref) — NOT (ref, mic).
        # Reversing them would feed the reference as speech and produce silent
        # or corrupt output with no error from speexdsp.
        return self._ec.process(mic_frame, ref)
