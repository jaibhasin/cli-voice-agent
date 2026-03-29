"""
voice_app/vad.py — Voice Activity Detection using Silero VAD.

The Silero model runs on fixed-size 512-sample (32 ms @ 16 kHz) windows of
mono float32 audio.  Raw frames that arrive from AudioCapture (int16 bytes)
may be any size; we accumulate them in an internal byte buffer and drain
completed 512-sample chunks into the model.

Public API is identical to the previous WebRTC-VAD implementation so the
Orchestrator and tests need no changes.
"""

import array
import collections
import struct

import torch
from silero_vad import load_silero_vad


_AGGRESSIVENESS_TO_THRESHOLD = {
    0: 0.3,
    1: 0.5,
    2: 0.7,
    3: 0.9,
}

# Silero model chunk sizes (samples, not bytes) per sample-rate.
_CHUNK_SAMPLES = {
    16000: 512,
    8000: 256,
}


class _SileroWrapper:
    """Thin wrapper around the Silero VAD model to emulate the webrtcvad API
    (``is_speech(frame_bytes, sample_rate) -> bool``) so existing tests can
    patch ``vad.vad.is_speech`` without modification."""

    def __init__(self, model: torch.nn.Module, threshold: float) -> None:
        self._model = model
        self.threshold = threshold

    def is_speech(self, frame_bytes: bytes, sample_rate: int) -> bool:
        """Return True when the model speech probability exceeds the threshold.

        ``frame_bytes`` must contain exactly *chunk_samples* int16 samples.
        """
        n_samples = len(frame_bytes) // 2
        samples = struct.unpack(f"{n_samples}h", frame_bytes)
        audio = torch.tensor(samples, dtype=torch.float32) / 32768.0
        with torch.no_grad():
            prob = self._model(audio, sample_rate).item()
        return prob >= self.threshold

    def reset_states(self) -> None:
        self._model.reset_states()


class VADDetector:
    """Silero VAD with a sliding window to smooth transient false positives.

    Constructor arguments match the original WebRTC-VAD implementation so that
    no changes are needed in the Orchestrator or configuration:

    - ``aggressiveness`` (0–3) is mapped to a detection probability threshold
      (0.3 / 0.5 / 0.7 / 0.9).  Values outside [0, 3] clamp to the nearest
      boundary.
    - ``ring_buffer_size`` and ``speech_start_frames`` preserve the existing
      smoothing behaviour.
    """

    def __init__(
        self,
        sample_rate: int,
        aggressiveness: int,
        ring_buffer_size: int,
        speech_start_frames: int,
    ) -> None:
        if sample_rate not in _CHUNK_SAMPLES:
            raise ValueError(
                f"silero-vad only supports 8000 or 16000 Hz; got {sample_rate}"
            )

        aggressiveness = max(0, min(3, aggressiveness))
        threshold = _AGGRESSIVENESS_TO_THRESHOLD[aggressiveness]

        model = load_silero_vad()
        self.vad = _SileroWrapper(model, threshold)

        self.sample_rate = sample_rate
        self.ring_buffer_size = ring_buffer_size
        self.speech_start_frames = speech_start_frames

        self._chunk_samples: int = _CHUNK_SAMPLES[sample_rate]
        self._chunk_bytes: int = self._chunk_samples * 2  # int16 → 2 bytes/sample
        self._byte_buffer: bytearray = bytearray()

        self._ring_buffer: collections.deque[bool] = collections.deque(
            maxlen=ring_buffer_size
        )

    def process_frame(self, frame: bytes) -> bool:
        """Return True when enough recent model calls classify a chunk as speech.

        Incoming ``frame`` bytes are accumulated until a full 512-sample chunk
        is available, at which point the Silero model is consulted.  The ring
        buffer is updated for every completed chunk.
        """
        self._byte_buffer.extend(frame)

        while len(self._byte_buffer) >= self._chunk_bytes:
            chunk = bytes(self._byte_buffer[: self._chunk_bytes])
            del self._byte_buffer[: self._chunk_bytes]

            try:
                is_speech = self.vad.is_speech(chunk, self.sample_rate)
            except Exception:
                is_speech = False

            self._ring_buffer.append(is_speech)

        return sum(self._ring_buffer) >= self.speech_start_frames

    def reset(self) -> None:
        """Forget prior frame classifications and reset model LSTM states."""
        self._ring_buffer.clear()
        self._byte_buffer.clear()
        self.vad.reset_states()
