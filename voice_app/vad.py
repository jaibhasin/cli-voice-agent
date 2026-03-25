import collections

import webrtcvad


class VADDetector:
    """Wrap WebRTC VAD with a sliding window to smooth false positives."""

    def __init__(
        self,
        sample_rate: int,
        aggressiveness: int,
        ring_buffer_size: int,
        speech_start_frames: int,
    ) -> None:
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = sample_rate
        self.ring_buffer_size = ring_buffer_size
        self.speech_start_frames = speech_start_frames
        self._ring_buffer: collections.deque[bool] = collections.deque(
            maxlen=ring_buffer_size
        )

    def process_frame(self, frame: bytes) -> bool:
        """Return True when enough recent frames are classified as speech."""

        try:
            is_speech = self.vad.is_speech(frame, self.sample_rate)
        except Exception:
            is_speech = False

        self._ring_buffer.append(is_speech)
        return sum(self._ring_buffer) >= self.speech_start_frames

    def reset(self) -> None:
        """Forget prior frame classifications."""

        self._ring_buffer.clear()
