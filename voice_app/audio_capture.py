import queue
import threading

import pyaudio


class AudioCapture:
    """Continuously read microphone frames and fan them out to consumers."""

    FRAME_DURATION_MS = 20

    def __init__(self, sample_rate: int, channels: int) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = int(sample_rate * self.FRAME_DURATION_MS / 1000)

        self._queues: list[queue.Queue] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._pa = pyaudio.PyAudio()
        self._stream = None

    def add_consumer(self, consumer_queue: queue.Queue) -> None:
        """Register a queue that should receive each captured frame."""

        self._queues.append(consumer_queue)

    def start(self) -> None:
        """Open the microphone and start the background capture loop."""

        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
        )
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="AudioCapture",
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop capturing and release audio resources."""

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        self._pa.terminate()

    def _capture_loop(self) -> None:
        """Read one frame at a time and enqueue it for every consumer."""

        while not self._stop_event.is_set():
            try:
                frame = self._stream.read(
                    self.chunk_size,
                    exception_on_overflow=False,
                )
                for consumer_queue in self._queues:
                    try:
                        consumer_queue.put_nowait(frame)
                    except queue.Full:
                        pass
            except Exception:
                break
