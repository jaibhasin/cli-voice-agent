import queue
import threading
import time

import pyaudio


class AudioCapture:
    """Continuously read microphone frames and fan them out to consumers."""

    FRAME_DURATION_MS = 20

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        aec_processor=None,
    ) -> None:
        """
        Parameters
        ----------
        sample_rate:   Mic sample rate in Hz (16 000 for this pipeline).
        channels:      Number of channels (1 = mono).
        aec_processor: Optional AECProcessor instance.  When supplied, each mic
                       frame is passed through AEC before being forwarded to
                       consumers (VAD queue and STT queue).  Set to None when
                       aec.enabled is False (half-duplex path).
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = int(sample_rate * self.FRAME_DURATION_MS / 1000)

        # Optional AEC processor injected by the orchestrator when aec.enabled = True.
        self._aec_processor = aec_processor

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
        """
        Read one frame at a time, optionally apply AEC, and enqueue to consumers.

        AEC path (aec.enabled = True)
        ------------------------------
        After reading a raw mic frame we immediately stamp it with
        time.monotonic_ns() and pass it through AECProcessor.process().
        The processor looks up the speaker reference frame that was playing at
        that timestamp (from SpeakerReferenceBuffer) and subtracts it from the
        mic signal using speexdsp.  The cleaned frame is then enqueued.

        Half-duplex path (aec.enabled = False)
        ----------------------------------------
        The frame is enqueued as-is.  The STT client handles silence injection
        via the echo-suppression flag set by the orchestrator.
        """
        while not self._stop_event.is_set():
            try:
                frame = self._stream.read(
                    self.chunk_size,
                    exception_on_overflow=False,
                )
                # Apply AEC before distributing to VAD/STT consumers
                if self._aec_processor is not None:
                    frame = self._aec_processor.process(frame, time.monotonic_ns())

                for consumer_queue in self._queues:
                    try:
                        consumer_queue.put_nowait(frame)
                    except queue.Full:
                        pass
            except Exception:
                break
