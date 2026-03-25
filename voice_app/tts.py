import queue
import threading

import pyttsx3


class TTSEngine:
    """Speak queued sentence chunks with interruption and generation gating."""

    _FINISH_MARKER = object()

    def __init__(self, rate: int, volume: float, event_queue: queue.Queue | None = None) -> None:
        self.rate = rate
        self.volume = volume
        self.event_queue = event_queue
        self._tts_queue: queue.Queue = queue.Queue()
        self._interrupt_flag = threading.Event()
        self._stop_event = threading.Event()
        self._current_gen_id: int = -1
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Launch the TTS worker thread."""

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="TTSEngine",
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the TTS worker."""

        self._stop_event.set()
        self._interrupt_flag.set()
        if self._thread is not None:
            self._thread.join(timeout=3)

    def set_generation(self, gen_id: int) -> None:
        """Start a new generation and clear any prior interrupt signal."""

        self._current_gen_id = gen_id
        self._interrupt_flag.clear()

    def speak(self, text: str, gen_id: int) -> None:
        """Queue a text chunk for speech synthesis."""

        self._tts_queue.put((text, gen_id))

    def finish(self, gen_id: int) -> None:
        """Signal that no more chunks are coming for this generation."""

        self._tts_queue.put((self._FINISH_MARKER, gen_id))

    def interrupt(self) -> None:
        """Stop current speech and discard any queued chunks."""

        self._interrupt_flag.set()
        self._drain_queue()

    def _drain_queue(self) -> None:
        while not self._tts_queue.empty():
            try:
                self._tts_queue.get_nowait()
            except queue.Empty:
                break

    def _run(self) -> None:
        engine = pyttsx3.init()
        engine.setProperty("rate", self.rate)
        engine.setProperty("volume", self.volume)

        def on_word_start(name, location, length):
            if self._interrupt_flag.is_set():
                engine.stop()

        engine.connect("started-word", on_word_start)

        while not self._stop_event.is_set():
            try:
                text, gen_id = self._tts_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if gen_id != self._current_gen_id:
                continue

            if text is self._FINISH_MARKER:
                if not self._interrupt_flag.is_set() and self.event_queue is not None:
                    self.event_queue.put({"type": "TTS_COMPLETE", "gen_id": gen_id})
                continue

            if self._interrupt_flag.is_set():
                continue

            engine.say(text)
            engine.runAndWait()
