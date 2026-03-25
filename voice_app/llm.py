import logging
import queue
import re
import threading
from typing import Dict, List

from openai import OpenAI

logger = logging.getLogger(__name__)


def split_into_sentences(text: str) -> List[str]:
    """Split text on sentence boundaries while preserving final fragments."""

    text = text.strip()
    if not text:
        return []

    parts = re.split(r"(?<=[.!?])\s+", text)
    return [part for part in parts if part.strip()]


class LLMClient:
    """Stream chat completions and hand complete sentences to TTS."""

    def __init__(
        self,
        api_key: str,
        config,
        event_queue: queue.Queue,
        tts_engine,
    ) -> None:
        self.config = config
        self.event_queue = event_queue
        self.tts_engine = tts_engine
        self._client = OpenAI(api_key=api_key)
        self._input_queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._current_gen_id: int = 0
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Launch the LLM worker thread."""

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="LLMClient",
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the LLM worker thread."""

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def submit(self, messages: List[Dict], gen_id: int) -> None:
        """Queue a generation request."""

        self._current_gen_id = gen_id
        self._input_queue.put((messages, gen_id))

    def cancel(self) -> None:
        """Invalidate the active generation and drain pending work."""

        self._current_gen_id += 1
        while not self._input_queue.empty():
            try:
                self._input_queue.get_nowait()
            except queue.Empty:
                break

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                messages, gen_id = self._input_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            self._stream_response(messages, gen_id)

    def _stream_response(self, messages: List[Dict], gen_id: int) -> None:
        buffer = ""
        full_response = ""
        first_chunk_sent = False

        try:
            stream = self._client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream=True,
            )

            for chunk in stream:
                if gen_id != self._current_gen_id:
                    logger.debug("LLM gen_id %d stale, aborting stream", gen_id)
                    return

                delta = chunk.choices[0].delta.content
                if delta is None:
                    continue

                buffer += delta
                sentences = split_into_sentences(buffer)

                if len(sentences) > 1:
                    for sentence in sentences[:-1]:
                        if gen_id != self._current_gen_id:
                            return

                        if not first_chunk_sent:
                            self.event_queue.put({"type": "FIRST_TTS_CHUNK"})
                            first_chunk_sent = True

                        self.tts_engine.speak(sentence, gen_id)
                        full_response += sentence + " "

                    buffer = sentences[-1]

            if buffer.strip() and gen_id == self._current_gen_id:
                if not first_chunk_sent:
                    self.event_queue.put({"type": "FIRST_TTS_CHUNK"})
                    first_chunk_sent = True
                self.tts_engine.speak(buffer.strip(), gen_id)
                full_response += buffer.strip()

            if gen_id == self._current_gen_id:
                if not full_response.strip():
                    self.event_queue.put(
                        {"type": "ERROR", "error": "LLM returned empty response"}
                    )
                    return

                self.tts_engine.finish(gen_id)
                self.event_queue.put(
                    {
                        "type": "LLM_RESPONSE_READY",
                        "gen_id": gen_id,
                        "response": full_response.strip(),
                    }
                )
        except Exception as exc:
            logger.error("LLM error: %s", exc)
            self.event_queue.put({"type": "ERROR", "error": str(exc)})
