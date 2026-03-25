import asyncio
import logging
import queue
import threading

from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType

logger = logging.getLogger(__name__)


def _blocking_queue_get(audio_queue: queue.Queue, timeout: float):
    """Used with asyncio.to_thread so the event loop is not blocked."""

    try:
        return audio_queue.get(timeout=timeout)
    except queue.Empty:
        return None


class STTClient:
    """Stream microphone audio to Deepgram and emit transcript events."""

    def __init__(
        self,
        api_key: str,
        config,
        audio_queue: queue.Queue,
        event_queue: queue.Queue,
        *,
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> None:
        self.api_key = api_key
        self.config = config
        self.audio_queue = audio_queue
        self.event_queue = event_queue
        self.sample_rate = sample_rate
        self.channels = channels

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._final_segments: list[str] = []
        # When set, send silence to Deepgram and ignore transcripts (prevents speaker bleed → STT).
        self._suppress_echo = threading.Event()

    def set_echo_suppression(self, active: bool) -> None:
        """Half-duplex guard: stop feeding mic audio to STT while assistant audio plays."""

        if active:
            self._suppress_echo.set()
        else:
            self._suppress_echo.clear()

    def start(self) -> None:
        """Launch the STT worker thread."""

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="STTClient",
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the STT worker to stop and wait for shutdown."""

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        """Own an asyncio loop inside the worker thread."""

        asyncio.run(self._async_run())

    async def _async_run(self) -> None:
        """Maintain the websocket connection and forward transcript events."""

        dg = AsyncDeepgramClient(api_key=self.api_key)

        self._final_segments.clear()

        try:
            async with dg.listen.v1.connect(
                model=self.config.model,
                language=self.config.language,
                smart_format=str(self.config.smart_format).lower(),
                # Deepgram requires interim_results when using utterance_end_ms (otherwise HTTP 400).
                interim_results="true",
                endpointing=str(self.config.endpointing),
                utterance_end_ms=str(self.config.utterance_end_ms),
                channels=str(self.channels),
                sample_rate=str(self.sample_rate),
                encoding="linear16",
            ) as connection:

                async def on_message(message) -> None:
                    # UtteranceEnd has no transcript; must be handled before transcript extraction.
                    msg_type = getattr(message, "type", None)
                    if msg_type == "UtteranceEnd":
                        if getattr(message, "last_word_end", 0.0) == -1:
                            logger.debug("STT: UtteranceEnd ignored (last_word_end=-1)")
                            return
                        if self._suppress_echo.is_set():
                            self._final_segments.clear()
                            return
                        text = self._flush_pending_utterance()
                        if text:
                            self.event_queue.put({"type": "UTTERANCE_COMPLETE", "text": text})
                        return

                    if self._suppress_echo.is_set():
                        return

                    transcript = self._extract_transcript(message)
                    if not transcript:
                        return

                    if not getattr(message, "is_final", False):
                        logger.debug("STT interim: %s", transcript)
                        self.event_queue.put(
                            {"type": "TRANSCRIPT_INTERIM", "text": transcript}
                        )
                        return

                    self._merge_final_segment(transcript)
                    logger.debug("STT final segment: %s", transcript)

                    if getattr(message, "speech_final", False):
                        text = self._flush_pending_utterance()
                        if text:
                            self.event_queue.put({"type": "UTTERANCE_COMPLETE", "text": text})

                async def on_error(error) -> None:
                    logger.error("Deepgram error: %s", error)
                    self.event_queue.put({"type": "ERROR", "error": str(error)})

                connection.on(EventType.MESSAGE, on_message)
                connection.on(EventType.ERROR, on_error)
                listener_task = asyncio.create_task(connection.start_listening())
                # Allow start_listening() to emit OPEN and begin recv (do not block the loop on queue.get).
                await asyncio.sleep(0)

                while not self._stop_event.is_set():
                    frame = await asyncio.to_thread(
                        _blocking_queue_get, self.audio_queue, 0.1
                    )
                    if frame is None:
                        continue

                    if self._suppress_echo.is_set():
                        frame = b"\x00" * len(frame)

                    try:
                        await connection.send_media(frame)
                    except Exception as exc:
                        logger.error("STT send error: %s", exc)
                        self.event_queue.put({"type": "ERROR", "error": str(exc)})
                        break

                try:
                    await connection.send_finalize()
                    await connection.send_close_stream()
                except Exception:
                    pass

                await asyncio.wait_for(listener_task, timeout=2)
        except Exception as exc:
            logger.error("Deepgram connection error: %s", exc)
            self.event_queue.put({"type": "ERROR", "error": str(exc)})

    def _merge_final_segment(self, transcript: str) -> None:
        """Merge a locked-in final segment, replacing the last segment when Deepgram refines it."""

        if not self._final_segments:
            self._final_segments.append(transcript)
            return
        last = self._final_segments[-1]
        if transcript.startswith(last) or last.startswith(transcript):
            self._final_segments[-1] = transcript
        else:
            self._final_segments.append(transcript)

    def _flush_pending_utterance(self) -> str:
        text = " ".join(self._final_segments).strip()
        self._final_segments.clear()
        return text

    @staticmethod
    def _extract_transcript(message) -> str:
        """Pull the best transcript text from a Deepgram response object."""

        try:
            alternatives = message.channel.alternatives
            transcript = alternatives[0].transcript
        except (AttributeError, IndexError, TypeError):
            return ""

        return transcript.strip()
