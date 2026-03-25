import asyncio
import logging
import queue
import threading

from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType

logger = logging.getLogger(__name__)


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

        try:
            async with dg.listen.v1.connect(
                model=self.config.model,
                language=self.config.language,
                smart_format=str(self.config.smart_format).lower(),
                endpointing=str(self.config.endpointing),
                utterance_end_ms=str(self.config.utterance_end_ms),
                channels=str(self.channels),
                sample_rate=str(self.sample_rate),
                encoding="linear16",
            ) as connection:

                async def on_message(message) -> None:
                    transcript = self._extract_transcript(message)
                    if not transcript:
                        return

                    if getattr(message, "is_final", False):
                        event_type = (
                            "UTTERANCE_COMPLETE"
                            if getattr(message, "speech_final", False)
                            else "TRANSCRIPT_INTERIM"
                        )
                        self.event_queue.put(
                            {
                                "type": event_type,
                                "text": transcript,
                            }
                        )

                async def on_error(error) -> None:
                    logger.error("Deepgram error: %s", error)
                    self.event_queue.put({"type": "ERROR", "error": str(error)})

                connection.on(EventType.MESSAGE, on_message)
                connection.on(EventType.ERROR, on_error)
                listener_task = asyncio.create_task(connection.start_listening())

                while not self._stop_event.is_set():
                    try:
                        frame = self.audio_queue.get(timeout=0.1)
                    except queue.Empty:
                        await asyncio.sleep(0)
                        continue

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

    @staticmethod
    def _extract_transcript(message) -> str:
        """Pull the best transcript text from a Deepgram response object."""

        try:
            alternatives = message.channel.alternatives
            transcript = alternatives[0].transcript
        except (AttributeError, IndexError, TypeError):
            return ""

        return transcript.strip()
