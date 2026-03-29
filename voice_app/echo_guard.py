import re
import threading
import time
from difflib import SequenceMatcher


_NORMALIZE_RE = re.compile(r"[^a-z0-9\s]+")
_MULTISPACE_RE = re.compile(r"\s+")


def _normalize_text(text: str) -> str:
    normalized = _NORMALIZE_RE.sub(" ", text.lower())
    return _MULTISPACE_RE.sub(" ", normalized).strip()


class EchoTranscriptGuard:
    """
    Filter residual speaker leakage that survives AEC.

    AEC remains the primary defence. This guard only activates while the
    assistant is speaking and for a short cooldown after playback ends, using
    text similarity to drop transcripts that are likely just the assistant's
    own voice coming back through the microphone.
    """

    def __init__(
        self,
        cooldown_ms: int = 2000,
        interrupted_cooldown_ms: int = 600,
    ) -> None:
        self._cooldown_ns = max(0, cooldown_ms) * 1_000_000
        self._interrupted_cooldown_ns = max(0, interrupted_cooldown_ms) * 1_000_000
        self._guard_until_ns = 0
        self._active_gen_id = -1
        self._active_text = ""
        self._recent_text = ""
        self._lock = threading.Lock()

    def start_generation(self, gen_id: int) -> None:
        with self._lock:
            self._active_gen_id = gen_id
            self._active_text = ""

    def note_tts_chunk(self, gen_id: int, text: str) -> None:
        normalized = _normalize_text(text)
        if not normalized:
            return

        with self._lock:
            if gen_id != self._active_gen_id:
                self._active_gen_id = gen_id
                self._active_text = normalized
            elif self._active_text:
                self._active_text = f"{self._active_text} {normalized}"
            else:
                self._active_text = normalized

    def note_response_ready(self, gen_id: int, text: str) -> None:
        normalized = _normalize_text(text)
        if not normalized:
            return

        with self._lock:
            if gen_id == self._active_gen_id and len(normalized) >= len(self._active_text):
                self._active_text = normalized
            elif len(normalized) >= len(self._recent_text):
                self._recent_text = normalized

    def note_tts_complete(self, gen_id: int, now_ns: int | None = None) -> None:
        self._promote_active_text(
            gen_id=gen_id,
            guard_ns=self._cooldown_ns,
            now_ns=now_ns,
        )

    def note_interrupt(self, gen_id: int, now_ns: int | None = None) -> None:
        self._promote_active_text(
            gen_id=gen_id,
            guard_ns=self._interrupted_cooldown_ns,
            now_ns=now_ns,
        )

    def should_gate_vad(self, now_ns: int | None = None) -> bool:
        current_ns = time.monotonic_ns() if now_ns is None else now_ns
        with self._lock:
            return current_ns <= self._guard_until_ns

    def is_probable_echo(
        self,
        transcript: str,
        *,
        speaking_active: bool,
        now_ns: int | None = None,
    ) -> bool:
        normalized_transcript = _normalize_text(transcript)
        if not normalized_transcript:
            return False

        current_ns = time.monotonic_ns() if now_ns is None else now_ns
        with self._lock:
            if speaking_active and self._active_text:
                reference = self._active_text
            elif current_ns <= self._guard_until_ns:
                reference = self._recent_text or self._active_text
            else:
                reference = ""

        if not reference:
            return False

        return _looks_like_echo(normalized_transcript, reference)

    def clear(self) -> None:
        with self._lock:
            self._guard_until_ns = 0
            self._active_gen_id = -1
            self._active_text = ""
            self._recent_text = ""

    def _promote_active_text(
        self,
        *,
        gen_id: int,
        guard_ns: int,
        now_ns: int | None,
    ) -> None:
        current_ns = time.monotonic_ns() if now_ns is None else now_ns

        with self._lock:
            if gen_id == self._active_gen_id and self._active_text:
                self._recent_text = self._active_text
                self._active_text = ""
            self._guard_until_ns = max(self._guard_until_ns, current_ns + guard_ns)


def _looks_like_echo(transcript: str, reference: str) -> bool:
    transcript_tokens = transcript.split()
    if len(transcript) < 8 and len(transcript_tokens) < 2:
        return False

    if transcript in reference:
        return True

    longest_match = SequenceMatcher(None, transcript, reference).find_longest_match(
        0,
        len(transcript),
        0,
        len(reference),
    )
    if longest_match.size / max(1, len(transcript)) >= 0.75:
        return True

    if len(transcript_tokens) < 3:
        return False

    reference_tokens = set(reference.split())
    overlap = sum(1 for token in transcript_tokens if token in reference_tokens)
    return overlap / len(transcript_tokens) >= 0.85
