"""
Tests for voice_app/aec.py — SpeakerReferenceBuffer and AECProcessor.

SpeakerReferenceBuffer is pure Python and fully testable without speexdsp.

AECProcessor wraps the speexdsp library.  Its tests mock speexdsp so they run
on any system regardless of whether libspeexdsp is installed.
"""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from voice_app.aec import FRAME_BYTES, FRAME_SIZE, SpeakerReferenceBuffer


# ===========================================================================
# SpeakerReferenceBuffer
# ===========================================================================


class TestSpeakerReferenceBuffer:
    """Unit tests covering the timestamped ring-buffer behaviour."""

    def _make_buf(self, maxlen: int = 50) -> SpeakerReferenceBuffer:
        return SpeakerReferenceBuffer(maxlen=maxlen, frame_size=FRAME_SIZE)

    def _silence(self) -> bytes:
        return bytes(FRAME_BYTES)

    # -----------------------------------------------------------------------
    # Basic push / get
    # -----------------------------------------------------------------------

    def test_get_frame_at_returns_silence_when_empty(self):
        """An empty buffer always returns the silence constant."""
        buf = self._make_buf()
        assert buf.get_frame_at(time.monotonic_ns()) == self._silence()

    def test_push_then_get_returns_pushed_frame(self):
        """A single frame can be retrieved by its exact timestamp."""
        buf = self._make_buf()
        ts = time.monotonic_ns()
        frame = bytes(range(FRAME_BYTES % 256)) * (FRAME_BYTES // 256 + 1)
        frame = frame[:FRAME_BYTES]
        buf.push(ts, frame)

        result = buf.get_frame_at(ts)
        assert result == frame

    def test_get_returns_closest_frame_by_timestamp(self):
        """get_frame_at returns the entry with the smallest timestamp distance."""
        buf = self._make_buf()
        base_ns = time.monotonic_ns()
        frame_a = b"\xAA" * FRAME_BYTES
        frame_b = b"\xBB" * FRAME_BYTES
        # frame_a at t=0, frame_b at t=100ms
        buf.push(base_ns, frame_a)
        buf.push(base_ns + 100_000_000, frame_b)

        # Query at t=80ms — closer to frame_b
        result = buf.get_frame_at(base_ns + 80_000_000)
        assert result == frame_b

    # -----------------------------------------------------------------------
    # Stale frame handling
    # -----------------------------------------------------------------------

    def test_get_returns_silence_for_stale_frame(self):
        """A frame more than 2 frame durations away is treated as stale → silence."""
        buf = self._make_buf()
        old_ts = time.monotonic_ns() - 500_000_000  # 500 ms in the past
        buf.push(old_ts, b"\xFF" * FRAME_BYTES)

        # Query at "now" — far from the buffered frame
        result = buf.get_frame_at(time.monotonic_ns())
        assert result == self._silence()

    # -----------------------------------------------------------------------
    # Clear
    # -----------------------------------------------------------------------

    def test_clear_removes_all_frames(self):
        """After clear(), get_frame_at returns silence regardless of query time."""
        buf = self._make_buf()
        ts = time.monotonic_ns()
        buf.push(ts, b"\x01" * FRAME_BYTES)
        buf.push(ts + 20_000_000, b"\x02" * FRAME_BYTES)

        buf.clear()

        assert buf.get_frame_at(ts) == self._silence()

    # -----------------------------------------------------------------------
    # Ring-buffer overflow / maxlen
    # -----------------------------------------------------------------------

    def test_oldest_frame_evicted_at_maxlen(self):
        """When the deque is full, the oldest entry is evicted automatically."""
        maxlen = 5
        buf = self._make_buf(maxlen=maxlen)
        base = time.monotonic_ns()

        oldest_ts = base
        oldest_frame = b"\xDE" * FRAME_BYTES
        buf.push(oldest_ts, oldest_frame)  # will be evicted

        for i in range(1, maxlen + 1):
            buf.push(base + i * 20_000_000, bytes([i % 256]) * FRAME_BYTES)

        # The oldest frame is too stale now; its entry was evicted
        result = buf.get_frame_at(oldest_ts)
        # Either silence (evicted + stale) or a nearby frame — must NOT be oldest_frame
        assert result != oldest_frame

    # -----------------------------------------------------------------------
    # Thread safety
    # -----------------------------------------------------------------------

    def test_concurrent_push_and_get_do_not_crash(self):
        """
        Concurrent producers and consumers must not raise due to race conditions.
        This is a stress test — it checks for absence of exceptions, not specific
        output values.
        """
        buf = self._make_buf(maxlen=100)
        errors: list[Exception] = []

        def producer():
            for _ in range(200):
                buf.push(time.monotonic_ns(), bytes(FRAME_BYTES))

        def consumer():
            for _ in range(200):
                try:
                    buf.get_frame_at(time.monotonic_ns())
                except Exception as exc:
                    errors.append(exc)

        threads = [
            threading.Thread(target=producer),
            threading.Thread(target=consumer),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=2)

        assert not errors, f"Thread-safety errors: {errors}"


# ===========================================================================
# AECProcessor
# ===========================================================================


class TestAECProcessor:
    """Tests for AECProcessor — mocks speexdsp to avoid the native dependency."""

    def _make_mock_speexdsp(self):
        """Return a mock speexdsp module with a functional EchoCanceller.create()."""
        mock_ec = MagicMock()
        mock_ec.process.side_effect = lambda mic, ref: mic  # identity for simplicity
        mock_module = MagicMock()
        mock_module.EchoCanceller.create.return_value = mock_ec
        return mock_module, mock_ec

    def _make_aec_config(self, filter_length: int = 2048):
        from voice_app.config import AECConfig

        return AECConfig(enabled=True, filter_length=filter_length)

    # -----------------------------------------------------------------------
    # Happy path
    # -----------------------------------------------------------------------

    def test_process_calls_ec_with_mic_and_ref(self):
        """process() fetches the reference frame and passes both to ec.process()."""
        mock_speexdsp, mock_ec = self._make_mock_speexdsp()

        with patch.dict("sys.modules", {"speexdsp": mock_speexdsp}):
            from voice_app.aec import AECProcessor

            ref_buf = SpeakerReferenceBuffer(maxlen=10, frame_size=FRAME_SIZE)
            ts = time.monotonic_ns()
            ref_frame = b"\xBB" * FRAME_BYTES
            ref_buf.push(ts, ref_frame)

            cfg = self._make_aec_config()
            proc = AECProcessor(cfg, ref_buffer=ref_buf)

            mic_frame = b"\xAA" * FRAME_BYTES
            proc.process(mic_frame, ts)

        mock_ec.process.assert_called_once_with(mic_frame, ref_frame)

    def test_process_uses_silence_when_no_ref_buffer(self):
        """With ref_buffer=None the processor passes silence as the reference."""
        mock_speexdsp, mock_ec = self._make_mock_speexdsp()

        with patch.dict("sys.modules", {"speexdsp": mock_speexdsp}):
            from voice_app.aec import AECProcessor

            cfg = self._make_aec_config()
            proc = AECProcessor(cfg, ref_buffer=None)

            mic_frame = b"\xAA" * FRAME_BYTES
            proc.process(mic_frame, time.monotonic_ns())

        _, ref_arg = mock_ec.process.call_args[0]
        assert ref_arg == bytes(FRAME_BYTES), "Expected silence when no ref_buffer"

    def test_ec_created_with_correct_params(self):
        """EchoCanceller.create() receives frame_size, filter_length, sample_rate."""
        mock_speexdsp, _ = self._make_mock_speexdsp()

        with patch.dict("sys.modules", {"speexdsp": mock_speexdsp}):
            from voice_app.aec import AECProcessor

            cfg = self._make_aec_config(filter_length=1024)
            AECProcessor(cfg, sample_rate=16_000, frame_size=FRAME_SIZE)

        mock_speexdsp.EchoCanceller.create.assert_called_once_with(
            FRAME_SIZE, 1024, 16_000
        )

    # -----------------------------------------------------------------------
    # Graceful ImportError
    # -----------------------------------------------------------------------

    def test_import_error_raised_with_instructions_when_speexdsp_missing(self):
        """If speexdsp is not installed, AECProcessor raises ImportError with guidance."""
        # Remove speexdsp from sys.modules to simulate its absence
        with patch.dict("sys.modules", {"speexdsp": None}):
            # Re-import to pick up the patched sys.modules state
            import importlib

            import voice_app.aec as aec_mod

            importlib.reload(aec_mod)

            cfg = self._make_aec_config()
            with pytest.raises(ImportError, match="speexdsp"):
                aec_mod.AECProcessor(cfg)
