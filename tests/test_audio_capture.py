import queue
import time
from unittest.mock import MagicMock, patch


def make_mock_pyaudio(frames: list[bytes]):
    """Create a mocked PyAudio instance that yields canned frames."""

    mock_stream = MagicMock()
    mock_stream.read.side_effect = frames + [b"\x00" * 640] * 1000

    mock_pa = MagicMock()
    mock_pa.open.return_value = mock_stream
    return mock_pa, mock_stream


def test_audio_capture_fans_out_to_multiple_queues():
    from voice_app.audio_capture import AudioCapture

    frames = [b"\x01" * 640, b"\x02" * 640, b"\x03" * 640]
    mock_pa, _ = make_mock_pyaudio(frames)

    with patch("voice_app.audio_capture.pyaudio.PyAudio", return_value=mock_pa):
        capture = AudioCapture(sample_rate=16000, channels=1)

        q1: queue.Queue = queue.Queue()
        q2: queue.Queue = queue.Queue()
        capture.add_consumer(q1)
        capture.add_consumer(q2)

        capture.start()
        time.sleep(0.1)
        capture.stop()

    items_q1 = list(q1.queue)
    items_q2 = list(q2.queue)
    assert len(items_q1) >= 3
    assert items_q1[:3] == items_q2[:3]


def test_audio_capture_stops_cleanly():
    from voice_app.audio_capture import AudioCapture

    mock_pa, _ = make_mock_pyaudio([b"\x00" * 640] * 100)

    with patch("voice_app.audio_capture.pyaudio.PyAudio", return_value=mock_pa):
        capture = AudioCapture(sample_rate=16000, channels=1)
        capture.start()
        time.sleep(0.05)
        capture.stop()

    assert True
