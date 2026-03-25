from unittest.mock import patch

from voice_app.vad import VADDetector

SILENT_FRAME = b"\x00" * 640
VOICED_FRAME = b"\x01" * 640


def make_vad(speech_start_frames=6, ring_buffer_size=8):
    return VADDetector(
        sample_rate=16000,
        aggressiveness=2,
        ring_buffer_size=ring_buffer_size,
        speech_start_frames=speech_start_frames,
    )


def test_no_speech_detected_on_all_silent_frames():
    vad = make_vad()
    with patch.object(vad.vad, "is_speech", return_value=False):
        results = [vad.process_frame(SILENT_FRAME) for _ in range(10)]
    assert not any(results)


def test_speech_detected_when_threshold_met():
    vad = make_vad(speech_start_frames=3, ring_buffer_size=5)
    with patch.object(vad.vad, "is_speech", return_value=True):
        results = [vad.process_frame(VOICED_FRAME) for _ in range(5)]
    assert results[2] is True
    assert results[4] is True


def test_speech_not_detected_below_threshold():
    vad = make_vad(speech_start_frames=4, ring_buffer_size=5)

    call_count = 0

    def is_speech_side_effect(frame, rate):
        nonlocal call_count
        call_count += 1
        return call_count <= 3

    with patch.object(vad.vad, "is_speech", side_effect=is_speech_side_effect):
        results = [vad.process_frame(VOICED_FRAME) for _ in range(5)]

    assert not any(results)


def test_reset_clears_ring_buffer():
    vad = make_vad(speech_start_frames=3, ring_buffer_size=5)
    with patch.object(vad.vad, "is_speech", return_value=True):
        for _ in range(3):
            vad.process_frame(VOICED_FRAME)

    vad.reset()

    with patch.object(vad.vad, "is_speech", return_value=False):
        result = vad.process_frame(SILENT_FRAME)
    assert result is False


def test_vad_error_treated_as_silence():
    vad = make_vad(speech_start_frames=2, ring_buffer_size=3)
    with patch.object(vad.vad, "is_speech", side_effect=Exception("bad frame")):
        results = [vad.process_frame(VOICED_FRAME) for _ in range(5)]
    assert not any(results)
