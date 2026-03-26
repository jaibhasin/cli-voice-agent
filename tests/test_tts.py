"""
Tests for the Deepgram TTS WebSocket + PyAudio TTSEngine.

Tests that only exercise queue logic (speak / interrupt / set_generation) do NOT
call start() and therefore need no audio-device mocking beyond the conftest autouse
fixture.

Tests that run the worker thread (stale-chunk test) rely on the empty api_key path
which activates _run_silent_loop — no network calls, no Deepgram SDK needed.
The conftest autouse fixture mocks voice_app.tts.pyaudio.PyAudio so start() never
opens a real audio device.
"""

import queue
import time


def test_speak_queues_chunk_with_correct_gen_id():
    """speak() places exactly one (text, gen_id) item into _tts_queue."""
    from voice_app.tts import TTSEngine

    tts = TTSEngine(rate=175, volume=0.9)
    tts.set_generation(1)
    tts.speak("Hello world", gen_id=1)

    item = tts._tts_queue.get_nowait()
    assert item == ("Hello world", 1)


def test_interrupt_drains_queue_and_sets_flag():
    """interrupt() empties _tts_queue and sets _interrupt_flag."""
    from voice_app.tts import TTSEngine

    tts = TTSEngine(rate=175, volume=0.9)
    tts.set_generation(1)
    tts.speak("sentence one", gen_id=1)
    tts.speak("sentence two", gen_id=1)
    tts.speak("sentence three", gen_id=1)

    tts.interrupt()

    assert tts._tts_queue.empty()
    assert tts._interrupt_flag.is_set()


def test_interrupt_drains_playback_queue():
    """interrupt() also drains the PCM playback queue to silence the speaker."""
    from voice_app.tts import FRAME_BYTES, TTSEngine

    tts = TTSEngine(rate=175, volume=0.9)
    # Simulate PCM frames already queued for playback
    tts._playback_queue.put(bytes(FRAME_BYTES))
    tts._playback_queue.put(bytes(FRAME_BYTES))

    tts.interrupt()

    assert tts._playback_queue.empty()


def test_interrupt_clears_ref_buffer():
    """interrupt() calls clear() on the AEC reference buffer when wired."""
    from unittest.mock import MagicMock

    from voice_app.tts import TTSEngine

    mock_ref = MagicMock()
    tts = TTSEngine(rate=175, volume=0.9, ref_buffer=mock_ref)
    tts.interrupt()

    mock_ref.clear.assert_called_once()


def test_set_generation_clears_interrupt_flag():
    """set_generation() clears any previously set interrupt flag."""
    from voice_app.tts import TTSEngine

    tts = TTSEngine(rate=175, volume=0.9)
    tts._interrupt_flag.set()
    tts.set_generation(5)

    assert not tts._interrupt_flag.is_set()
    assert tts._current_gen_id == 5


def test_stale_chunks_are_discarded_by_run_loop():
    """
    Items with a superseded gen_id are silently dropped by the silent loop.
    No TTS_COMPLETE should be emitted for gen_id=1 when current is gen_id=2.
    """
    event_q: queue.Queue = queue.Queue()

    # No api_key → _run_silent_loop (no Deepgram WS, no network)
    from voice_app.tts import TTSEngine

    tts = TTSEngine(rate=175, volume=0.9, event_queue=event_q)
    tts.set_generation(2)
    tts._tts_queue.put(("old response", 1))  # stale gen_id=1

    tts.start()
    time.sleep(0.15)
    tts.stop()

    # Nothing should have been emitted for the stale item
    assert event_q.empty()


def test_finish_marker_emits_tts_complete_in_silent_loop():
    """_FINISH_MARKER triggers TTS_COMPLETE via _run_silent_loop (no api_key)."""
    event_q: queue.Queue = queue.Queue()

    from voice_app.tts import TTSEngine

    tts = TTSEngine(rate=175, volume=0.9, event_queue=event_q)
    tts.set_generation(3)

    tts.start()
    tts.speak("Hello", gen_id=3)
    tts.finish(gen_id=3)
    time.sleep(0.2)
    tts.stop()

    events = []
    while not event_q.empty():
        events.append(event_q.get_nowait())

    assert any(
        e.get("type") == "TTS_COMPLETE" and e.get("gen_id") == 3 for e in events
    ), f"Expected TTS_COMPLETE gen_id=3 in {events}"


def test_interrupted_finish_does_not_emit_tts_complete():
    """interrupt() before _FINISH_MARKER must suppress TTS_COMPLETE."""
    event_q: queue.Queue = queue.Queue()

    from voice_app.tts import TTSEngine

    tts = TTSEngine(rate=175, volume=0.9, event_queue=event_q)
    tts.set_generation(4)

    tts.start()
    tts.interrupt()  # set flag before finish marker is processed
    tts.finish(gen_id=4)
    time.sleep(0.2)
    tts.stop()

    events = []
    while not event_q.empty():
        events.append(event_q.get_nowait())

    assert not any(e.get("type") == "TTS_COMPLETE" for e in events), (
        f"TTS_COMPLETE must not fire after interrupt; got: {events}"
    )


def test_pyaudio_callback_returns_silence_on_empty_queue():
    """
    _pyaudio_callback falls back to silence when _playback_queue is empty.
    This verifies the real-time callback never blocks or raises.
    """
    from voice_app.tts import FRAME_BYTES, TTSEngine

    tts = TTSEngine(rate=175, volume=0.9)
    # Queue is empty; callback must not raise
    frame, flag = tts._pyaudio_callback(None, 320, None, None)

    import pyaudio

    assert frame == bytes(FRAME_BYTES)
    assert flag == pyaudio.paContinue


def test_pyaudio_callback_pushes_to_ref_buffer():
    """When ref_buffer is set, _pyaudio_callback pushes the frame into it."""
    from unittest.mock import MagicMock

    from voice_app.tts import FRAME_BYTES, TTSEngine

    mock_ref = MagicMock()
    tts = TTSEngine(rate=175, volume=0.9, ref_buffer=mock_ref)
    # Exactly FRAME_BYTES bytes with a recognisable pattern (i % 256)
    frame_data = bytes(i % 256 for i in range(FRAME_BYTES))
    tts._playback_queue.put(frame_data)

    tts._pyaudio_callback(None, 320, None, None)

    mock_ref.push.assert_called_once()
    _, pushed_frame = mock_ref.push.call_args[0]
    assert pushed_frame == frame_data
