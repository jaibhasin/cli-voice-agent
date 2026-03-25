import time
from unittest.mock import MagicMock, patch


def make_engine_mock():
    mock = MagicMock()
    mock.connect.return_value = None
    return mock


def test_speak_queues_chunk_with_correct_gen_id():
    with patch("voice_app.tts.pyttsx3.init", return_value=make_engine_mock()):
        from voice_app.tts import TTSEngine

        tts = TTSEngine(rate=175, volume=0.9)
        tts.set_generation(1)
        tts.speak("Hello world", gen_id=1)
        item = tts._tts_queue.get_nowait()
        assert item == ("Hello world", 1)


def test_interrupt_drains_queue_and_sets_flag():
    with patch("voice_app.tts.pyttsx3.init", return_value=make_engine_mock()):
        from voice_app.tts import TTSEngine

        tts = TTSEngine(rate=175, volume=0.9)
        tts.set_generation(1)
        tts.speak("sentence one", gen_id=1)
        tts.speak("sentence two", gen_id=1)
        tts.speak("sentence three", gen_id=1)

        tts.interrupt()

        assert tts._tts_queue.empty()
        assert tts._interrupt_flag.is_set()


def test_stale_chunks_are_discarded_by_run_loop():
    mock_engine = make_engine_mock()

    with patch("voice_app.tts.pyttsx3.init", return_value=mock_engine):
        from voice_app.tts import TTSEngine

        tts = TTSEngine(rate=175, volume=0.9)
        tts.set_generation(2)
        tts._tts_queue.put(("old response", 1))

        tts.start()
        time.sleep(0.15)
        tts.stop()

    for call in mock_engine.say.call_args_list:
        assert "old response" not in str(call)


def test_set_generation_clears_interrupt_flag():
    with patch("voice_app.tts.pyttsx3.init", return_value=make_engine_mock()):
        from voice_app.tts import TTSEngine

        tts = TTSEngine(rate=175, volume=0.9)
        tts._interrupt_flag.set()
        tts.set_generation(5)

        assert not tts._interrupt_flag.is_set()
        assert tts._current_gen_id == 5
