import pytest


@pytest.fixture(autouse=True)
def _tts_pyttsx3_in_tests(monkeypatch):
    """Production uses macOS `say` from a worker thread; tests keep mocking pyttsx3."""

    monkeypatch.setattr(
        "voice_app.tts.TTSEngine._use_macos_say",
        lambda self: False,
    )
