"""
Shared pytest fixtures for the voice-caller test suite.

The TTSEngine now uses PyAudio directly (no pyttsx3).  This autouse fixture
replaces pyaudio.PyAudio with a lightweight mock so that tests never open
a real audio output device.  Each test in test_audio_capture.py patches its
own pyaudio.PyAudio reference and is unaffected by this fixture.
"""

from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _mock_tts_pyaudio(monkeypatch):
    """
    Prevent TTSEngine.start() from opening a real PyAudio output device.

    The mock stream accepts start_stream() / stop_stream() / close() calls
    and returns False for is_active() so cleanup paths exit cleanly.
    Any test that explicitly patches voice_app.tts.pyaudio.PyAudio itself
    (e.g. via a 'with patch(...)' context manager) will shadow this fixture
    for the duration of that patch.
    """
    mock_stream = MagicMock()
    mock_stream.is_active.return_value = False

    mock_pa_instance = MagicMock()
    mock_pa_instance.open.return_value = mock_stream

    mock_pa_class = MagicMock(return_value=mock_pa_instance)
    monkeypatch.setattr("voice_app.tts.pyaudio.PyAudio", mock_pa_class)
