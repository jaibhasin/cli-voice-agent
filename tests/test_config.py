import os
import textwrap
from unittest.mock import patch

import pytest


def test_load_config_reads_yaml_and_env(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        textwrap.dedent(
            """\
            system_prompt: "Hello!"
            audio:
              sample_rate: 16000
              frame_duration_ms: 20
              channels: 1
            vad:
              aggressiveness: 2
              speech_start_frames: 6
              ring_buffer_size: 8
            deepgram:
              model: "nova-2"
              language: "en"
              endpointing: 300
              utterance_end_ms: 1000
              smart_format: true
            llm:
              model: "gpt-4o"
              temperature: 0.8
              max_tokens: 500
            tts:
              rate: 175
              volume: 0.9
            history:
              file: "conv.json"
              max_messages_in_context: 50
            """
        ),
        encoding="utf-8",
    )

    env_vars = {
        "OPENAI_API_KEY": "sk-test-123",
        "DEEPGRAM_API_KEY": "dg-test-456",
    }

    with patch.dict(os.environ, env_vars):
        from voice_app.config import load_config

        cfg = load_config(str(config_file))

    assert cfg.system_prompt == "Hello!"
    assert cfg.openai_api_key == "sk-test-123"
    assert cfg.deepgram_api_key == "dg-test-456"
    assert cfg.audio.sample_rate == 16000
    assert cfg.vad.aggressiveness == 2
    assert cfg.deepgram.model == "nova-2"
    assert cfg.llm.model == "gpt-4o"
    assert cfg.tts.rate == 175
    assert cfg.history.file == "conv.json"


def test_load_config_raises_if_api_key_missing(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "system_prompt: hi\n"
        "audio:\n  sample_rate: 16000\n  frame_duration_ms: 20\n  channels: 1\n"
        "vad:\n  aggressiveness: 2\n  speech_start_frames: 6\n  ring_buffer_size: 8\n"
        "deepgram:\n  model: nova-2\n  language: en\n  endpointing: 300\n"
        "  utterance_end_ms: 1000\n  smart_format: true\n"
        "llm:\n  model: gpt-4o\n  temperature: 0.8\n  max_tokens: 500\n"
        "tts:\n  rate: 175\n  volume: 0.9\n"
        "history:\n  file: x.json\n  max_messages_in_context: 50\n",
        encoding="utf-8",
    )

    clean_env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("OPENAI_API_KEY", "DEEPGRAM_API_KEY")
    }
    with patch.dict(os.environ, clean_env, clear=True):
        from voice_app.config import load_config

        with pytest.raises(KeyError):
            load_config(str(config_file))
