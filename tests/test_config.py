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
              model: "nova-3"
              language: "en"
              endpointing: 300
              utterance_end_ms: 1000
              smart_format: true
            llm:
              model: "gpt-4o-mini"
              temperature: 0.8
              max_tokens: 500
            tts:
              rate: 175
              volume: 0.9
              model_path: "models/test.onnx"
              piper_bin: "piper"
            aec:
              enabled: true
              filter_length: 2048
              speaker_delay_ms: 0
              ref_buffer_frames: 200
              residual_echo_guard_ms: 2000
              interrupted_echo_guard_ms: 600
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
    assert cfg.deepgram.model == "nova-3"
    assert cfg.llm.model == "gpt-4o-mini"
    assert cfg.tts.rate == 175
    assert cfg.tts.model_path == "models/test.onnx"
    assert cfg.tts.piper_bin == "piper"
    assert cfg.history.file == "conv.json"
    # AECConfig fields
    assert cfg.aec.enabled is True
    assert cfg.aec.filter_length == 2048
    assert cfg.aec.speaker_delay_ms == 0
    assert cfg.aec.ref_buffer_frames == 200
    assert cfg.aec.residual_echo_guard_ms == 2000
    assert cfg.aec.interrupted_echo_guard_ms == 600


def test_load_config_aec_defaults_when_section_absent(tmp_path):
    """AECConfig defaults to enabled=False when the aec: key is missing."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "system_prompt: hi\n"
        "audio:\n  sample_rate: 16000\n  frame_duration_ms: 20\n  channels: 1\n"
        "vad:\n  aggressiveness: 2\n  speech_start_frames: 6\n  ring_buffer_size: 8\n"
        "deepgram:\n  model: nova-3\n  language: en\n  endpointing: 300\n"
        "  utterance_end_ms: 1000\n  smart_format: true\n"
        "llm:\n  model: gpt-4o-mini\n  temperature: 0.8\n  max_tokens: 500\n"
        "tts:\n  rate: 175\n  volume: 0.9\n"
        "history:\n  file: x.json\n  max_messages_in_context: 50\n",
        encoding="utf-8",
    )
    env_vars = {"OPENAI_API_KEY": "sk-test-123", "DEEPGRAM_API_KEY": "dg-test-456"}
    with patch.dict(os.environ, env_vars):
        from voice_app.config import load_config

        cfg = load_config(str(config_file))

    assert cfg.aec.enabled is False
    assert cfg.aec.filter_length == 2048  # default


def test_load_config_raises_if_api_key_missing(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "system_prompt: hi\n"
        "audio:\n  sample_rate: 16000\n  frame_duration_ms: 20\n  channels: 1\n"
        "vad:\n  aggressiveness: 2\n  speech_start_frames: 6\n  ring_buffer_size: 8\n"
        "deepgram:\n  model: nova-3\n  language: en\n  endpointing: 300\n"
        "  utterance_end_ms: 1000\n  smart_format: true\n"
        "llm:\n  model: gpt-4o-mini\n  temperature: 0.8\n  max_tokens: 500\n"
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
