import os
from dataclasses import dataclass

import yaml
from dotenv import load_dotenv


@dataclass
class AudioConfig:
    """Raw microphone capture settings."""

    sample_rate: int = 16000
    frame_duration_ms: int = 20
    channels: int = 1


@dataclass
class VADConfig:
    """Voice activity detection tuning."""

    aggressiveness: int = 2
    speech_start_frames: int = 6
    ring_buffer_size: int = 8


@dataclass
class DeepgramConfig:
    """Deepgram live transcription settings."""

    model: str = "nova-2"
    language: str = "en"
    endpointing: int = 300
    utterance_end_ms: int = 1000
    smart_format: bool = True


@dataclass
class LLMConfig:
    """OpenAI response generation settings."""

    model: str = "gpt-4o"
    temperature: float = 0.8
    max_tokens: int = 500


@dataclass
class TTSConfig:
    """Speech synthesis settings."""

    rate: int = 175
    volume: float = 0.9


@dataclass
class HistoryConfig:
    """Conversation history persistence settings."""

    file: str = "conversation_history.json"
    max_messages_in_context: int = 50


@dataclass
class AppConfig:
    """Application-wide configuration loaded from YAML and environment."""

    system_prompt: str
    audio: AudioConfig
    vad: VADConfig
    deepgram: DeepgramConfig
    llm: LLMConfig
    tts: TTSConfig
    history: HistoryConfig
    openai_api_key: str
    deepgram_api_key: str


def load_config(config_path: str = "config.yaml") -> AppConfig:
    """
    Load configuration from YAML plus API keys from the environment.

    Raises:
        KeyError: If required API keys are missing.
        FileNotFoundError: If the config path does not exist.
    """

    load_dotenv()

    with open(config_path, encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    return AppConfig(
        system_prompt=raw.get("system_prompt", "You are a friendly assistant."),
        audio=AudioConfig(**raw.get("audio", {})),
        vad=VADConfig(**raw.get("vad", {})),
        deepgram=DeepgramConfig(**raw.get("deepgram", {})),
        llm=LLMConfig(**raw.get("llm", {})),
        tts=TTSConfig(**raw.get("tts", {})),
        history=HistoryConfig(**raw.get("history", {})),
        openai_api_key=os.environ["OPENAI_API_KEY"],
        deepgram_api_key=os.environ["DEEPGRAM_API_KEY"],
    )
