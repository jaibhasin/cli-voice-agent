import os
from dataclasses import dataclass, field

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
    # If True, mic speech can interrupt TTS (barge-in). False avoids speaker bleed
    # triggering false interrupts; use headphones if True.
    # When aec.enabled is True this flag is ignored — AEC removes the echo so
    # barge-in works safely without headphones.
    barge_in_while_speaking: bool = False


@dataclass
class DeepgramConfig:
    """Deepgram live transcription settings."""

    # nova-3 is the current Deepgram flagship STT model (better accuracy than nova-2).
    model: str = "nova-3"
    language: str = "en"
    endpointing: int = 300
    utterance_end_ms: int = 1000
    smart_format: bool = True


@dataclass
class LLMConfig:
    """OpenAI response generation settings."""

    model: str = "gpt-4o-mini"
    temperature: float = 0.8
    max_tokens: int = 500


@dataclass
class AECConfig:
    """
    Software Acoustic Echo Cancellation (speexdsp) settings.

    When enabled = True:
      • TTSEngine records every played PCM frame into SpeakerReferenceBuffer.
      • AudioCapture runs each mic frame through AECProcessor (speexdsp) before
        queuing it for VAD and STT — removing the speaker signal from the mic.
      • The STT half-duplex silence-injection and echo-suppress timer are disabled.
      • barge_in_while_speaking is ignored (always on when AEC is active).

    When enabled = False (default):
      • The half-duplex path (silence injection + echo_suppress_tail_ms timer)
        is used exactly as before.  No AEC components are created.

    filter_length:       speexdsp tail length in samples; 2048 = 128 ms at 16 kHz.
    speaker_delay_ms:    Expected playback buffer delay; 0 = let speexdsp adapt.
    ref_buffer_frames:   Depth of the speaker reference ring-buffer (200 ≈ 4 s).
    """

    enabled: bool = False
    filter_length: int = 2048
    speaker_delay_ms: int = 0
    ref_buffer_frames: int = 200


@dataclass
class TTSConfig:
    """
    Speech synthesis settings (local piper-tts).

    echo_suppress_tail_ms is only used when aec.enabled is False (half-duplex fallback).
    """

    rate: int = 175
    volume: float = 0.9
    # After TTS ends, keep mic audio from reaching STT this many ms longer (room echo).
    # Only relevant when aec.enabled = false.
    echo_suppress_tail_ms: int = 350
    # Absolute or relative path to the piper .onnx voice model.
    # Falls back to the PIPER_MODEL environment variable.
    model_path: str = ""
    # Name or path of the piper executable.
    piper_bin: str = "piper"


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
    # aec must come last because it has a default (Python dataclass ordering rule).
    aec: AECConfig = field(default_factory=AECConfig)


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
        aec=AECConfig(**raw.get("aec", {})),
    )
