# Voice LLM Full-Duplex Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI Python app that lets you hold a natural, interruptible voice conversation with GPT-4o — like a phone call.

**Architecture:** Five background threads (AudioCapture, VAD, STT, LLM, TTS) post events to a central queue consumed by the orchestrator on the main thread. A state machine (IDLE→LISTENING→PROCESSING→SPEAKING) drives all transitions. An interrupt mechanism using webrtcvad + a generation-ID pattern lets the user cut off the AI mid-sentence.

**Tech Stack:** Python 3.11+, OpenAI GPT-4o (streaming), Deepgram WebSocket STT (nova-2), webrtcvad, PyAudio, pyttsx3, PyYAML, python-dotenv

---

## Chunk 1: Foundation

### Task 1: Project Scaffold

**Files:**
- Create: `requirements.txt`
- Create: `config.yaml`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `voice_app/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create `requirements.txt`**

```text
openai>=1.30.0
deepgram-sdk>=3.5.0
pyaudio>=0.2.14
webrtcvad>=2.0.10
pyttsx3>=2.90
pyyaml>=6.0
python-dotenv>=1.0.0
pytest>=8.0.0
pytest-mock>=3.14.0
```

- [ ] **Step 2: Create `config.yaml`**

```yaml
# Personality — the AI's voice and style
system_prompt: |
  You are a friendly, casual companion. You speak naturally like a close
  friend would. Keep responses concise (2-3 sentences) since this is a
  voice conversation. Be warm, use casual language, avoid being formal.

# Raw audio capture settings
audio:
  sample_rate: 16000      # Hz — Deepgram and webrtcvad both expect 16kHz
  frame_duration_ms: 20   # webrtcvad requires 10, 20, or 30ms frames
  channels: 1             # Mono

# Voice Activity Detection — tunes interrupt sensitivity
vad:
  aggressiveness: 2       # 0 (least) to 3 (most) filtering of non-speech
  speech_start_frames: 6  # Frames in ring buffer that must be speech to trigger
  ring_buffer_size: 8     # Rolling window size — larger = slower but fewer false positives

# Deepgram live transcription settings
deepgram:
  model: "nova-2"
  language: "en"
  endpointing: 300        # ms of silence before sending is_final (fast response)
  utterance_end_ms: 1000  # ms of silence to signal "user is done speaking"
  smart_format: true      # Adds punctuation, numbers formatted correctly

# OpenAI LLM settings
llm:
  model: "gpt-4o"
  temperature: 0.8
  max_tokens: 500

# Text-to-speech settings (macOS NSSpeechSynthesizer via pyttsx3)
tts:
  rate: 175               # Words per minute (default macOS is ~200)
  volume: 0.9

# Conversation persistence
history:
  file: "conversation_history.json"
  max_messages_in_context: 50   # Trims oldest messages when context grows
```

- [ ] **Step 3: Create `.env.example`**

```bash
# Copy this file to .env and fill in your API keys
OPENAI_API_KEY=sk-...
DEEPGRAM_API_KEY=...
```

- [ ] **Step 4: Create `.gitignore`**

```gitignore
.env
conversation_history.json
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
dist/
build/
.venv/
venv/
```

- [ ] **Step 5: Create empty `voice_app/__init__.py` and `tests/__init__.py`**

Both files should be empty (zero bytes). They tell Python these are packages.

- [ ] **Step 6: Install prerequisites and dependencies**

```bash
brew install portaudio
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Expected: No errors. `pip show openai deepgram-sdk webrtcvad pyaudio pyttsx3` should list all packages.

- [ ] **Step 7: Commit**

```bash
git add requirements.txt config.yaml .env.example .gitignore voice_app/__init__.py tests/__init__.py
git commit -m "chore: project scaffold — deps, config, gitignore"
```

---

### Task 2: Config Loader

**Files:**
- Create: `voice_app/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
import os
import textwrap
import pytest
from unittest.mock import patch


def test_load_config_reads_yaml_and_env(tmp_path):
    """
    load_config() should read a YAML file and pick up API keys from environment.
    This test writes a minimal config, sets env vars, and asserts the dataclass
    fields are populated correctly.
    """
    # Arrange: write a minimal config.yaml to a temp dir
    config_file = tmp_path / "config.yaml"
    config_file.write_text(textwrap.dedent("""\
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
    """))

    env_vars = {
        "OPENAI_API_KEY": "sk-test-123",
        "DEEPGRAM_API_KEY": "dg-test-456",
    }

    with patch.dict(os.environ, env_vars):
        from voice_app.config import load_config
        cfg = load_config(str(config_file))

    # Assert top-level fields
    assert cfg.system_prompt == "Hello!"
    assert cfg.openai_api_key == "sk-test-123"
    assert cfg.deepgram_api_key == "dg-test-456"

    # Assert nested dataclass fields
    assert cfg.audio.sample_rate == 16000
    assert cfg.vad.aggressiveness == 2
    assert cfg.deepgram.model == "nova-2"
    assert cfg.llm.model == "gpt-4o"
    assert cfg.tts.rate == 175
    assert cfg.history.file == "conv.json"


def test_load_config_raises_if_api_key_missing(tmp_path):
    """Missing API keys should raise KeyError, not silently return None."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("system_prompt: hi\naudio:\n  sample_rate: 16000\n  frame_duration_ms: 20\n  channels: 1\nvad:\n  aggressiveness: 2\n  speech_start_frames: 6\n  ring_buffer_size: 8\ndeepgram:\n  model: nova-2\n  language: en\n  endpointing: 300\n  utterance_end_ms: 1000\n  smart_format: true\nllm:\n  model: gpt-4o\n  temperature: 0.8\n  max_tokens: 500\ntts:\n  rate: 175\n  volume: 0.9\nhistory:\n  file: x.json\n  max_messages_in_context: 50\n")

    # Remove any existing API keys from env
    clean_env = {k: v for k, v in os.environ.items()
                 if k not in ("OPENAI_API_KEY", "DEEPGRAM_API_KEY")}
    with patch.dict(os.environ, clean_env, clear=True):
        from voice_app.config import load_config
        with pytest.raises(KeyError):
            load_config(str(config_file))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_config.py -v
```

Expected: `FAILED` — `ModuleNotFoundError: No module named 'voice_app.config'`

- [ ] **Step 3: Write `voice_app/config.py`**

```python
# voice_app/config.py
#
# Loads the app's configuration from two sources:
#   1. config.yaml  — all tunable parameters (safe to commit)
#   2. .env         — API keys (never committed)
#
# Returns a single AppConfig dataclass so every module gets typed access
# to its settings without importing yaml/os/dotenv themselves.

import os
from dataclasses import dataclass
import yaml
from dotenv import load_dotenv


# ── Per-section dataclasses ────────────────────────────────────────────────

@dataclass
class AudioConfig:
    """Raw microphone capture settings. sample_rate must match webrtcvad's expectation."""
    sample_rate: int = 16000
    frame_duration_ms: int = 20
    channels: int = 1


@dataclass
class VADConfig:
    """
    Tunes how aggressively the local Voice Activity Detector filters noise.
    Higher aggressiveness = fewer false positives but may miss soft speech.
    """
    aggressiveness: int = 2
    speech_start_frames: int = 6   # N frames in ring buffer that must be speech
    ring_buffer_size: int = 8      # Rolling window size (M in "N of M" check)


@dataclass
class DeepgramConfig:
    """Deepgram live WebSocket transcription options."""
    model: str = "nova-2"
    language: str = "en"
    endpointing: int = 300         # ms of silence → send is_final
    utterance_end_ms: int = 1000   # ms of silence → fire utterance_end event
    smart_format: bool = True


@dataclass
class LLMConfig:
    """OpenAI chat completion settings."""
    model: str = "gpt-4o"
    temperature: float = 0.8
    max_tokens: int = 500


@dataclass
class TTSConfig:
    """pyttsx3 speech synthesis settings."""
    rate: int = 175
    volume: float = 0.9


@dataclass
class HistoryConfig:
    """Persistent conversation history settings."""
    file: str = "conversation_history.json"
    max_messages_in_context: int = 50


# ── Root config dataclass ──────────────────────────────────────────────────

@dataclass
class AppConfig:
    """Single object that holds every configuration value the app needs."""
    system_prompt: str
    audio: AudioConfig
    vad: VADConfig
    deepgram: DeepgramConfig
    llm: LLMConfig
    tts: TTSConfig
    history: HistoryConfig
    openai_api_key: str
    deepgram_api_key: str


# ── Loader ─────────────────────────────────────────────────────────────────

def load_config(config_path: str = "config.yaml") -> AppConfig:
    """
    Load configuration from a YAML file and .env.

    Raises:
        KeyError: if OPENAI_API_KEY or DEEPGRAM_API_KEY are not set in the environment.
        FileNotFoundError: if config_path does not exist.
    """
    # Load .env into os.environ (no-op if .env doesn't exist)
    load_dotenv()

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    return AppConfig(
        system_prompt=raw.get("system_prompt", "You are a friendly assistant."),
        audio=AudioConfig(**raw.get("audio", {})),
        vad=VADConfig(**raw.get("vad", {})),
        deepgram=DeepgramConfig(**raw.get("deepgram", {})),
        llm=LLMConfig(**raw.get("llm", {})),
        tts=TTSConfig(**raw.get("tts", {})),
        history=HistoryConfig(**raw.get("history", {})),
        # Raises KeyError with a clear message if either key is missing
        openai_api_key=os.environ["OPENAI_API_KEY"],
        deepgram_api_key=os.environ["DEEPGRAM_API_KEY"],
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_config.py -v
```

Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add voice_app/config.py tests/test_config.py
git commit -m "feat: config loader — typed dataclass from YAML + .env"
```

---

### Task 3: State Machine

**Files:**
- Create: `voice_app/state_machine.py`
- Create: `tests/test_state_machine.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_state_machine.py
#
# The state machine is a pure data structure with no I/O — ideal for
# exhaustive unit testing. We test every valid transition and every
# invalid/ignored transition.

import pytest
from voice_app.state_machine import StateMachine, State, AppEvent


class TestValidTransitions:
    """Every row in the TRANSITIONS table should work correctly."""

    def test_idle_speech_detected_goes_to_listening(self):
        sm = StateMachine()
        assert sm.state == State.IDLE
        result = sm.transition(AppEvent.SPEECH_DETECTED)
        assert result == State.LISTENING
        assert sm.state == State.LISTENING

    def test_listening_utterance_complete_goes_to_processing(self):
        sm = StateMachine()
        sm.state = State.LISTENING
        result = sm.transition(AppEvent.UTTERANCE_COMPLETE)
        assert result == State.PROCESSING
        assert sm.state == State.PROCESSING

    def test_processing_first_tts_chunk_goes_to_speaking(self):
        sm = StateMachine()
        sm.state = State.PROCESSING
        result = sm.transition(AppEvent.FIRST_TTS_CHUNK)
        assert result == State.SPEAKING
        assert sm.state == State.SPEAKING

    def test_speaking_tts_complete_goes_to_idle(self):
        sm = StateMachine()
        sm.state = State.SPEAKING
        result = sm.transition(AppEvent.TTS_COMPLETE)
        assert result == State.IDLE
        assert sm.state == State.IDLE

    def test_speaking_interrupt_goes_to_listening(self):
        sm = StateMachine()
        sm.state = State.SPEAKING
        result = sm.transition(AppEvent.INTERRUPT)
        assert result == State.LISTENING
        assert sm.state == State.LISTENING

    def test_processing_interrupt_goes_to_listening(self):
        sm = StateMachine()
        sm.state = State.PROCESSING
        result = sm.transition(AppEvent.INTERRUPT)
        assert result == State.LISTENING
        assert sm.state == State.LISTENING


class TestErrorTransition:
    """ERROR from any state resets to IDLE."""

    def test_error_from_idle(self):
        sm = StateMachine()
        sm.state = State.IDLE
        result = sm.transition(AppEvent.ERROR)
        assert result == State.IDLE

    def test_error_from_speaking(self):
        sm = StateMachine()
        sm.state = State.SPEAKING
        result = sm.transition(AppEvent.ERROR)
        assert result == State.IDLE
        assert sm.state == State.IDLE

    def test_error_from_processing(self):
        sm = StateMachine()
        sm.state = State.PROCESSING
        result = sm.transition(AppEvent.ERROR)
        assert result == State.IDLE


class TestShutdownTransition:
    """SHUTDOWN from any state returns None (signals exit)."""

    def test_shutdown_returns_none(self):
        sm = StateMachine()
        result = sm.transition(AppEvent.SHUTDOWN)
        assert result is None

    def test_shutdown_from_speaking(self):
        sm = StateMachine()
        sm.state = State.SPEAKING
        result = sm.transition(AppEvent.SHUTDOWN)
        assert result is None


class TestIgnoredTransitions:
    """Invalid transitions (not in the table) should return None and not change state."""

    def test_idle_ignores_utterance_complete(self):
        sm = StateMachine()
        result = sm.transition(AppEvent.UTTERANCE_COMPLETE)
        assert result is None
        assert sm.state == State.IDLE

    def test_listening_ignores_tts_complete(self):
        sm = StateMachine()
        sm.state = State.LISTENING
        result = sm.transition(AppEvent.TTS_COMPLETE)
        assert result is None
        assert sm.state == State.LISTENING

    def test_idle_ignores_interrupt(self):
        sm = StateMachine()
        result = sm.transition(AppEvent.INTERRUPT)
        assert result is None
        assert sm.state == State.IDLE
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_state_machine.py -v
```

Expected: `FAILED` — `ModuleNotFoundError: No module named 'voice_app.state_machine'`

- [ ] **Step 3: Write `voice_app/state_machine.py`**

```python
# voice_app/state_machine.py
#
# Defines all conversation states and the valid transitions between them.
#
# The state machine is the "brain" of the app — it enforces that only
# valid state transitions happen. For example, you can't go from IDLE
# directly to SPEAKING without passing through LISTENING and PROCESSING.
#
# All threads communicate by posting events to a queue; the orchestrator
# reads those events and calls sm.transition() to advance the machine.

from enum import Enum, auto
from typing import Optional


# ── States ─────────────────────────────────────────────────────────────────

class State(Enum):
    """
    The four conversation states the app can be in at any moment.

        IDLE       — Waiting. Mic is live, VAD is watching, nobody is speaking.
        LISTENING  — User is speaking. Deepgram is receiving audio.
        PROCESSING — User finished. LLM is generating a response.
        SPEAKING   — AI is speaking via TTS. User can interrupt.
    """
    IDLE = auto()
    LISTENING = auto()
    PROCESSING = auto()
    SPEAKING = auto()


# ── Events ─────────────────────────────────────────────────────────────────

class AppEvent(Enum):
    """
    Events that threads post to the event queue. The orchestrator maps
    each event to a state transition via the TRANSITIONS table below.
    """
    SPEECH_DETECTED = auto()    # VAD: user started speaking
    UTTERANCE_COMPLETE = auto() # Deepgram: user finished speaking, text ready
    FIRST_TTS_CHUNK = auto()    # LLM: first sentence sent to TTS queue
    TTS_COMPLETE = auto()       # TTS: all chunks spoken
    INTERRUPT = auto()          # VAD: user spoke while AI was speaking
    ERROR = auto()              # Any thread: unrecoverable error
    SHUTDOWN = auto()           # Ctrl-C or graceful exit


# ── Transition table ────────────────────────────────────────────────────────
#
# (current_state, event) → next_state
#
# Only valid transitions are listed. Anything not in this table is a no-op
# (returns None, state unchanged). This prevents race conditions where a
# stale event arrives after the state has already moved on.

TRANSITIONS: dict[tuple[State, AppEvent], State] = {
    (State.IDLE,       AppEvent.SPEECH_DETECTED):    State.LISTENING,
    (State.LISTENING,  AppEvent.UTTERANCE_COMPLETE): State.PROCESSING,
    (State.PROCESSING, AppEvent.FIRST_TTS_CHUNK):    State.SPEAKING,
    (State.SPEAKING,   AppEvent.TTS_COMPLETE):       State.IDLE,
    (State.SPEAKING,   AppEvent.INTERRUPT):          State.LISTENING,
    (State.PROCESSING, AppEvent.INTERRUPT):          State.LISTENING,
}


# ── State machine ──────────────────────────────────────────────────────────

class StateMachine:
    """
    Drives conversation state. Thread-safe reads are fine; only the
    orchestrator (main thread) calls transition(), so no locking needed.
    """

    def __init__(self) -> None:
        self.state: State = State.IDLE

    def transition(self, event: AppEvent) -> Optional[State]:
        """
        Attempt a state transition triggered by `event`.

        Returns:
            The new State if the transition was valid.
            None if the transition was ignored (no entry in table) OR if
            event is SHUTDOWN (signals the orchestrator to exit).

        Special cases handled before the table lookup:
            ERROR   — always resets to IDLE regardless of current state.
            SHUTDOWN — always returns None (exit signal).
        """
        # Special: ERROR resets to IDLE from any state
        if event == AppEvent.ERROR:
            self.state = State.IDLE
            return State.IDLE

        # Special: SHUTDOWN signals exit — do not change state
        if event == AppEvent.SHUTDOWN:
            return None

        # Normal: look up in transition table
        new_state = TRANSITIONS.get((self.state, event))
        if new_state is not None:
            self.state = new_state
        return new_state
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_state_machine.py -v
```

Expected: `13 passed`

- [ ] **Step 5: Commit**

```bash
git add voice_app/state_machine.py tests/test_state_machine.py
git commit -m "feat: conversation state machine with full transition table"
```

---

### Task 4: Conversation History

**Files:**
- Create: `voice_app/history.py`
- Create: `tests/test_history.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_history.py
#
# History is a plain JSON list of {"role": ..., "content": ...} dicts —
# the same format OpenAI's messages array expects, so no conversion needed.

import json
import pytest
from voice_app.history import load_history, save_history, append_message


def test_load_returns_empty_list_when_file_missing(tmp_path):
    """Missing history file is normal on first run — must not crash."""
    result = load_history(str(tmp_path / "no_such_file.json"))
    assert result == []


def test_save_and_reload_round_trip(tmp_path):
    """Data written by save_history should be identical when loaded back."""
    filepath = str(tmp_path / "history.json")
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    save_history(filepath, messages)
    loaded = load_history(filepath)
    assert loaded == messages


def test_append_adds_message_and_persists(tmp_path):
    """append_message should add to existing history and write to disk."""
    filepath = str(tmp_path / "history.json")
    # Start with one existing message
    save_history(filepath, [{"role": "user", "content": "First"}])

    result = append_message(filepath, "assistant", "Second", max_messages=50)

    assert len(result) == 2
    assert result[1] == {"role": "assistant", "content": "Second"}

    # Verify it was persisted to disk
    on_disk = load_history(filepath)
    assert len(on_disk) == 2


def test_append_trims_to_max_messages(tmp_path):
    """History should never grow beyond max_messages after trimming."""
    filepath = str(tmp_path / "history.json")
    # Pre-populate with 5 messages
    initial = [{"role": "user", "content": str(i)} for i in range(5)]
    save_history(filepath, initial)

    # Append with max=5 — should drop oldest and keep exactly 5
    result = append_message(filepath, "assistant", "new", max_messages=5)

    assert len(result) == 5
    # The oldest message should have been dropped
    assert result[0]["content"] == "1"
    assert result[-1]["content"] == "new"


def test_append_creates_file_if_missing(tmp_path):
    """append_message on a non-existent file should create it."""
    filepath = str(tmp_path / "new_history.json")
    result = append_message(filepath, "user", "Hello", max_messages=50)
    assert result == [{"role": "user", "content": "Hello"}]
    assert (tmp_path / "new_history.json").exists()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_history.py -v
```

Expected: `FAILED` — `ModuleNotFoundError: No module named 'voice_app.history'`

- [ ] **Step 3: Write `voice_app/history.py`**

```python
# voice_app/history.py
#
# Saves and loads conversation history as a JSON file on disk.
#
# The history format is a list of OpenAI-compatible message dicts:
#   [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
#
# This lets us pass the history directly to the OpenAI messages parameter
# without any conversion. The file is written on every new message so that
# a crash never loses more than the last turn.

import json
import os
from typing import List, Dict


def load_history(filepath: str) -> List[Dict]:
    """
    Load conversation history from a JSON file.
    Returns an empty list if the file doesn't exist yet (first run).
    """
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_history(filepath: str, messages: List[Dict]) -> None:
    """
    Persist the full message list to disk, overwriting the previous file.
    Uses indent=2 for human-readable output.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(messages, f, indent=2, ensure_ascii=False)


def append_message(
    filepath: str, role: str, content: str, max_messages: int
) -> List[Dict]:
    """
    Append one message to the history and persist it.

    Trims the list to `max_messages` by dropping the oldest entries first,
    so the file never grows unbounded.

    Returns the updated message list (useful for callers that need to pass
    it immediately to the LLM without re-reading from disk).
    """
    messages = load_history(filepath)
    messages.append({"role": role, "content": content})

    # Keep only the most recent max_messages entries
    if len(messages) > max_messages:
        messages = messages[-max_messages:]

    save_history(filepath, messages)
    return messages
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_history.py -v
```

Expected: `5 passed`

- [ ] **Step 5: Run all foundation tests together**

```bash
pytest tests/ -v
```

Expected: All tests pass (config + state_machine + history).

- [ ] **Step 6: Commit**

```bash
git add voice_app/history.py tests/test_history.py
git commit -m "feat: JSON conversation history with auto-trim"
```

---

## Chunk 2: Audio Pipeline

### Task 5: Audio Capture Thread

**Files:**
- Create: `voice_app/audio_capture.py`
- Create: `tests/test_audio_capture.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_audio_capture.py
#
# AudioCapture uses real hardware (microphone), so we mock PyAudio.
# We test the queue fan-out and the stop/start lifecycle, not the
# actual audio bytes.

import queue
import threading
import time
import pytest
from unittest.mock import MagicMock, patch


def make_mock_pyaudio(frames: list[bytes]):
    """Helper: returns a mock PyAudio that yields frames from a list."""
    mock_stream = MagicMock()
    # Each call to read() returns the next frame, then blocks forever
    side_effects = frames + [b"\x00" * 640] * 1000
    mock_stream.read.side_effect = side_effects

    mock_pa = MagicMock()
    mock_pa.open.return_value = mock_stream
    return mock_pa, mock_stream


def test_audio_capture_fans_out_to_multiple_queues():
    """
    Every captured frame must be placed into ALL registered consumer queues.
    This is critical because VAD and STT both need the same audio.
    """
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
        time.sleep(0.1)   # Let the capture thread run a few cycles
        capture.stop()

    # Both queues should have received the same frames
    items_q1 = list(q1.queue)
    items_q2 = list(q2.queue)
    assert len(items_q1) >= 3
    assert items_q1[:3] == items_q2[:3]


def test_audio_capture_stops_cleanly():
    """stop() must join the thread without hanging."""
    from voice_app.audio_capture import AudioCapture

    mock_pa, _ = make_mock_pyaudio([b"\x00" * 640] * 100)

    with patch("voice_app.audio_capture.pyaudio.PyAudio", return_value=mock_pa):
        capture = AudioCapture(sample_rate=16000, channels=1)
        capture.start()
        time.sleep(0.05)
        capture.stop()  # Must not block indefinitely

    # If we get here without hanging, the test passes
    assert True
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_audio_capture.py -v
```

Expected: `FAILED` — `ModuleNotFoundError`

- [ ] **Step 3: Write `voice_app/audio_capture.py`**

```python
# voice_app/audio_capture.py
#
# Captures audio from the microphone in a dedicated background thread
# and distributes frames to all registered consumer queues.
#
# Why fan-out to multiple queues?
#   VAD needs every frame for interrupt detection (<160ms latency).
#   STT needs every frame for accurate transcription.
#   They run independently and at different rates — separate queues
#   let each consumer fall behind without blocking the other.
#
# Frame size: 20ms at 16kHz = 320 samples × 2 bytes = 640 bytes.
# webrtcvad requires exactly 10ms, 20ms, or 30ms frames — we use 20ms.

import queue
import threading
import pyaudio


class AudioCapture:
    """
    Runs a mic capture loop in a daemon thread.
    Call add_consumer(q) before start() to register a queue.
    """

    # webrtcvad only accepts 10ms, 20ms, or 30ms frame durations
    FRAME_DURATION_MS = 20

    def __init__(self, sample_rate: int, channels: int) -> None:
        self.sample_rate = sample_rate
        self.channels = channels

        # Number of raw audio samples per frame
        # e.g. 16000 Hz × 0.020 s = 320 samples → 640 bytes (int16)
        self.chunk_size = int(sample_rate * self.FRAME_DURATION_MS / 1000)

        self._queues: list[queue.Queue] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # PyAudio objects — created in start(), torn down in stop()
        self._pa = pyaudio.PyAudio()
        self._stream = None

    def add_consumer(self, q: queue.Queue) -> None:
        """Register a queue to receive audio frames. Call before start()."""
        self._queues.append(q)

    def start(self) -> None:
        """Open the microphone stream and begin capturing frames."""
        self._stream = self._pa.open(
            format=pyaudio.paInt16,       # 16-bit signed integers
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
        )
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="AudioCapture"
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the capture thread to stop and wait for it to exit."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        self._pa.terminate()

    def _capture_loop(self) -> None:
        """
        Main loop: read one frame from the mic and push it to all consumers.
        Frames are dropped (not queued) if a consumer's queue is full — this
        prevents slow consumers from causing unbounded memory growth.
        """
        while not self._stop_event.is_set():
            try:
                # exception_on_overflow=False: don't crash on buffer overrun
                frame = self._stream.read(
                    self.chunk_size, exception_on_overflow=False
                )
                for q in self._queues:
                    try:
                        q.put_nowait(frame)
                    except queue.Full:
                        pass  # Drop frame; consumer is too slow
            except Exception:
                break  # Stream closed or device disconnected
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_audio_capture.py -v
```

Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add voice_app/audio_capture.py tests/test_audio_capture.py
git commit -m "feat: audio capture thread with multi-queue fan-out"
```

---

### Task 6: VAD Detector

**Files:**
- Create: `voice_app/vad.py`
- Create: `tests/test_vad.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vad.py
#
# webrtcvad works on real audio bytes. We synthesize silent frames
# (all zeros) and voiced frames (random non-zero bytes) to test the
# ring buffer logic without needing a microphone.
#
# A "silent frame" at 16kHz/20ms = 640 zero bytes → webrtcvad says not speech.
# A "voiced frame" is a valid PCM frame with energy → webrtcvad may say speech.
# We mock webrtcvad.Vad.is_speech() for deterministic tests.

import pytest
from unittest.mock import patch, MagicMock
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
    """Ring buffer full of silence → speech not detected."""
    vad = make_vad()
    with patch.object(vad.vad, "is_speech", return_value=False):
        results = [vad.process_frame(SILENT_FRAME) for _ in range(10)]
    assert not any(results)


def test_speech_detected_when_threshold_met():
    """
    With speech_start_frames=3 and ring_buffer_size=5:
    After 3 voiced frames in the buffer, process_frame returns True.
    """
    vad = make_vad(speech_start_frames=3, ring_buffer_size=5)
    with patch.object(vad.vad, "is_speech", return_value=True):
        results = [vad.process_frame(VOICED_FRAME) for _ in range(5)]
    # By frame 3, threshold is met
    assert results[2] is True
    assert results[4] is True


def test_speech_not_detected_below_threshold():
    """
    With speech_start_frames=4 and ring_buffer_size=5:
    Only 3 voiced frames → should NOT trigger.
    """
    vad = make_vad(speech_start_frames=4, ring_buffer_size=5)

    call_count = 0
    def is_speech_side_effect(frame, rate):
        nonlocal call_count
        call_count += 1
        return call_count <= 3  # First 3 calls return True, rest False

    with patch.object(vad.vad, "is_speech", side_effect=is_speech_side_effect):
        results = [vad.process_frame(VOICED_FRAME) for _ in range(5)]

    # Never reaches 4 voiced frames in buffer
    assert not any(results)


def test_reset_clears_ring_buffer():
    """After reset(), previously accumulated voiced frames are forgotten."""
    vad = make_vad(speech_start_frames=3, ring_buffer_size=5)
    with patch.object(vad.vad, "is_speech", return_value=True):
        # Fill buffer with voiced frames
        for _ in range(3):
            vad.process_frame(VOICED_FRAME)

    # Reset clears buffer
    vad.reset()

    with patch.object(vad.vad, "is_speech", return_value=False):
        result = vad.process_frame(SILENT_FRAME)
    assert result is False


def test_vad_error_treated_as_silence():
    """If webrtcvad raises an exception, the frame is treated as silence."""
    vad = make_vad(speech_start_frames=2, ring_buffer_size=3)
    with patch.object(vad.vad, "is_speech", side_effect=Exception("bad frame")):
        results = [vad.process_frame(VOICED_FRAME) for _ in range(5)]
    assert not any(results)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_vad.py -v
```

Expected: `FAILED` — `ModuleNotFoundError`

- [ ] **Step 3: Write `voice_app/vad.py`**

```python
# voice_app/vad.py
#
# Wraps Google's WebRTC Voice Activity Detector with a ring buffer.
#
# Why a ring buffer?
#   webrtcvad classifies each 20ms frame independently as speech or silence.
#   A single voiced frame might be a cough, click, or background noise.
#   By requiring N voiced frames out of the last M frames before declaring
#   "speech detected", we smooth out false positives dramatically.
#
# Two uses in the app:
#   1. IDLE state: detect when the user starts speaking → SPEECH_DETECTED event
#   2. SPEAKING state: detect when the user interrupts → INTERRUPT event
#   The same VADDetector instance handles both use cases.

import collections
import webrtcvad


class VADDetector:
    """
    Classifies mic frames as speech or silence using a sliding ring buffer.

    Args:
        sample_rate:        Audio sample rate in Hz (must be 8000, 16000, 32000, or 48000).
        aggressiveness:     webrtcvad aggressiveness 0-3 (0=least filtering, 3=most).
        ring_buffer_size:   Size M of the rolling window (number of recent frames).
        speech_start_frames: Number N of voiced frames in the window to fire speech.
    """

    def __init__(
        self,
        sample_rate: int,
        aggressiveness: int,
        ring_buffer_size: int,
        speech_start_frames: int,
    ) -> None:
        # The underlying C-extension VAD
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = sample_rate
        self.ring_buffer_size = ring_buffer_size
        self.speech_start_frames = speech_start_frames

        # Ring buffer: stores True/False (is_speech) for the last M frames
        # maxlen automatically drops the oldest entry when full
        self._ring_buffer: collections.deque[bool] = collections.deque(
            maxlen=ring_buffer_size
        )

    def process_frame(self, frame: bytes) -> bool:
        """
        Process one 20ms PCM frame and return True if speech onset is detected.

        Handles webrtcvad exceptions gracefully — bad frames are treated as
        silence so a corrupt audio buffer doesn't crash the app.
        """
        try:
            is_speech = self.vad.is_speech(frame, self.sample_rate)
        except Exception:
            # Malformed frame (wrong length, unsupported rate) → treat as silence
            is_speech = False

        self._ring_buffer.append(is_speech)

        # Count how many of the last M frames were voiced
        num_voiced = sum(self._ring_buffer)
        return num_voiced >= self.speech_start_frames

    def reset(self) -> None:
        """
        Clear the ring buffer. Called after an interrupt is processed so that
        the leftover voiced frames from the interruption don't immediately
        re-trigger speech detection.
        """
        self._ring_buffer.clear()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_vad.py -v
```

Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add voice_app/vad.py tests/test_vad.py
git commit -m "feat: VAD detector with ring buffer for interrupt detection"
```

---

### Task 7: Deepgram STT Client

**Files:**
- Create: `voice_app/stt.py`

> Note: No unit tests here — Deepgram's WebSocket requires real network I/O. Integration testing is done in Task 12 (manual smoke test).

- [ ] **Step 1: Write `voice_app/stt.py`**

```python
# voice_app/stt.py
#
# Streams mic audio to Deepgram's live transcription WebSocket and posts
# events to the orchestrator's event queue.
#
# Why asyncio inside a thread?
#   The Deepgram Python SDK uses async/await internally. We run it inside
#   a dedicated thread with its own asyncio event loop so it doesn't
#   interfere with PyAudio (blocking I/O) or pyttsx3.
#
# Events posted to event_queue:
#   {"type": "TRANSCRIPT_INTERIM", "text": "..."}  — live partial result
#   {"type": "UTTERANCE_COMPLETE",  "text": "..."}  — final result, user done

import asyncio
import queue
import threading
import logging

from deepgram import (
    DeepgramClient,
    LiveOptions,
    LiveTranscriptionEvents,
)

logger = logging.getLogger(__name__)


class STTClient:
    """
    Background thread that maintains a Deepgram WebSocket connection,
    feeds it audio frames, and forwards transcription events.
    """

    def __init__(
        self,
        api_key: str,
        config,                     # DeepgramConfig dataclass
        audio_queue: queue.Queue,   # Receives raw PCM frames from AudioCapture
        event_queue: queue.Queue,   # Posts events to the Orchestrator
    ) -> None:
        self.api_key = api_key
        self.config = config
        self.audio_queue = audio_queue
        self.event_queue = event_queue

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Launch the STT thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="STTClient"
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the STT thread to stop and wait for it."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        """Entry point for the STT thread — runs its own asyncio event loop."""
        asyncio.run(self._async_run())

    async def _async_run(self) -> None:
        """
        Opens the Deepgram WebSocket, registers callbacks, then feeds audio
        frames until the stop event is set.
        """
        dg = DeepgramClient(self.api_key)
        connection = dg.listen.asyncwebsocket.v("1")

        # ── Callbacks ────────────────────────────────────────────────────

        async def on_transcript(conn, result, **kwargs):
            """
            Called by Deepgram on every transcription result.

            result.is_final      → Deepgram is confident this sentence is done
            result.speech_final  → Deepgram detected end-of-utterance (pause)
            """
            try:
                text = result.channel.alternatives[0].transcript.strip()
            except (AttributeError, IndexError):
                return

            if not text:
                return

            if result.is_final:
                if result.speech_final:
                    # User finished speaking — send to LLM
                    self.event_queue.put({
                        "type": "UTTERANCE_COMPLETE",
                        "text": text,
                    })
                else:
                    # Intermediate final — show live in terminal
                    self.event_queue.put({
                        "type": "TRANSCRIPT_INTERIM",
                        "text": text,
                    })

        async def on_error(conn, error, **kwargs):
            logger.error("Deepgram error: %s", error)
            self.event_queue.put({"type": "ERROR", "error": str(error)})

        connection.on(LiveTranscriptionEvents.Transcript, on_transcript)
        connection.on(LiveTranscriptionEvents.Error, on_error)

        # ── Connect ───────────────────────────────────────────────────────

        options = LiveOptions(
            model=self.config.model,
            language=self.config.language,
            smart_format=self.config.smart_format,
            endpointing=self.config.endpointing,
            utterance_end_ms=str(self.config.utterance_end_ms),
        )

        started = await connection.start(options)
        if not started:
            self.event_queue.put({"type": "ERROR", "error": "Deepgram failed to connect"})
            return

        logger.debug("Deepgram WebSocket connected")

        # ── Feed audio frames ─────────────────────────────────────────────

        while not self._stop_event.is_set():
            try:
                frame = self.audio_queue.get(timeout=0.1)
                await connection.send(frame)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error("STT send error: %s", e)
                break

        await connection.finish()
        logger.debug("Deepgram WebSocket closed")
```

- [ ] **Step 2: Commit**

```bash
git add voice_app/stt.py
git commit -m "feat: Deepgram live WebSocket STT client"
```

---

## Chunk 3: Output Pipeline

### Task 8: TTS Engine

**Files:**
- Create: `voice_app/tts.py`
- Create: `tests/test_tts.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tts.py
#
# TTS wraps pyttsx3 which calls macOS NSSpeechSynthesizer under the hood.
# We mock pyttsx3 completely to test the interrupt logic, generation ID
# gating, and queue draining without making any actual sound.

import time
import queue
import pytest
from unittest.mock import MagicMock, patch, call


def make_engine_mock():
    """Returns a mock pyttsx3 engine that records calls."""
    mock = MagicMock()
    # connect() returns a callable we don't need to track
    mock.connect.return_value = None
    return mock


def test_speak_queues_chunk_with_correct_gen_id():
    """speak() should put (text, gen_id) into the internal queue."""
    with patch("voice_app.tts.pyttsx3.init", return_value=make_engine_mock()):
        from voice_app.tts import TTSEngine
        tts = TTSEngine(rate=175, volume=0.9)
        tts.set_generation(1)
        tts.speak("Hello world", gen_id=1)
        # Peek at the internal queue without consuming
        item = tts._tts_queue.get_nowait()
        assert item == ("Hello world", 1)


def test_interrupt_drains_queue_and_sets_flag():
    """interrupt() should empty the queue so stale chunks are never spoken."""
    with patch("voice_app.tts.pyttsx3.init", return_value=make_engine_mock()):
        from voice_app.tts import TTSEngine
        tts = TTSEngine(rate=175, volume=0.9)
        tts.set_generation(1)

        # Queue up several chunks
        tts.speak("sentence one", gen_id=1)
        tts.speak("sentence two", gen_id=1)
        tts.speak("sentence three", gen_id=1)

        tts.interrupt()

        assert tts._tts_queue.empty()
        assert tts._interrupt_flag.is_set()


def test_stale_chunks_are_discarded_by_run_loop():
    """
    Chunks with an old gen_id must not be spoken, even if they're
    already in the queue when the generation changes.
    """
    mock_engine = make_engine_mock()

    with patch("voice_app.tts.pyttsx3.init", return_value=mock_engine):
        from voice_app.tts import TTSEngine
        tts = TTSEngine(rate=175, volume=0.9)
        tts.set_generation(2)  # generation 2 is now current

        # Manually put a stale gen_id=1 chunk straight into the queue
        tts._tts_queue.put(("old response", 1))

        tts.start()
        time.sleep(0.15)  # Give the run loop time to process the stale chunk
        tts.stop()

    # engine.say() should NOT have been called with the stale text
    for c in mock_engine.say.call_args_list:
        assert "old response" not in str(c)


def test_set_generation_clears_interrupt_flag():
    """
    set_generation() is called before starting TTS for a new response.
    It should clear any previous interrupt flag so TTS runs unimpeded.
    """
    with patch("voice_app.tts.pyttsx3.init", return_value=make_engine_mock()):
        from voice_app.tts import TTSEngine
        tts = TTSEngine(rate=175, volume=0.9)

        # Simulate: interrupt was triggered
        tts._interrupt_flag.set()

        # New response starts
        tts.set_generation(5)

        assert not tts._interrupt_flag.is_set()
        assert tts._current_gen_id == 5
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_tts.py -v
```

Expected: `FAILED` — `ModuleNotFoundError`

- [ ] **Step 3: Write `voice_app/tts.py`**

```python
# voice_app/tts.py
#
# Wraps pyttsx3 to speak LLM responses aloud, one sentence at a time.
# Supports mid-speech interruption when the user starts talking.
#
# Interrupt mechanism (two-layer approach):
#   Layer 1 — "started-word" callback: pyttsx3 fires this before each word.
#             We check the interrupt flag and call engine.stop() immediately.
#             This gives sub-word granularity interruption on macOS.
#   Layer 2 — Generation ID: each LLM response gets a unique integer ID.
#             If an interrupt arrives, gen_id is bumped. The TTS run loop
#             discards any queued chunks whose gen_id no longer matches.
#
# Important: pyttsx3's engine must be created in the same thread that
#            calls runAndWait(). That's why we init inside _run().

import queue
import threading
import pyttsx3


class TTSEngine:
    """
    Background thread that speaks text chunks via pyttsx3.
    Interruptible via interrupt() or by changing the generation ID.
    """

    def __init__(self, rate: int, volume: float) -> None:
        self.rate = rate
        self.volume = volume

        # Queue of (text: str, gen_id: int) tuples waiting to be spoken
        self._tts_queue: queue.Queue = queue.Queue()

        # Set when an interrupt is requested — checked by the word callback
        self._interrupt_flag = threading.Event()

        # Signals the run loop to exit
        self._stop_event = threading.Event()

        # Only chunks with this gen_id will be spoken; others are discarded
        self._current_gen_id: int = -1

        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Launch the TTS thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="TTSEngine"
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the TTS thread to exit and wait for it."""
        self._stop_event.set()
        self._interrupt_flag.set()  # Unblock any in-progress speech
        if self._thread:
            self._thread.join(timeout=3)

    def set_generation(self, gen_id: int) -> None:
        """
        Called by the orchestrator before a new LLM response begins.
        Clears the interrupt flag so TTS can run freely for this generation.
        """
        self._current_gen_id = gen_id
        self._interrupt_flag.clear()

    def speak(self, text: str, gen_id: int) -> None:
        """Enqueue a text chunk to be spoken if gen_id is still current."""
        self._tts_queue.put((text, gen_id))

    def interrupt(self) -> None:
        """
        Called by the orchestrator when INTERRUPT event fires.
        Sets the flag (checked by the word callback) and drains the queue.
        """
        self._interrupt_flag.set()
        self._drain_queue()

    def _drain_queue(self) -> None:
        """Remove all pending chunks from the queue."""
        while not self._tts_queue.empty():
            try:
                self._tts_queue.get_nowait()
            except queue.Empty:
                break

    def _run(self) -> None:
        """
        Main TTS loop. Must run in its own thread because pyttsx3's
        runAndWait() is a blocking call that drives the speech synthesizer
        event loop on macOS (NSSpeechSynthesizer).
        """
        # Create the engine here — pyttsx3 on macOS requires creation and
        # usage to happen on the same thread.
        engine = pyttsx3.init()
        engine.setProperty("rate", self.rate)
        engine.setProperty("volume", self.volume)

        def on_word_start(name, location, length):
            """
            Callback fired by pyttsx3 before each word is spoken.
            If the interrupt flag is set, we halt speech immediately.
            """
            if self._interrupt_flag.is_set():
                engine.stop()

        engine.connect("started-word", on_word_start)

        while not self._stop_event.is_set():
            try:
                text, gen_id = self._tts_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Discard stale chunks from a previous (interrupted) generation
            if gen_id != self._current_gen_id:
                continue

            # Bail if interrupted before we even start speaking this chunk
            if self._interrupt_flag.is_set():
                continue

            engine.say(text)
            engine.runAndWait()
            # After runAndWait() returns, either speech completed naturally
            # or engine.stop() was called by the word callback. Either way,
            # we loop and pick up the next chunk (or discard it if stale).
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_tts.py -v
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add voice_app/tts.py tests/test_tts.py
git commit -m "feat: interruptible TTS engine with generation ID gating"
```

---

### Task 9: LLM Client

**Files:**
- Create: `voice_app/llm.py`
- Create: `tests/test_llm.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_llm.py
#
# Tests the sentence splitter (pure function, easy) and the generation
# cancellation logic. We mock OpenAI's streaming client so tests run
# without a network connection.

import pytest
from voice_app.llm import split_into_sentences


class TestSplitIntoSentences:
    """split_into_sentences() is a pure function — test it directly."""

    def test_single_sentence_no_split(self):
        result = split_into_sentences("Hello there.")
        assert result == ["Hello there."]

    def test_two_sentences_split_correctly(self):
        result = split_into_sentences("Hello there. How are you?")
        assert result == ["Hello there.", "How are you?"]

    def test_exclamation_and_question(self):
        result = split_into_sentences("That's great! Really? Yes.")
        assert result == ["That's great!", "Really?", "Yes."]

    def test_empty_string_returns_empty_list(self):
        result = split_into_sentences("")
        assert result == []

    def test_no_punctuation_returns_one_item(self):
        # Incomplete sentence (still streaming) — treat as one chunk
        result = split_into_sentences("Hey how are you doing today")
        assert result == ["Hey how are you doing today"]

    def test_strips_whitespace(self):
        result = split_into_sentences("  Hello.  World.  ")
        assert len(result) == 2
        assert result[0].strip() == "Hello."
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_llm.py -v
```

Expected: `FAILED` — `ModuleNotFoundError`

- [ ] **Step 3: Write `voice_app/llm.py`**

```python
# voice_app/llm.py
#
# Streams GPT-4o responses and chunks them into sentences for TTS.
#
# Why sentence chunking?
#   TTS must speak complete phrases to sound natural. If we fed raw
#   streaming tokens one-by-one, pyttsx3 would say "Hello" then pause,
#   then "there" then pause. Instead we accumulate tokens until we detect
#   a sentence boundary (. ! ?) then push the whole sentence to TTS.
#
# Why generation IDs?
#   When the user interrupts, the LLM may still be streaming tokens.
#   We bump the gen_id to invalidate the current stream. The LLM thread
#   checks gen_id on every chunk and aborts the stream immediately.
#   This prevents stale sentences from reaching TTS after an interrupt.
#
# Data flow:
#   Orchestrator.submit(messages, gen_id)
#     → _input_queue
#       → _stream_response() → TTS sentences + events on event_queue

import queue
import re
import threading
import logging
from typing import List, Dict

from openai import OpenAI

logger = logging.getLogger(__name__)


def split_into_sentences(text: str) -> List[str]:
    """
    Split a string into complete sentences on . ! ? boundaries.

    Returns a list of non-empty strings. Partial sentences (no terminal
    punctuation) are returned as a single element — the caller keeps them
    in a buffer and prepends them to the next batch of tokens.
    """
    text = text.strip()
    if not text:
        return []
    # Split after . ! ? followed by whitespace
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p for p in parts if p.strip()]


class LLMClient:
    """
    Background thread that sends conversation history to GPT-4o (streaming)
    and splits the response into sentences for the TTS engine.
    """

    def __init__(
        self,
        api_key: str,
        config,                      # LLMConfig dataclass
        event_queue: queue.Queue,    # Posts FIRST_TTS_CHUNK / TTS_COMPLETE / ERROR
        tts_engine,                  # TTSEngine instance to call speak() on
    ) -> None:
        self.config = config
        self.event_queue = event_queue
        self.tts_engine = tts_engine

        self._client = OpenAI(api_key=api_key)
        self._input_queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._current_gen_id: int = 0
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Launch the LLM thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="LLMClient"
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the LLM thread to exit."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def submit(self, messages: List[Dict], gen_id: int) -> None:
        """
        Submit a conversation for LLM processing.
        Called by the orchestrator when UTTERANCE_COMPLETE fires.
        """
        self._current_gen_id = gen_id
        self._input_queue.put((messages, gen_id))

    def cancel(self) -> None:
        """
        Invalidate the current generation so the streaming loop aborts.
        Called by the orchestrator on INTERRUPT. The running stream will
        notice gen_id mismatch on the next token and return early.
        """
        self._current_gen_id += 1
        # Drain any pending submissions
        while not self._input_queue.empty():
            try:
                self._input_queue.get_nowait()
            except queue.Empty:
                break

    def _run(self) -> None:
        """Main loop: wait for submissions and process them one at a time."""
        while not self._stop_event.is_set():
            try:
                messages, gen_id = self._input_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            self._stream_response(messages, gen_id)

    def _stream_response(self, messages: List[Dict], gen_id: int) -> None:
        """
        Stream a response from GPT-4o and chunk it into sentences.

        Token accumulation strategy:
            buffer += each streamed token
            When we have ≥2 sentences, push all but the last to TTS.
            (The last may be partial — keep it in the buffer.)
            After the stream ends, flush whatever remains in the buffer.
        """
        buffer = ""
        full_response = ""   # Accumulates the entire assistant reply for history
        first_chunk_sent = False

        try:
            stream = self._client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream=True,
            )

            for chunk in stream:
                # Abort immediately if this generation was superseded
                if gen_id != self._current_gen_id:
                    logger.debug("LLM gen_id %d stale, aborting stream", gen_id)
                    return

                delta = chunk.choices[0].delta.content
                if delta is None:
                    continue

                buffer += delta
                sentences = split_into_sentences(buffer)

                # If we have multiple sentences, the all-but-last are complete
                if len(sentences) > 1:
                    for sentence in sentences[:-1]:
                        if gen_id != self._current_gen_id:
                            return

                        # First sentence triggers SPEAKING state
                        if not first_chunk_sent:
                            self.event_queue.put({"type": "FIRST_TTS_CHUNK"})
                            first_chunk_sent = True

                        self.tts_engine.speak(sentence, gen_id)
                        full_response += sentence + " "

                    # Keep the (potentially incomplete) last sentence
                    buffer = sentences[-1]

            # ── Flush remaining buffer after stream ends ───────────────────
            if buffer.strip() and gen_id == self._current_gen_id:
                if not first_chunk_sent:
                    self.event_queue.put({"type": "FIRST_TTS_CHUNK"})
                self.tts_engine.speak(buffer.strip(), gen_id)
                full_response += buffer.strip()

            if gen_id == self._current_gen_id:
                # Include full response text so the orchestrator can persist it
                self.event_queue.put({
                    "type": "TTS_COMPLETE",
                    "gen_id": gen_id,
                    "response": full_response.strip(),
                })

        except Exception as e:
            logger.error("LLM error: %s", e)
            self.event_queue.put({"type": "ERROR", "error": str(e)})
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_llm.py -v
```

Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add voice_app/llm.py tests/test_llm.py
git commit -m "feat: LLM streaming client with sentence chunking and gen-ID abort"
```

---

## Chunk 4: Integration

### Task 10: Orchestrator

**Files:**
- Create: `voice_app/orchestrator.py`

- [ ] **Step 1: Write `voice_app/orchestrator.py`**

```python
# voice_app/orchestrator.py
#
# The Orchestrator is the central coordinator. It owns:
#   - The state machine (what state are we in?)
#   - The event queue (all threads post here)
#   - All component instances (audio, VAD, STT, LLM, TTS)
#
# It runs two things:
#   1. A VAD loop in a background thread (posts SPEECH_DETECTED / INTERRUPT events)
#   2. The main event loop on the main thread (consumes events, drives state machine)
#
# The orchestrator is the ONLY place state transitions happen. Components
# never call each other directly — they post events and let the orchestrator
# decide what to do next. This makes the system easy to reason about.

import queue
import threading
import logging

from voice_app.state_machine import StateMachine, State, AppEvent
from voice_app.config import AppConfig
from voice_app.history import load_history, append_message
from voice_app.audio_capture import AudioCapture
from voice_app.vad import VADDetector
from voice_app.stt import STTClient
from voice_app.llm import LLMClient
from voice_app.tts import TTSEngine

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Wires all components together and drives the conversation.

    Usage:
        orchestrator = Orchestrator(config, debug=False)
        orchestrator.run()   # Blocks until Ctrl-C
    """

    def __init__(self, config: AppConfig, debug: bool = False) -> None:
        self.config = config
        self.debug = debug

        # Central event queue — all threads post here, main thread reads
        self.event_queue: queue.Queue = queue.Queue()

        # State machine — only the main thread calls transition()
        self.state_machine = StateMachine()

        # Generation ID — incremented on every new LLM request or interrupt
        # Used to discard stale TTS chunks from previous generations
        self._gen_id: int = 0

        # Shutdown signal for the VAD loop thread
        self._shutdown_event = threading.Event()

        # ── Audio queues (one per consumer, prevents head-of-line blocking) ─
        self._vad_audio_queue: queue.Queue = queue.Queue(maxsize=200)
        self._stt_audio_queue: queue.Queue = queue.Queue(maxsize=200)

        # ── Components ────────────────────────────────────────────────────

        self.audio_capture = AudioCapture(
            sample_rate=config.audio.sample_rate,
            channels=config.audio.channels,
        )
        self.audio_capture.add_consumer(self._vad_audio_queue)
        self.audio_capture.add_consumer(self._stt_audio_queue)

        self.vad = VADDetector(
            sample_rate=config.audio.sample_rate,
            aggressiveness=config.vad.aggressiveness,
            ring_buffer_size=config.vad.ring_buffer_size,
            speech_start_frames=config.vad.speech_start_frames,
        )

        self.tts = TTSEngine(
            rate=config.tts.rate,
            volume=config.tts.volume,
        )

        self.llm = LLMClient(
            api_key=config.openai_api_key,
            config=config.llm,
            event_queue=self.event_queue,
            tts_engine=self.tts,
        )

        self.stt = STTClient(
            api_key=config.deepgram_api_key,
            config=config.deepgram,
            audio_queue=self._stt_audio_queue,
            event_queue=self.event_queue,
        )

        # Load persisted conversation history
        self._messages = load_history(config.history.file)

    # ── Public API ─────────────────────────────────────────────────────────

    def run(self) -> None:
        """
        Start all threads and run the event loop.
        Blocks until Ctrl-C or SHUTDOWN event.
        """
        self.tts.start()
        self.llm.start()
        self.audio_capture.start()
        self.stt.start()

        # VAD runs in its own thread, posts SPEECH_DETECTED / INTERRUPT events
        self._vad_thread = threading.Thread(
            target=self._vad_loop, daemon=True, name="VADLoop"
        )
        self._vad_thread.start()

        self._display_status("Ready — start talking! (Ctrl-C to quit)")

        try:
            self._event_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self._shutdown()

    # ── VAD loop (background thread) ───────────────────────────────────────

    def _vad_loop(self) -> None:
        """
        Reads mic frames from the VAD queue and fires speech events.

        - In IDLE state: SPEECH_DETECTED when user starts speaking.
        - In SPEAKING or PROCESSING state: INTERRUPT when user starts speaking.
          This is the full-duplex interrupt mechanism.
        """
        while not self._shutdown_event.is_set():
            try:
                frame = self._vad_audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            speech_detected = self.vad.process_frame(frame)
            current = self.state_machine.state

            if current == State.IDLE and speech_detected:
                self.event_queue.put({"type": "SPEECH_DETECTED"})

            elif current in (State.SPEAKING, State.PROCESSING) and speech_detected:
                self.event_queue.put({"type": "INTERRUPT"})

    # ── Main event loop ────────────────────────────────────────────────────

    def _event_loop(self) -> None:
        """
        Consumes events from the event queue and drives the state machine.
        Only this method (running on the main thread) calls sm.transition().
        """
        while True:
            try:
                event = self.event_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            event_type = event.get("type")
            logger.debug("Event: %s | State: %s", event_type, self.state_machine.state)

            if event_type == "SPEECH_DETECTED":
                self.state_machine.transition(AppEvent.SPEECH_DETECTED)
                self._display_status("Listening...")

            elif event_type == "UTTERANCE_COMPLETE":
                text = event["text"]
                self.state_machine.transition(AppEvent.UTTERANCE_COMPLETE)
                self._display_user(text)
                self._handle_utterance(text)

            elif event_type == "FIRST_TTS_CHUNK":
                self.state_machine.transition(AppEvent.FIRST_TTS_CHUNK)
                self._display_status("Speaking...")

            elif event_type == "TTS_COMPLETE":
                # Ignore stale TTS_COMPLETE events from old generations
                if event.get("gen_id") == self._gen_id:
                    self.state_machine.transition(AppEvent.TTS_COMPLETE)
                    # Persist the assistant's full response to conversation history
                    response_text = event.get("response", "")
                    if response_text:
                        self._messages = append_message(
                            self.config.history.file,
                            "assistant",
                            response_text,
                            self.config.history.max_messages_in_context,
                        )
                    self._display_status("Ready")

            elif event_type == "INTERRUPT":
                state = self.state_machine.state
                if state in (State.SPEAKING, State.PROCESSING):
                    # Bump gen_id to invalidate LLM stream and TTS chunks
                    self._gen_id += 1
                    self.tts.interrupt()
                    self.llm.cancel()
                    self.vad.reset()  # Clear ring buffer to avoid re-triggering
                    self.state_machine.transition(AppEvent.INTERRUPT)
                    self._display_status("Interrupted — listening...")

            elif event_type == "TRANSCRIPT_INTERIM":
                if self.debug:
                    print(f"  [{event['text']}]", end="\r", flush=True)

            elif event_type == "ERROR":
                logger.error("Error event: %s", event.get("error"))
                print(f"  Error: {event.get('error')}")
                self.state_machine.transition(AppEvent.ERROR)
                self._display_status("Ready (after error)")

            elif event_type == "SHUTDOWN":
                break

    # ── Utterance handling ─────────────────────────────────────────────────

    def _handle_utterance(self, text: str) -> None:
        """
        Persist the user's message, build the full message list,
        bump the gen_id, and submit to the LLM thread.
        """
        # Persist user message and get updated history
        self._messages = append_message(
            self.config.history.file,
            "user",
            text,
            self.config.history.max_messages_in_context,
        )

        # New generation starts here
        self._gen_id += 1
        self.tts.set_generation(self._gen_id)

        # Build OpenAI messages: [system] + conversation history
        messages = [{"role": "system", "content": self.config.system_prompt}]
        messages.extend(self._messages[-self.config.history.max_messages_in_context:])

        self.llm.submit(messages, self._gen_id)

    # ── Display helpers ────────────────────────────────────────────────────

    def _display_status(self, status: str) -> None:
        """Print a status line (always shown — gives the user live feedback)."""
        print(f"  [{status}]")

    def _display_user(self, text: str) -> None:
        """Print the user's transcribed speech."""
        print(f"\nYou: {text}")

    # ── Shutdown ───────────────────────────────────────────────────────────

    def _shutdown(self) -> None:
        """Gracefully stop all threads in reverse start order."""
        self._shutdown_event.set()
        self.stt.stop()
        self.audio_capture.stop()
        self.llm.stop()
        self.tts.stop()
        print("\nGoodbye!")
```

- [ ] **Step 2: Write `tests/test_orchestrator.py`**

```python
# tests/test_orchestrator.py
#
# Tests the orchestrator's event dispatch logic using mocked components.
# We inject events directly into the event queue and assert that the
# state machine and components respond correctly — no real audio or
# network I/O needed.

import queue
import threading
import pytest
from unittest.mock import MagicMock, patch, call
from voice_app.state_machine import State


def make_mock_config(tmp_path):
    """Build a minimal AppConfig-like mock pointing history to a temp file."""
    from voice_app.config import (
        AppConfig, AudioConfig, VADConfig, DeepgramConfig,
        LLMConfig, TTSConfig, HistoryConfig
    )
    return AppConfig(
        system_prompt="You are a test assistant.",
        audio=AudioConfig(),
        vad=VADConfig(),
        deepgram=DeepgramConfig(),
        llm=LLMConfig(),
        tts=TTSConfig(),
        history=HistoryConfig(file=str(tmp_path / "history.json")),
        openai_api_key="sk-test",
        deepgram_api_key="dg-test",
    )


def make_orchestrator(tmp_path):
    """
    Build an Orchestrator with all I/O components mocked out.
    Returns (orchestrator, mocked_tts, mocked_llm).
    """
    from voice_app.orchestrator import Orchestrator

    cfg = make_mock_config(tmp_path)

    with patch("voice_app.orchestrator.AudioCapture") as MockAudio, \
         patch("voice_app.orchestrator.VADDetector") as MockVAD, \
         patch("voice_app.orchestrator.STTClient") as MockSTT, \
         patch("voice_app.orchestrator.LLMClient") as MockLLM, \
         patch("voice_app.orchestrator.TTSEngine") as MockTTS:

        orch = Orchestrator(cfg, debug=False)
        mock_tts = orch.tts
        mock_llm = orch.llm
        mock_vad = orch.vad

    return orch, mock_tts, mock_llm, mock_vad


def pump_event(orch, event: dict, then_shutdown: bool = True):
    """
    Put an event onto the orchestrator's queue, optionally follow with SHUTDOWN,
    then run _event_loop() in a thread so it processes and exits.
    """
    orch.event_queue.put(event)
    if then_shutdown:
        orch.event_queue.put({"type": "SHUTDOWN"})

    t = threading.Thread(target=orch._event_loop)
    t.start()
    t.join(timeout=2)
    assert not t.is_alive(), "Event loop did not exit in time"


class TestStateMachineIntegration:
    """Events correctly advance the state machine."""

    def test_speech_detected_moves_to_listening(self, tmp_path):
        orch, _, _, _ = make_orchestrator(tmp_path)
        assert orch.state_machine.state == State.IDLE
        pump_event(orch, {"type": "SPEECH_DETECTED"})
        assert orch.state_machine.state == State.LISTENING

    def test_utterance_complete_moves_to_processing(self, tmp_path):
        orch, mock_tts, mock_llm, _ = make_orchestrator(tmp_path)
        orch.state_machine.state = State.LISTENING
        pump_event(orch, {"type": "UTTERANCE_COMPLETE", "text": "Hello"})
        assert orch.state_machine.state == State.PROCESSING
        mock_llm.submit.assert_called_once()

    def test_tts_complete_moves_to_idle(self, tmp_path):
        orch, _, _, _ = make_orchestrator(tmp_path)
        orch.state_machine.state = State.SPEAKING
        orch._gen_id = 3
        pump_event(orch, {"type": "TTS_COMPLETE", "gen_id": 3, "response": "Hi!"})
        assert orch.state_machine.state == State.IDLE

    def test_stale_tts_complete_ignored(self, tmp_path):
        orch, _, _, _ = make_orchestrator(tmp_path)
        orch.state_machine.state = State.SPEAKING
        orch._gen_id = 5
        # gen_id=3 is stale — should be ignored
        pump_event(orch, {"type": "TTS_COMPLETE", "gen_id": 3, "response": ""})
        assert orch.state_machine.state == State.SPEAKING


class TestInterruptPath:
    """The interrupt mechanism must cancel TTS and LLM and reset to LISTENING."""

    def test_interrupt_during_speaking_calls_tts_and_llm(self, tmp_path):
        orch, mock_tts, mock_llm, mock_vad = make_orchestrator(tmp_path)
        orch.state_machine.state = State.SPEAKING
        pump_event(orch, {"type": "INTERRUPT"})

        mock_tts.interrupt.assert_called_once()
        mock_llm.cancel.assert_called_once()
        mock_vad.reset.assert_called_once()
        assert orch.state_machine.state == State.LISTENING

    def test_interrupt_during_processing_also_works(self, tmp_path):
        orch, mock_tts, mock_llm, mock_vad = make_orchestrator(tmp_path)
        orch.state_machine.state = State.PROCESSING
        pump_event(orch, {"type": "INTERRUPT"})

        mock_tts.interrupt.assert_called_once()
        assert orch.state_machine.state == State.LISTENING

    def test_interrupt_bumps_gen_id(self, tmp_path):
        orch, _, _, _ = make_orchestrator(tmp_path)
        orch.state_machine.state = State.SPEAKING
        orch._gen_id = 7
        pump_event(orch, {"type": "INTERRUPT"})
        assert orch._gen_id == 8

    def test_interrupt_from_idle_is_ignored(self, tmp_path):
        """INTERRUPT while IDLE should not call tts.interrupt() or change state."""
        orch, mock_tts, mock_llm, _ = make_orchestrator(tmp_path)
        assert orch.state_machine.state == State.IDLE
        pump_event(orch, {"type": "INTERRUPT"})
        mock_tts.interrupt.assert_not_called()
        assert orch.state_machine.state == State.IDLE


class TestHistoryPersistence:
    """Assistant responses must be saved to history when TTS completes."""

    def test_assistant_response_persisted_on_tts_complete(self, tmp_path):
        from voice_app.history import load_history
        orch, _, _, _ = make_orchestrator(tmp_path)
        orch.state_machine.state = State.SPEAKING
        orch._gen_id = 1
        pump_event(orch, {
            "type": "TTS_COMPLETE",
            "gen_id": 1,
            "response": "Hey! I'm doing great.",
        })
        messages = load_history(str(tmp_path / "history.json"))
        assert any(m["role"] == "assistant" and "great" in m["content"]
                   for m in messages)
```

- [ ] **Step 3: Run test to verify it passes**

```bash
pytest tests/test_orchestrator.py -v
```

Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add voice_app/orchestrator.py tests/test_orchestrator.py
git commit -m "feat: orchestrator — event loop, VAD loop, interrupt handling"
```

---

### Task 11: Entry Point

**Files:**
- Create: `main.py`

- [ ] **Step 1: Write `main.py`**

```python
# main.py
#
# Entry point for the Voice LLM app.
#
# Usage:
#   python main.py              — normal start, loads existing conversation
#   python main.py --new        — clears history, starts a fresh conversation
#   python main.py --debug      — verbose thread-level logging to stdout
#   python main.py --config X   — use a different config file (for testing)

import argparse
import logging

from voice_app.config import load_config
from voice_app.history import save_history
from voice_app.orchestrator import Orchestrator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Voice LLM — Full Duplex with VAD. Talk to GPT-4o like a phone call."
    )
    parser.add_argument(
        "--new",
        action="store_true",
        help="Clear conversation history and start a fresh session.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose logging (thread events, interim transcripts).",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config YAML file (default: config.yaml).",
    )
    args = parser.parse_args()

    # Configure logging level
    log_level = logging.DEBUG if args.debug else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # Load config (raises if .env is missing API keys)
    config = load_config(args.config)

    if args.new:
        save_history(config.history.file, [])
        print("Started a new conversation.")

    orchestrator = Orchestrator(config, debug=args.debug)
    orchestrator.run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run all tests one final time**

```bash
pytest tests/ -v
```

Expected: All tests pass.

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "feat: main.py entry point with --new and --debug flags"
```

---

## Chunk 5: Smoke Tests (Manual)

> These require a microphone, speakers/headphones, and valid API keys in `.env`.

### Task 12: Manual Verification

- [ ] **Smoke test 1: VAD check**

```bash
python -c "
import queue, time
from voice_app.config import load_config
from voice_app.audio_capture import AudioCapture
from voice_app.vad import VADDetector

cfg = load_config()
q = queue.Queue()
cap = AudioCapture(cfg.audio.sample_rate, cfg.audio.channels)
cap.add_consumer(q)
vad = VADDetector(cfg.audio.sample_rate, cfg.vad.aggressiveness,
                  cfg.vad.ring_buffer_size, cfg.vad.speech_start_frames)

cap.start()
print('Speak into the mic — should see SPEECH / silence alternating')
for _ in range(200):
    frame = q.get()
    result = vad.process_frame(frame)
    print('SPEECH' if result else '.', end='', flush=True)
cap.stop()
"
```

Expected: Prints dots during silence, `SPEECH` when you talk.

- [ ] **Smoke test 2: STT check**

```bash
python -c "
import queue, time
from voice_app.config import load_config
from voice_app.audio_capture import AudioCapture
from voice_app.stt import STTClient

cfg = load_config()
audio_q = queue.Queue()
event_q = queue.Queue()

cap = AudioCapture(cfg.audio.sample_rate, cfg.audio.channels)
cap.add_consumer(audio_q)
stt = STTClient(cfg.deepgram_api_key, cfg.deepgram, audio_q, event_q)

cap.start()
stt.start()
print('Speak a sentence — should see transcript appear')
end = time.time() + 15
while time.time() < end:
    try:
        e = event_q.get(timeout=0.5)
        print(e)
    except:
        pass
stt.stop()
cap.stop()
"
```

Expected: Your spoken words appear as `UTTERANCE_COMPLETE` events within 1-2 seconds.

- [ ] **Smoke test 3: Full conversation loop**

```bash
# Only copy the template if .env doesn't exist yet (avoids overwriting real keys)
[ -f .env ] || cp .env.example .env
# Edit .env with real API keys, then:
python main.py --debug
```

Speak a sentence. Expected flow:
1. Terminal shows `[Listening...]`
2. Transcript appears: `You: <your words>`
3. Terminal shows `[Speaking...]`
4. AI responds aloud
5. Terminal shows `[Ready]`

- [ ] **Smoke test 4: Interrupt test**

While the AI is speaking, say something. Expected:
1. Terminal shows `[Interrupted — listening...]`
2. AI stops mid-sentence
3. Your new utterance is transcribed and a new response begins

- [ ] **Smoke test 5: History persistence**

```bash
python main.py --debug
# Say: "My name is Alex"
# Ctrl-C, then restart:
python main.py --debug
# Say: "What's my name?"
```

Expected: AI knows your name from the previous session.

---
