# Voice LLM — Full Duplex with VAD

> A CLI Python app for continuous voice conversation with GPT-4o.
> Speak naturally, get spoken responses, and interrupt the AI mid-sentence — like a phone call.

---

## Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| Language | Python | User preference |
| LLM | OpenAI GPT-4o (streaming) | User preference |
| STT | Deepgram live WebSocket (nova-2) | Free tier ($200 credits), fast, built-in endpointing |
| TTS | pyttsx3 | Free, offline, uses macOS NSSpeechSynthesizer |
| VAD | webrtcvad (local) | Fast interrupt detection (<160ms latency) |
| Audio | PyAudio | Microphone capture (requires `brew install portaudio`) |

---

## Project Structure

```
minnetonka/
├── main.py                      # Entry point — args, bootstrap, run
├── requirements.txt             # All Python dependencies
├── config.yaml                  # Tunable settings (VAD thresholds, TTS rate, etc.)
├── .env.example                 # Template for API keys
├── docs/
│   └── voice-llm-fullduplex-design.md   # This design doc
├── voice_app/
│   ├── __init__.py
│   ├── config.py                # Loads config.yaml + .env into typed dataclass
│   ├── state_machine.py         # ConversationState enum + transition logic
│   ├── audio_capture.py         # Mic capture thread (PyAudio, 20ms frames)
│   ├── vad.py                   # webrtcvad wrapper — interrupt detection
│   ├── stt.py                   # Deepgram WebSocket streaming client
│   ├── llm.py                   # OpenAI streaming + sentence chunking
│   ├── tts.py                   # pyttsx3 wrapper with interrupt support
│   ├── orchestrator.py          # Central coordinator — owns state, wires threads
│   └── history.py               # JSON conversation persistence
└── conversation_history.json    # Auto-created at runtime
```

---

## Threading Architecture (5 threads + main)

```
Main Thread (Orchestrator event loop)
│
├── AudioCaptureThread — reads mic via PyAudio, pushes 20ms PCM frames to two queues
│
├── VADThread — runs webrtcvad on frames, detects speech onset for interrupts
│
├── STTThread — sends frames to Deepgram WebSocket, receives transcriptions
│
├── LLMThread — sends user text to GPT-4o streaming, chunks response into sentences
│
└── TTSThread — speaks sentence chunks via pyttsx3, supports mid-word interrupt
```

### Inter-thread Communication
- **`queue.Queue`** — thread-safe data flow (audio frames, text chunks)
- **`threading.Event`** — signal flags (interrupt, shutdown)
- **Single `event_queue`** — all threads post events here, orchestrator consumes them on the main thread

### Why Threads (not asyncio)?
PyAudio's `stream.read()` and pyttsx3's `runAndWait()` are blocking C-library calls. Wrapping them in asyncio executors adds complexity without benefit. Threads with queues are the natural fit for I/O-bound work.

---

## State Machine

### States
```
IDLE        — App running, nobody talking. Mic captured, VAD watching.
LISTENING   — User is speaking. Deepgram receiving audio, producing transcripts.
PROCESSING  — User finished speaking. LLM generating response.
SPEAKING    — TTS playing AI response aloud.
```

### Transitions
```
IDLE → LISTENING → PROCESSING → SPEAKING → IDLE
                                    ↓
                        (INTERRUPT) → LISTENING
```

| Current State | Event | Next State | Action |
|---------------|-------|------------|--------|
| IDLE | SPEECH_DETECTED | LISTENING | Update terminal display |
| LISTENING | UTTERANCE_COMPLETE | PROCESSING | Send finalized text to LLM thread |
| PROCESSING | FIRST_TTS_CHUNK | SPEAKING | TTS starts playing first sentence |
| SPEAKING | TTS_COMPLETE | IDLE | All chunks spoken, return to idle |
| SPEAKING | INTERRUPT | LISTENING | Stop TTS, cancel LLM, drain queues |
| PROCESSING | INTERRUPT | LISTENING | Cancel LLM generation, drain queues |
| ANY | ERROR | IDLE | Log error, reset all pipelines |
| ANY | SHUTDOWN | (exit) | Graceful shutdown of all threads |

---

## Data Flow (End to End)

```
MICROPHONE
    │
    │ (PyAudio, 20ms PCM frames, 640 bytes each)
    │
    ├────────────────────────────────┐
    │                                │
    ▼                                ▼
 VAD QUEUE                      STT QUEUE
    │                                │
    ▼                                ▼
 VADThread                      STTThread
 (webrtcvad)                    (Deepgram WebSocket)
    │                                │
    │ SPEECH_DETECTED                │ interim results → live terminal display
    │ INTERRUPT                      │ UTTERANCE_COMPLETE + final text
    │                                │
    └──────────┐    ┌────────────────┘
               │    │
               ▼    ▼
           EVENT QUEUE
               │
               ▼
          ORCHESTRATOR (main thread)
               │
               │ sends finalized user text
               ▼
          LLM INPUT QUEUE
               │
               ▼
           LLMThread (OpenAI GPT-4o streaming)
               │
               │ sentence-sized chunks with generation_id
               ▼
           TTS QUEUE
               │
               ▼
           TTSThread (pyttsx3)
               │
               ▼
           SPEAKER OUTPUT
```

---

## Key Design Decisions

### 1. Interrupt Mechanism
The core of the full-duplex experience:

- **VAD detects speech onset** during SPEAKING state using a webrtcvad ring buffer. When 6 out of 8 frames contain speech → fire INTERRUPT event.
- **Generation ID pattern**: Each LLM generation gets a unique integer ID. On interrupt, the orchestrator invalidates the current ID. The LLM thread checks this on every streamed chunk and aborts if stale. The TTS thread also discards chunks with stale IDs.
- **pyttsx3 interrupt**: Uses the `started-word` callback to check an interrupt flag before each word. Calls `engine.stop()` to halt speech immediately. **Fallback**: subprocess isolation if the callback approach is unreliable.

### 2. Two VAD Systems (Local + Server)
- **webrtcvad (local)**: Zero network latency. Detects speech *start* within ~160ms. Critical for responsive interrupts.
- **Deepgram endpointing (server)**: Uses language model context to accurately detect when an utterance is truly *finished*. `endpointing=300ms` for fast partial finals, `utterance_end_ms=1000ms` for the "user is done" signal.

### 3. LLM Sentence Chunking
GPT-4o streams tokens → accumulated into sentence-sized chunks → pushed to TTS queue. The first sentence starts playing while the rest is still generating. This balances low latency with natural-sounding speech.

### 4. Audio Feedback Prevention
Headphones recommended. If using speakers, an optional mic-mute flag during TTS playback can be enabled (degrades to half-duplex).

---

## Configuration

### .env (API keys — never committed to git)
```
OPENAI_API_KEY=sk-...
DEEPGRAM_API_KEY=...
```

### config.yaml (all tunable parameters)
```yaml
# Personality
system_prompt: |
  You are a friendly, casual companion. You speak naturally like a close
  friend would. Keep responses concise (2-3 sentences typically) since
  this is a voice conversation. Be warm, use casual language, and don't
  be overly formal.

# Audio
audio:
  sample_rate: 16000
  frame_duration_ms: 20
  channels: 1

# VAD
vad:
  aggressiveness: 2           # 0-3, higher = more aggressive filtering
  speech_start_frames: 6      # How many frames in ring buffer must be speech
  ring_buffer_size: 8         # Rolling window size

# Deepgram
deepgram:
  model: "nova-2"
  language: "en"
  endpointing: 300            # ms of silence to trigger is_final
  utterance_end_ms: 1000      # ms of silence to trigger utterance_end
  smart_format: true

# LLM
llm:
  model: "gpt-4o"
  temperature: 0.8
  max_tokens: 500

# TTS
tts:
  rate: 175                   # Words per minute
  volume: 0.9

# History
history:
  file: "conversation_history.json"
  max_messages_in_context: 50
```

---

## Dependencies (requirements.txt)

```
openai>=1.30.0            # GPT-4o streaming chat completions
deepgram-sdk>=3.5.0       # Live WebSocket STT
pyaudio>=0.2.14           # Microphone capture (brew install portaudio first)
webrtcvad>=2.0.10         # Local VAD for interrupt detection
pyttsx3>=2.90             # Offline macOS TTS
pyyaml>=6.0               # Config file parsing
python-dotenv>=1.0.0      # .env file for API keys
```

### macOS Prerequisites
```bash
brew install portaudio
```

---

## Implementation Order

### Phase 1: Foundation
1. `config.py` + `config.yaml` + `.env.example` — config loading
2. `state_machine.py` — state enum + transitions
3. `history.py` — JSON persistence

### Phase 2: Audio Pipeline
4. `audio_capture.py` — mic capture thread
5. `vad.py` — VAD with ring buffer
6. `stt.py` — Deepgram WebSocket streaming

### Phase 3: Output Pipeline
7. `tts.py` — pyttsx3 with interrupt support
8. `llm.py` — OpenAI streaming + sentence chunking

### Phase 4: Integration
9. `orchestrator.py` — wire everything, event loop, interrupt handling
10. `main.py` — entry point with `--new` and `--debug` flags

### Phase 5: Polish
11. Terminal display (status indicators, live transcript)
12. Error handling + reconnection (Deepgram, OpenAI)

---

## Known Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|-----------|
| pyttsx3 `stop()` unreliable from another thread | High | `started-word` callback approach; fallback to subprocess isolation |
| Speaker audio feedback into mic | Medium | Recommend headphones; optional mic-mute during TTS |
| webrtcvad false positive interrupts | Medium | Tune aggressiveness=2, require 6/8 frames, min 200ms duration |
| Deepgram WebSocket drops | Low | Exponential backoff reconnect, buffer last 2s of audio |
| Python GIL bottleneck | Low | All threads are I/O-bound; GIL released during I/O ops |

---

## Verification Plan

1. **Unit test** state machine transitions
2. **Mic test**: Run audio capture + VAD, verify "speech"/"silence" prints correctly
3. **STT test**: Speak into mic, verify Deepgram prints live transcription
4. **TTS test**: Queue canned text, verify speech + interrupt via keyboard
5. **Integration test**: Full conversation loop — speak, get response, interrupt mid-sentence
6. **History test**: Restart app, verify previous conversation loads

---

## Usage

```bash
python main.py              # Normal start, loads existing conversation
python main.py --new        # Start fresh conversation
python main.py --debug      # Verbose logging (shows thread activity)
```
