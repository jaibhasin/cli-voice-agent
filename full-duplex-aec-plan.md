# Plan: True Full-Duplex via Software AEC (speexdsp + PyAudio TTS)

## Context

The current implementation uses **half-duplex echo suppression** as a workaround: when TTS plays, the mic is silenced to Deepgram (silence bytes injected) so the agent can't hear itself. This works, but means the agent cannot listen while speaking. The `barge_in_while_speaking` flag exists but only works safely with headphones.

The goal is true full-duplex — agent speaks through speakers **and** listens at the same time, without feedback — using software Acoustic Echo Cancellation (AEC). The AEC subtracts the known speaker output from the mic signal before it reaches VAD/STT.

**Industry approach**: WebRTC browsers get this for free via built-in AEC. For Python CLI apps, the standard solution is `speexdsp` (SpeexDSP library) — the same algorithm used in PJSIP, Rhasspy, and many open-source voice agents. The key requirement: AEC needs the raw PCM being played through the speaker as a **reference signal**, which means TTS must produce and route PCM through PyAudio (not an opaque subprocess like `say`).

---

## Approach: PyAudio TTS Output + speexdsp AEC

### High-Level Data Flow (after change)

```
Neural TTS PCM ──► PyAudio output callback ──► Speakers
                           │
                           ▼
                  SpeakerReferenceBuffer
                    (timestamped ring buffer)
                           │
Microphone ──► AudioCapture │
                    │       ▼
                    └──► AECProcessor.process(mic_frame, ref_frame)
                                  │
                                  ▼  cleaned frames
                          VAD queue + STT queue
```

When `aec.enabled`, echo suppression in `orchestrator.py` and `stt.py` is **removed**; when AEC is off, existing half-duplex suppression stays as today.

---

## Files to Modify / Create

| File | Change |
|------|--------|
| `voice_app/aec.py` | **NEW** — `SpeakerReferenceBuffer` + `AECProcessor` |
| `voice_app/tts.py` | Replace `say` subprocess + pyttsx3 with PyAudio output stream + Deepgram TTS PCM |
| `voice_app/audio_capture.py` | Inject optional `aec_processor` — apply AEC to frames before queuing |
| `voice_app/config.py` | Add `AECConfig` dataclass; extend `AppConfig` and `TTSConfig` |
| `voice_app/orchestrator.py` | When `aec.enabled`, remove echo-suppression timers and wire `AECProcessor` → `AudioCapture`; when disabled, keep current behavior |
| `voice_app/stt.py` | When `aec.enabled`, drop echo suppression; when disabled, keep `set_echo_suppression` / silence path |
| `config.yaml` | Add `aec:` section; add TTS `service`/`voice` fields |
| `requirements.txt` | Add `speexdsp` (PyPI package builds against system **libspeexdsp**; needs compiler + often `brew install speexdsp` on macOS / `libspeexdsp-dev` on Debian) |
| `environment.yml` | Add `speexdsp` (conda-forge) for the native lib; Python bindings may still be installed via pip |
| `tests/test_aec.py` | **NEW** — unit tests for buffer and AEC processor |
| `tests/test_tts.py` | Update for new PyAudio-based TTS |
| `tests/test_orchestrator.py` | With `aec.enabled` in test config: assert echo suppression helpers are unused; with AEC off, existing half-duplex behavior can stay covered |
| `tests/conftest.py` | Replace pyttsx3 autouse fixture with PyAudio mock fixture |
| `tests/test_config.py` | Add `AECConfig` parsing assertions |

---

## Implementation Steps

### Step 1 — Config additions (no behavior change)
- Add `AECConfig` dataclass to `voice_app/config.py`:
  ```python
  @dataclass
  class AECConfig:
      enabled: bool = False
      filter_length: int = 2048   # samples; 128ms tail at 16kHz
      speaker_delay_ms: int = 0   # 0 = let speexdsp adapt
      ref_buffer_frames: int = 200  # 4s of reference at 20ms/frame
  ```
- Extend `TTSConfig` with `service: str = "deepgram"`, `voice: str = "aura-asteria-en"`
- Add `aec: AECConfig` to `AppConfig`; parse `raw.get("aec", {})` in `load_config`
- Update `config.yaml`:
  ```yaml
  aec:
    enabled: true
    filter_length: 2048
    speaker_delay_ms: 0
    ref_buffer_frames: 200
  tts:
    service: "deepgram"
    voice: "aura-asteria-en"
    rate: 175
    volume: 0.9
    echo_suppress_tail_ms: 350   # kept for aec.enabled=false fallback
  ```
- Update `tests/test_config.py`

### Step 2 — `voice_app/aec.py` (new module, fully isolated)
- **`SpeakerReferenceBuffer`**:
  - `threading.Lock`-protected `collections.deque(maxlen=ref_buffer_frames)`
  - Each entry: `(timestamp_ns: int, frame: bytes)`
  - `push(timestamp_ns, frame)` — called from PyAudio output callback
  - `get_frame_at(query_ns) -> bytes` — finds closest frame to `query_ns - speaker_delay_ns`; returns silence (`bytes(frame_size * 2)` for `linear16`) if buffer empty or closest entry is >2 frame durations away (`frame_size` = samples per frame, e.g. 320)
  - `clear()` — called on interrupt to avoid stale reference frames

- **`AECProcessor`**:
  - Holds `speexdsp.EchoCanceller.create(frame_size, filter_length, sample_rate)` ([PyPI `speexdsp` example](https://pypi.org/project/speexdsp/)): `frame_size` is **samples per frame** (320 to match capture), each frame is `frame_size * 2` bytes for `linear16`; bindings are **mono-only**
  - `process(mic_frame: bytes, mic_time_ns: int) -> bytes`
    1. `ref = self._ref_buffer.get_frame_at(mic_time_ns)`
    2. `return self._ec.process(mic_frame, ref)`
  - Graceful import: if `speexdsp` not installed, raise `ImportError` with install instructions
- Write `tests/test_aec.py`

### Step 3 — Rewrite `voice_app/tts.py`
- Replace `_speak_macos_say()` / pyttsx3 paths with:
  - `_fetch_pcm_deepgram(text, gen_id) -> bytes | None` — HTTP POST to Deepgram `/v1/speak` with `encoding=linear16&sample_rate=16000&container=none`; returns raw PCM; returns `None` if gen_id stale
  - `_pyaudio_callback(in_data, frame_count, time_info, status)` — pops one **320-sample** `linear16` frame (**640 bytes**) from `_playback_queue`; records `(monotonic_ns(), frame)` to `_ref_buffer` (if AEC enabled); returns `(frame, paContinue)`
  - `_run_pyaudio()` — main TTS worker loop: dequeues `(text, gen_id)`, fetches PCM (streaming the HTTP body is fine), slices into **640-byte** (`320-sample`) frames, enqueues to `_playback_queue`, waits for drain, emits `TTS_COMPLETE`
- **Preserve all existing public API**: `speak()`, `finish()`, `interrupt()`, `set_generation()`, `start()`, `stop()`
- `interrupt()` additionally clears `_playback_queue` and drains `_ref_buffer`
- `TTSEngine.__init__` accepts optional `ref_buffer: SpeakerReferenceBuffer | None`
- Update `tests/test_tts.py`; update `tests/conftest.py` (replace pyttsx3 fixture with PyAudio mock)

### Step 4 — Inject AEC into `voice_app/audio_capture.py`
- Add `aec_processor: AECProcessor | None = None` param to `AudioCapture.__init__`
- In `_capture_loop`, after reading each frame:
  ```python
  if self._aec_processor:
      frame = self._aec_processor.process(frame, time.monotonic_ns())
  ```
- This is a 3-line change; no existing tests need updating

### Step 5 — Simplify `voice_app/orchestrator.py`
- **If `config.aec.enabled`**:
  - **Remove**: `_echo_suppress_timer`, `_cancel_echo_suppress_timer()`, `_set_stt_echo_suppression()`, `_schedule_stt_echo_release()`, and all call sites (full-duplex path; STT receives cleaned audio)
  - In `__init__`: Create `SpeakerReferenceBuffer(config.aec.ref_buffer_frames, frame_size=320)` (samples; 640-byte frames)
  - Create `AECProcessor(config.aec, sample_rate=16000, frame_size=320, ref_buffer=ref_buffer)`
  - Pass `aec_processor` to `AudioCapture`, pass `ref_buffer` to `TTSEngine`
  - VAD loop: remove `barge_in_while_speaking` gate — always emit `INTERRUPT` when speech detected in SPEAKING state (AEC prevents false positives)
- **If `config.aec.enabled` is false**: **leave** current half-duplex echo suppression, `barge_in_while_speaking`, and `echo_suppress_tail_ms` behavior unchanged (matches `config.yaml` fallback and verification §4)
- Update `tests/test_orchestrator.py`

### Step 6 — Echo suppression in `voice_app/stt.py`
- Plumb `aec_enabled` from config into `STTClient` (or pass `AppConfig`) so one codebase can branch at init
- **If `config.aec.enabled`**: Remove `_suppress_echo`, `set_echo_suppression()`, and silence injection / transcript drops tied to it — `STTClient` is a pure passthrough (orchestrator no longer toggles suppression on this path)
- **If AEC disabled**: **keep** the existing half-duplex guard (same as today); orchestrator continues to drive `set_echo_suppression` as in current `orchestrator.py`

---

## Key Design Detail: Time Alignment

speexdsp is adaptive — it learns the room's acoustic delay within ~1 second of audio. The `SpeakerReferenceBuffer` uses timestamps (`time.monotonic_ns()`) so `AECProcessor` can find the reference frame that was playing when the mic frame was processed; stamp mic frames at capture (`_capture_loop`) and reference frames in the **output** callback so both sides use the same clock. Residual playback-buffer delay can be trimmed with `speaker_delay_ms` if needed. When TTS is silent, `get_frame_at()` returns a silence frame, which is the correct no-op input for the AEC.

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| `speexdsp` macOS install requires `brew install speexdsp` | `aec.enabled: false` by default — import only attempted when enabled; clear error message on failure |
| Deepgram TTS REST can show **~600ms+ TTFB** and **~700ms+** end-to-end in their docs' examples ([Text-to-speech latency](https://developers.deepgram.com/docs/text-to-speech-latency)) | Use **HTTP `stream=True`** and feed playback from the **first response bytes** ([Streaming the audio output](https://developers.deepgram.com/docs/streaming-the-audio-output)); optionally pipeline sentence N+1 while N plays ([Text chunking for TTS](https://developers.deepgram.com/docs/text-chunking-for-tts-optimization)) |
| PyAudio output callback is real-time — must not block | `_playback_queue.get_nowait()` with silence fallback; no locks in callback hot path |
| AEC convergence takes ~1s of audio | Acceptable for real conversations; AEC degrades gracefully to partial suppression during warmup |

---

## Verification

1. `pytest tests/` — all tests pass
2. Manual: `python main.py` with speakers (no headphones) — speak while agent responds; verify no feedback loop, no self-interrupts
3. Manual: interrupt agent mid-sentence by speaking — verify clean barge-in, agent stops and listens
4. Manual: run with `aec.enabled: false` in `config.yaml` — verify old half-duplex behavior still works (backward compat)

---

## Sources Consulted

- [Deepgram Voice Agent Echo Cancellation](https://developers.deepgram.com/docs/voice-agent-echo-cancellation)
- [OpenAI Realtime API VAD](https://platform.openai.com/docs/guides/realtime-vad)
- [speexdsp-python GitHub](https://github.com/xiongyihui/speexdsp-python)
- [Deepgram — Streaming the audio output](https://developers.deepgram.com/docs/streaming-the-audio-output)
- [Optimizing Voice Agent Barge-in Detection 2025](https://sparkco.ai/blog/optimizing-voice-agent-barge-in-detection-for-2025)
- [Real-Time vs Turn-Based Voice Agent Architecture](https://softcery.com/lab/ai-voice-agents-real-time-vs-turn-based-tts-stt-architecture)
