# Plan: True Full-Duplex via Software AEC (speexdsp + Deepgram WS TTS + PyAudio)

## Context

The current implementation uses **half-duplex echo suppression** as a workaround: when TTS plays, the mic is silenced to Deepgram (silence bytes injected) so the agent can't hear itself. This works, but means the agent cannot listen while speaking. The `barge_in_while_speaking` flag exists but only works safely with headphones.

The goal is true full-duplex — agent speaks through speakers **and** listens at the same time, without feedback — using software Acoustic Echo Cancellation (AEC). The AEC subtracts the known speaker output from the mic signal before it reaches VAD/STT.

**Industry approach**: WebRTC browsers get this for free via built-in AEC. For Python CLI apps, the standard solution is `speexdsp` (SpeexDSP library) — the same algorithm used in PJSIP, Rhasspy, and many open-source voice agents. The key requirement: AEC needs the raw PCM being played through the speaker as a **reference signal**, which means TTS must produce and route PCM through PyAudio (not an opaque subprocess like `say`).

---

## Approach: PyAudio TTS Output + speexdsp AEC + Deepgram TTS WebSocket

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
| `voice_app/tts.py` | Replace `say` subprocess + pyttsx3 with a persistent Deepgram TTS WebSocket + PyAudio output stream + PCM reference tap |
| `voice_app/audio_capture.py` | Inject optional `aec_processor` — apply AEC to frames before queuing |
| `voice_app/config.py` | Add `AECConfig` dataclass; extend `AppConfig` and `TTSConfig` |
| `voice_app/orchestrator.py` | When `aec.enabled`, remove echo-suppression timers and wire `AECProcessor` → `AudioCapture`; when disabled, keep current behavior |
| `voice_app/stt.py` | When `aec.enabled`, drop echo suppression; when disabled, keep `set_echo_suppression` / silence path |
| `config.yaml` | Add `aec:` section; add TTS `service`/`voice` fields; bump default STT model from `nova-2` to `nova-3` |
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
- Extend `TTSConfig` with `service: str = "deepgram"`, `voice: str = "aura-2-thalia-en"`
- Add `aec: AECConfig` to `AppConfig`; parse `raw.get("aec", {})` in `load_config`
- Refresh the STT baseline in `config.yaml`: set `deepgram.model: "nova-3"` by default. Keep the current `endpointing` / `utterance_end_ms` path in this plan. Do **not** switch to Flux inside the AEC rollout, because Flux changes the turn-detection contract and would confound the migration.
- Update `config.yaml`:
  ```yaml
  deepgram:
    model: "nova-3"
    language: "en"
    endpointing: 300
    utterance_end_ms: 1000
  aec:
    enabled: true
    filter_length: 2048
    speaker_delay_ms: 0
    ref_buffer_frames: 200
  tts:
    service: "deepgram"
    voice: "aura-2-thalia-en"
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
- Replace `_speak_macos_say()` / pyttsx3 paths with a single persistent Deepgram TTS WebSocket + PyAudio output pipeline:
  - `_open_tts_ws()` — open one `deepgram.speak.v1.connect(...)` websocket per conversation with `model=config.tts.voice`, `encoding="linear16"`, `sample_rate=16000`; keep voice/media settings fixed for the session
  - `_tts_ws_on_message(message)` — when `message` is `bytes`, slice it into **640-byte** (`320-sample`) frames and enqueue into a **bounded** `_playback_queue`; when it is metadata / control output, log or ignore
  - `_pyaudio_callback(in_data, frame_count, time_info, status)` — pops one **320-sample** `linear16` frame (**640 bytes**) from `_playback_queue`; records `(monotonic_ns(), frame)` to `_ref_buffer` (if AEC enabled); returns `(frame, paContinue)`; on underrun, returns silence
  - `_run_tts_ws()` — main TTS worker loop: maintain the websocket and its listener, dequeue `(text, gen_id)` sentence chunks, `send_text(...)` each chunk immediately, and send `Flush` only when an assistant turn is complete (`finish(gen_id)`) or a hard boundary must be forced
- Keep the existing sentence chunking in `voice_app/llm.py`; do **not** flush on every token. Deepgram documents a `Flush` limit of **20 per 60 seconds**, and very frequent flushes can degrade quality.
- **Preserve all existing public API**: `speak()`, `finish()`, `interrupt()`, `set_generation()`, `start()`, `stop()`
- `interrupt()` additionally clears `_playback_queue`, drains `_ref_buffer`, and sends `Clear` on the TTS websocket; if the connection is ambiguous or stale audio keeps arriving, close and recreate the websocket before the next turn
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
- Keep Deepgram STT on the existing WebSocket path, but make `nova-3` the default model in config for this migration. Evaluate Flux only after AEC stabilizes, because Flux replaces the current endpointing/utterance-end behavior with model-integrated turn detection.
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
| Deepgram TTS WebSocket requires correct `Flush` / `Clear` / `Close` handling, and `Flush` is limited to **20 per 60s** | Keep sentence-sized text chunking, send `Flush` only at assistant-turn end (or deliberate hard boundaries), use `Clear` on interrupt, and recreate the socket on protocol ambiguity |
| PyAudio output callback is real-time — must not block | `_playback_queue.get_nowait()` with silence fallback in the callback; keep `_playback_queue` **bounded** so the websocket receiver can block and apply normal TCP backpressure instead of growing memory unbounded |
| AEC convergence takes ~1s of audio | Acceptable for real conversations; AEC degrades gracefully to partial suppression during warmup |

---

## Verification

1. `pytest tests/` — all tests pass
2. Manual: `python main.py` with speakers (no headphones) — speak while agent responds; verify no feedback loop, no self-interrupts
3. Manual: interrupt agent mid-sentence by speaking — verify clean barge-in, agent stops and listens
4. Manual: run with `aec.enabled: false` in `config.yaml` — verify old half-duplex behavior still works (backward compat)

---

## Post-Implementation Fix: Speaker Reference Timing Alignment (2026-03-26)

### Problem

After the AEC pipeline was wired up, the agent was still self-interrupting: VAD detected speech while the agent was speaking, causing it to interrupt itself mid-sentence.

**Root cause**: The AEC was receiving a misaligned reference signal. The PyAudio output callback fires when a frame is *queued to the hardware buffer*, not when it physically exits the speaker. By the time the mic captures that frame, 40–80 ms of hardware + acoustic propagation delay has elapsed. So `get_frame_at(mic_time_ns)` was finding the reference frame queued at `mic_time_ns`, not the one *physically playing* at `mic_time_ns` — speexdsp was cancelling the wrong echo frame and passing the speaker bleed through to VAD unchecked.

### Fix

**`voice_app/aec.py`** — `SpeakerReferenceBuffer.get_frame_at()`:
- Added `speaker_delay_ms: int = 0` parameter (default = no change).
- Before the buffer scan: `adjusted_ns = query_ns - speaker_delay_ms * 1_000_000`.
- Buffer searches against `adjusted_ns` instead of `query_ns`, finding the frame queued at the moment it was *actually playing* in the mic.

**`voice_app/aec.py`** — `AECProcessor.__init__()`:
- Reads `config.speaker_delay_ms` and stores as `self._speaker_delay_ms`.

**`voice_app/aec.py`** — `AECProcessor.process()`:
- Forwards `self._speaker_delay_ms` to `get_frame_at()`.
- Added explicit comment confirming `ec.process(mic_frame, ref)` argument order (silent wrong-output if reversed).

**`calibrate_aec.py`** (new, project root):
- Standalone script: plays one click frame via PyAudio output, reads mic frames until peak > 8000, computes and prints `delay_ms = (mic_time_ns - ref_time_ns) / 1_000_000`, and outputs the exact YAML line to paste into `config.yaml`.
- Timeout: 3 seconds with actionable guidance if no click detected.
- No new dependencies (`pyaudio` + `numpy` already in `requirements.txt`).

**`config.yaml`** — `aec.speaker_delay_ms: 0` already present (default keeps existing behaviour unchanged for users who skip calibration).

**`tests/test_aec.py`** — 4 new tests added (15 total, all pass):
- `test_get_frame_at_zero_delay_returns_closest_frame` — zero delay = original behaviour
- `test_get_frame_at_nonzero_delay_shifts_lookup_back` — 60 ms shift returns earlier frame
- `test_get_frame_at_empty_buffer_returns_silence_with_delay` — empty + delay = silence
- `test_process_passes_speaker_delay_to_get_frame_at` — end-to-end: AECProcessor reads config delay and correct reference reaches `ec.process()`

### Calibration Workflow

```bash
python calibrate_aec.py
# → Speaker delay detected: 62.3ms
# → Add this to config.yaml under aec.speaker_delay_ms:
#     aec:
#       speaker_delay_ms: 62
```

Paste the printed value into `config.yaml`. Typical macOS values: 40–80 ms. With this value set, speexdsp receives time-aligned (mic, reference) pairs and converges immediately rather than adapting over the first ~1 s of each turn.

---

## Sources Consulted

- [Deepgram Voice Agent Echo Cancellation](https://developers.deepgram.com/docs/voice-agent-echo-cancellation)
- [Deepgram Text-to-Speech Streaming](https://developers.deepgram.com/docs/tts-websocket)
- [Deepgram Streaming Text-to-Speech Getting Started](https://developers.deepgram.com/docs/streaming-text-to-speech)
- [Deepgram Flush Control Message](https://developers.deepgram.com/docs/tts-ws-flush)
- [Deepgram Clear Control Message](https://developers.deepgram.com/docs/tts-ws-clear)
- [Deepgram Compare Flux to Nova-3](https://developers.deepgram.com/docs/flux/flux-nova-3-comparison)
- [speexdsp-python GitHub](https://github.com/xiongyihui/speexdsp-python)
- [Deepgram Real-Time TTS with WebSockets](https://developers.deepgram.com/docs/tts-websocket-streaming)
- [Deepgram Text Chunking for TTS](https://developers.deepgram.com/docs/tts-text-chunking)
- [OpenAI Realtime API Reference](https://developers.openai.com/api/reference/resources/realtime)
- [websockets Buffer Limits](https://websockets.readthedocs.io/en/stable/reference/legacy/common.html)
