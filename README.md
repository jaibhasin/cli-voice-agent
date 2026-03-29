# CLI Voice Agent 🎙️🤖

A lightning-fast, terminal-based voice conversational agent built in Python. Talk to an AI (like OpenAI's GPT-4o-mini) as naturally as a phone call directly from your command line.

This project implements a **true full-duplex** voice pipeline featuring real-time speech-to-text, local fast text-to-speech, and acoustic echo cancellation. It can listen to you *while* it's speaking, meaning you can interrupt the AI mid-sentence just by talking!

## Features

- **🗣️ Full-Duplex Conversation**: The AI uses Acoustic Echo Cancellation (AEC) to remove its own voice from your microphone. You don't need headphones to use it!
- **⚡ Ultra-Low Latency Pipeline**:
  - **VAD**: Local Silero VAD detects your speech instantly.
  - **STT**: Deepgram WebSockets provide millisecond-latency streaming transcription.
  - **LLM**: OpenAI GPT models stream responses token-by-token.
  - **TTS**: Local Piper TTS synthesizes speech quickly and plays it out via PyAudio.
- **🎤 Interruptible**: Say exactly what you think when you think of it—the AI immediately stops talking and listens.
- **💾 Conversation History**: Saves your chat histories to `conversation_history.json` so the AI remembers context between runs.

---

## Prerequisites

1. **Python 3.10+** (A Conda environment is recommended).
2. **API Keys**:
   - [Deepgram API Key](https://deepgram.com/) (For Speech-to-Text)
   - [OpenAI API Key](https://platform.openai.com/) (For the LLM)
3. **System Dependencies**:
   - For Acoustic Echo Cancellation, you need the `speexdsp` system library installed.
   - **macOS**: `brew install speexdsp`
   - **Ubuntu/Debian**: `sudo apt update && sudo apt install libspeexdsp-dev`

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/cli-voice-agent.git
   cd cli-voice-agent
   ```

2. **Set up the Python environment:**
   ```bash
   conda create -n cli-voice-agent python=3.12
   conda activate cli-voice-agent
   
   # Note: If installing on macOS, ensure you've run `brew install speexdsp` first
   pip install -r requirements.txt
   ```

3. **Download a Voice Model:**
   The project uses [Piper TTS](https://github.com/rhasspy/piper) to generate speech locally. You need to download an `.onnx` voice model (e.g., the `en_US-lessac-medium` voice).
   ```bash
   mkdir -p models
   
   # Download the model and config
   curl -L -o models/en_US-lessac-medium.onnx https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
   
   curl -L -o models/en_US-lessac-medium.onnx.json https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
   ```

4. **Configure your API Keys:**
   Create a `.env` file in the root directory and add your API keys:
   ```env
   OPENAI_API_KEY=sk-your-openai-key-here
   DEEPGRAM_API_KEY=your-deepgram-key-here
   ```

## Usage

Start the agent by running:

```bash
python main.py
```

To start a fresh conversation (wiping the memory of the previous chat):
```bash
python main.py --new
```

To see verbose debug information (including timing and interim STT transcripts):
```bash
python main.py --debug
```

## Configuration

You can heavily customize the pipeline's behavior by editing `config.yaml`.
- **System Prompt**: Change how the AI behaves and speaks.
- **VAD Sensitivity**: Adjust `aggressiveness` to make it easier or harder to trigger interrupts.
- **AEC Delay**: Tune the `speaker_delay_ms` if you find the echo cancellation struggling on your specific hardware.
- **Half-Duplex Fallback**: If full-duplex is unstable on your machine's built-in speakers, set `aec.enabled: false` to use aggressive mic-muting while the AI speaks.

## How It Works

The orchestrator spawns several worker threads that communicate via a central event queue:
1. `AudioCapture` continuously reads raw PCM audio from your mic (and applies AEC if enabled).
2. It fans the clean audio out to **Deepgram** (for transcription) and **Silero** (for VAD chunking).
3. Once you stop speaking, the LLM worker streams the transcript to OpenAI.
4. As sentences arrive from OpenAI, they are passed to the local Piper TTS subprocess.
5. Piper spits out raw PCM audio which is played out over your speakers in a real-time callback.
