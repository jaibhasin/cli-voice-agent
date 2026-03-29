"""
calibrate_aec.py — Measure the speaker→microphone round-trip delay.

How it works
============
The script opens a PyAudio output stream (callback-based) and a separate
blocking input stream, both at 16 kHz / mono / int16 — matching the live
voice app's audio pipeline.

On the very first output callback it:
  1. Records ref_time_ns = time.monotonic_ns()
  2. Returns a single "click" frame: a short burst of alternating +32767 /
     −32768 samples loud enough to survive the speaker→room→mic path.

All subsequent output callbacks return silence so the click is unambiguous.

The main thread reads mic frames in a tight loop, converts each to
numpy.int16, and looks for a peak amplitude above CLICK_THRESHOLD.  When
found it computes:

    delay_ms = (mic_time_ns - ref_time_ns) / 1_000_000

and prints the value ready to paste into config.yaml.

Usage
-----
  python calibrate_aec.py

Requirements
------------
  pip install pyaudio numpy       (both already in requirements.txt)
  macOS: system speaker + built-in mic (or headphones — quieter rooms give
         cleaner measurements).

Typical results
---------------
  macOS built-in speaker + mic:  40–80 ms
  Bluetooth speaker:             150–300 ms (latency-heavy; not recommended)
"""

import sys
import time

import numpy as np
import pyaudio

# ---------------------------------------------------------------------------
# Audio constants — must match the voice app pipeline
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16_000          # Hz
FRAME_SIZE = 320              # samples per 20 ms frame
FRAME_BYTES = FRAME_SIZE * 2  # int16 = 2 bytes per sample

# ---------------------------------------------------------------------------
# Calibration constants
# ---------------------------------------------------------------------------
# Detection threshold: peak amplitude in the mic frame that signals the click.
# 32767 is the int16 maximum; we use 8000 (~25 % full scale) to be robust
# against background noise while avoiding false positives from ambient sound.
CLICK_THRESHOLD = 8000

# Build the click frame: alternating max (+32767) and min (−32768) samples.
# This maximises energy across all frequencies for a single 20 ms burst.
_click_samples = np.array(
    [32767 if i % 2 == 0 else -32768 for i in range(FRAME_SIZE)],
    dtype=np.int16,
)
CLICK_FRAME: bytes = _click_samples.tobytes()

# Silent frame used for all callbacks after the first
SILENCE_FRAME: bytes = bytes(FRAME_BYTES)

# How long to wait for the click to appear in the mic before giving up
TIMEOUT_S = 3.0


def main() -> None:
    """
    Open output + input streams, play one click, measure the mic arrival time.

    Exit codes
    ----------
    0 — delay successfully measured and printed
    1 — timeout or error (message printed to stderr)
    """
    pa = pyaudio.PyAudio()

    # Shared state between the output callback and the main thread.
    # Using a mutable list avoids the need for a full threading.Event object
    # while still being safe for single-writer / single-reader usage.
    ref_time_ns: list[int] = [0]    # [0] set once by the first output callback
    click_sent: list[bool] = [False]  # becomes True after the click frame is returned

    # ------------------------------------------------------------------
    # PyAudio output callback
    # ------------------------------------------------------------------

    def output_callback(in_data, frame_count, time_info, status):
        """
        Real-time output callback.  Must not block.

        First call: stamp ref_time_ns and return the click frame.
        All subsequent calls: return silence to keep the speaker quiet.
        """
        if not click_sent[0]:
            # Record the moment we hand the click to the hardware buffer.
            # This is *before* the click physically exits the speaker, so
            # delay_ms will represent the full hardware + acoustic path.
            ref_time_ns[0] = time.monotonic_ns()
            click_sent[0] = True
            return (CLICK_FRAME, pyaudio.paContinue)
        return (SILENCE_FRAME, pyaudio.paContinue)

    # ------------------------------------------------------------------
    # Open streams
    # ------------------------------------------------------------------

    out_stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        output=True,
        frames_per_buffer=FRAME_SIZE,
        stream_callback=output_callback,
    )

    in_stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=FRAME_SIZE,
    )

    out_stream.start_stream()

    print("Playing click… listening for echo in the microphone.")
    print(f"(Timeout: {TIMEOUT_S:.0f} s — make sure your speaker volume is audible)")

    # ------------------------------------------------------------------
    # Detection loop
    # ------------------------------------------------------------------

    deadline = time.monotonic() + TIMEOUT_S
    delay_ms: float | None = None

    while time.monotonic() < deadline:
        # Blocking read: waits until a full frame is available from the mic.
        raw = in_stream.read(FRAME_SIZE, exception_on_overflow=False)
        mic_time_ns = time.monotonic_ns()

        # Skip frames captured before the click was even sent
        if not click_sent[0]:
            continue

        # Convert raw bytes to a numpy int16 array to find the peak amplitude
        samples = np.frombuffer(raw, dtype=np.int16)
        peak = int(np.max(np.abs(samples)))

        if peak >= CLICK_THRESHOLD:
            # The click has arrived at the microphone.
            delay_ms = (mic_time_ns - ref_time_ns[0]) / 1_000_000
            break

    # ------------------------------------------------------------------
    # Clean up
    # ------------------------------------------------------------------

    try:
        in_stream.stop_stream()
        in_stream.close()
        out_stream.stop_stream()
        out_stream.close()
    except Exception:
        pass
    finally:
        pa.terminate()

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    if delay_ms is None:
        print(
            "\n[TIMEOUT] No click detected within the timeout window.",
            file=sys.stderr,
        )
        print(
            "  • Try increasing speaker volume.\n"
            "  • Make sure you are not using headphones (the mic won't hear them).\n"
            "  • If the room is very noisy, reduce background noise and retry.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\nSpeaker delay detected: {delay_ms:.1f}ms")
    print(f"Add this to config.yaml under aec.speaker_delay_ms:\n")
    print(f"  aec:")
    print(f"    speaker_delay_ms: {int(round(delay_ms))}")


if __name__ == "__main__":
    main()
