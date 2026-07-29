# 🖐 Palm Harmonizer

A real-time interactive music tool: show your **open palm** to the webcam and it auto-generates lush harmonies from your voice. Close your hand → harmonies stop instantly.

---

## How It Works

1. **Webcam** (MediaPipe Hands) detects whether your palm is open.
2. **Microphone** captures your voice continuously.
3. When palm is open → pitch detected (aubio YIN) → 3–4 sine-wave harmony voices generated and mixed.
4. Everything runs in parallel threads with < 100ms latency.

---

## Setup (Intel MacBook Pro + VS Code)

### Prerequisites

Make sure you have **Python 3.10 or 3.11** installed. Check with:

```bash
python3 --version
```

If you don't have it, install via [python.org](https://python.org) or Homebrew:
```bash
brew install python@3.11
```

You also need **PortAudio** (required by sounddevice):
```bash
brew install portaudio
```

---

### 1. Open in VS Code

```bash
# Clone or place the project folder, then:
cd palm_harmonizer
code .
```

### 2. Create a virtual environment

Open the **VS Code integrated terminal** (`Ctrl + `` ` ```) and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` in your terminal prompt.

In VS Code, press `Cmd+Shift+P` → "Python: Select Interpreter" → choose the `.venv` option.

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note on `aubio`:** If the pip install fails on Intel Mac, try:
> ```bash
> brew install aubio
> pip install aubio
> ```

> **Note on `mediapipe`:** MediaPipe 0.10.x works on Intel Macs with Python 3.10/3.11. If you hit issues on Python 3.12, downgrade to 3.11.

### 4. Camera & Microphone permissions

On macOS you **must** grant Terminal/VS Code permission to access your camera and microphone:

- System Settings → Privacy & Security → **Camera** → enable for Terminal / VS Code
- System Settings → Privacy & Security → **Microphone** → enable for Terminal / VS Code

### 5. Run it

```bash
python main.py
```

An OpenCV window will open showing your webcam feed. **Show an open palm** and start singing or humming!

---

## Keyboard Controls

| Key | Action |
|-----|--------|
| `Q` / `ESC` | Quit |
| `R` | Reset / recalibrate gesture state and pitch |
| `H` | Toggle hearing your own voice through speakers |
| `V` | Toggle Voice Activity Detection (VAD) |
| `M` | Cycle through harmony modes |
| `S` | Save a screenshot (`screenshot_000.png`, etc.) |

---

## Harmony Modes

Cycle through them with `M`:

| Mode | Intervals | Sound |
|------|-----------|-------|
| `major` | Root, M3rd, 5th, Octave | Bright, happy |
| `minor` | Root, m3rd, 5th, Octave | Dark, emotive |
| `power` | Root, 5th, Octave, 5th+Oct | Rock power chord |
| `major7` | Root, M3, 5th, M7, Oct | Jazz warmth |
| `sus2` | Root, 2nd, 5th, Oct | Ambient, open |

---

## Fine-Tuning Parameters

All tunable parameters are at the top of each file:

### `gesture.py`
| Parameter | Default | Effect |
|-----------|---------|--------|
| `ACTIVATE_FRAMES` | 3 | Frames of open palm needed to activate |
| `DEACTIVATE_FRAMES` | 5 | Frames without palm needed to deactivate |
| `MIN_DETECTION_CONF` | 0.7 | MediaPipe detection confidence |
| `EXTENSION_THRESHOLD` | 0.15 | How extended fingers must be |

### `audio_processor.py`
| Parameter | Default | Effect |
|-----------|---------|--------|
| `BLOCK_SIZE` | 512 | Audio latency (lower = faster but more CPU) |
| `VAD_RMS_THRESH` | 0.008 | Silence threshold (raise if background noise) |
| `PITCH_CONF_MIN` | 0.65 | Pitch confidence threshold |
| `PITCH_SMOOTH` | 0.25 | Pitch smoothing (lower = smoother but slower) |
| `PITCH_RANGE_HZ` | (60, 1200) | Valid singing frequency range |

### `synthesizer.py`
| Parameter | Default | Effect |
|-----------|---------|--------|
| `ATTACK_MS` | 25 | Fade-in time (ms) per voice |
| `RELEASE_MS` | 60 | Fade-out time (ms) per voice |
| `VOICE_GAIN` | 0.22 | Per-voice volume |

---

## Project Structure

```
palm_harmonizer/
├── main.py            # Entry point — video loop, UI, keyboard controls
├── gesture.py         # MediaPipe hand detection + open-palm logic
├── audio_processor.py # Microphone capture, pitch detection, harmony control
├── synthesizer.py     # Sine-wave oscillator engine
├── utils.py           # Note conversion, OpenCV drawing helpers
├── requirements.txt
└── README.md
```

---

## Troubleshooting

### "Cannot open camera"
- Make sure no other app is using the camera
- Try changing `CAMERA_INDEX = 1` in `main.py`
- Grant camera permission to Terminal/VS Code

### "Could not start audio"
- Make sure PortAudio is installed: `brew install portaudio`
- Grant microphone permission to Terminal/VS Code
- Check `python -c "import sounddevice; print(sounddevice.query_devices())"` to see available devices

### Harmonies sound noisy/pitchy during silence
- Raise `VAD_RMS_THRESH` in `audio_processor.py` (e.g. `0.015`)
- Or press `V` to toggle VAD off/on live

### Gesture detection is too sensitive / not sensitive enough
- Adjust `ACTIVATE_FRAMES` and `DEACTIVATE_FRAMES` in `gesture.py`
- Adjust `MIN_DETECTION_CONF` (raise it if too many false positives)

### High CPU usage
- Increase `BLOCK_SIZE` to `1024` in `audio_processor.py` (slightly higher latency)
- Reduce camera resolution: change `FRAME_WIDTH/HEIGHT` to `640`/`480`

### mediapipe install fails on macOS
```bash
pip install mediapipe --no-deps
pip install numpy opencv-python protobuf attrs
```

---

## Extending the Project

- **Add a harmony mode**: add an entry to `HARMONY_MODES` dict in `utils.py`
- **Change synthesis**: replace `SineVoice` in `synthesizer.py` with a richer waveform (square, sawtooth via Fourier series)
- **MIDI output**: use `python-rtmidi` to send MIDI note-on/off messages from `audio_processor.py`
- **Session recording**: use `soundfile` to write mic + harmony mix to a WAV file

---

## Requirements

```
opencv-python>=4.8.0
mediapipe>=0.10.9
numpy>=1.24.0
sounddevice>=0.4.6
aubio>=0.4.9
scipy>=1.11.0
```
