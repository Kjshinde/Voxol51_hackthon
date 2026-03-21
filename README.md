# CircuitMind 🔬⚡

> **Real-time AI Lab Supervisor for Electronics**  
> Point your webcam at a workbench. CircuitMind identifies components, draws bounding boxes, and prints a live inventory — powered by Google Gemini Vision.

---

## What It Does

CircuitMind watches your electronics workbench through a webcam and uses the Google Gemini Vision API to recognise components in real time. Every 30 frames it sends a snapshot to Gemini, receives a structured JSON response, and overlays colour-coded bounding boxes on the live video feed while printing a running inventory to the terminal.

**Detects:** resistors, capacitors, ICs, LEDs, transistors, diodes, and other components.

---

## Demo Output

```
──────────────────────────────────────────────────
  CircuitMind — Scan at 14:22:07  (1.8s API)
──────────────────────────────────────────────────
  1×  ic [NE555]
  2×  led [red]
  3×  resistor [10kΩ]
  1×  capacitor [100µF]
──────────────────────────────────────────────────
  INVENTORY: 1x ic [NE555], 2x led [red], 3x resistor [10kΩ], 1x capacitor [100µF]
```

Live OpenCV window shows colour-coded boxes:

| Component   | Colour  |
|-------------|---------|
| Resistor    | Orange  |
| Capacitor   | Green   |
| IC          | Red     |
| LED         | Blue    |
| Transistor  | Yellow  |
| Diode       | Purple  |

---

## Project Roadmap

| Part | Status | Description |
|------|--------|-------------|
| **Part 1** | ✅ Complete | Real-time webcam detection + terminal inventory |
| Part 2 | 🔜 Planned | Auto-schematic generation from detected components |
| Part 3 | 🔜 Planned | FiftyOne dataset browser + scan history |
| Part 4 | 🔜 Planned | REST API + web dashboard |

---

## Requirements

- **OS:** Linux (Ubuntu 22.04 / 24.04 recommended)
- **Python:** 3.10 or higher
- **Webcam:** USB or built-in (V4L2 compatible)
- **API key:** Google Gemini ([get one free](https://aistudio.google.com/app/apikey))

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/circuitmind.git
cd circuitmind
```

### 2. (Recommended) Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Install system-level dependencies

These are needed for OpenCV's display backend on Linux:

```bash
sudo apt update
sudo apt install -y libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 v4l-utils
```

### 5. Set your Gemini API key

```bash
export GEMINI_API_KEY="your_key_here"
```

To make this permanent, add the line to your `~/.bashrc` or `~/.zshrc`:

```bash
echo 'export GEMINI_API_KEY="your_key_here"' >> ~/.bashrc
source ~/.bashrc
```

---

## Usage

```bash
python3 circuitmind.py
```

- Point your webcam at any electronics (breadboard, PCB, loose components)
- Bounding boxes and labels update every 30 frames
- Press **Q** or **Esc** to quit

### Check available webcams (optional)

```bash
v4l2-ctl --list-devices
```

If your camera isn't index `0`, edit this line at the top of `circuitmind.py`:

```python
WEBCAM_INDEX = 1   # try 1, 2, etc.
```

---

## Configuration

All tunable settings are at the top of `circuitmind.py`:

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | `gemini-1.5-flash` | Gemini model to use. Swap to `gemini-1.5-pro` for higher accuracy |
| `PROCESS_EVERY` | `30` | Analyse every Nth frame. Increase to reduce API calls |
| `WEBCAM_INDEX` | `0` | Camera index. Change if your webcam isn't the default device |

---

## Project Structure

```
circuitmind/
├── circuitmind.py      # Main script — Part 1
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── .gitignore          # (recommended — see below)
```

---

## .gitignore (recommended)

Create a `.gitignore` to keep secrets and junk out of the repo:

```
.venv/
__pycache__/
*.pyc
.env
*.jpg
*.png
*.mp4
```

**Never commit your `GEMINI_API_KEY`.** Always load it from an environment variable or a `.env` file that is listed in `.gitignore`.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `GEMINI_API_KEY not set` | Run `export GEMINI_API_KEY="..."` before launching |
| `Could not open webcam (index 0)` | Try `WEBCAM_INDEX = 1` or `2` |
| OpenCV window doesn't appear | Run `sudo apt install libgl1 libglib2.0-0` |
| JSON parse errors in terminal | Usually transient — if persistent, switch to `gemini-1.5-pro` |
| API quota / rate limit errors | Increase `PROCESS_EVERY` to `60` or `90` |
| Slow performance | Add `cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)` after `VideoCapture()` |

---

## How It Works

```
Webcam frame (every 30th)
        │
        ▼
 BGR → PIL RGB conversion
        │
        ▼
 Gemini Vision API
 (gemini-1.5-flash)
        │
        ▼
 JSON: [{type, value, region, confidence}, …]
        │
        ├──► Draw coloured bounding boxes on live frame (OpenCV)
        │
        └──► Print inventory summary to terminal
```

Gemini returns detections mapped to a 3×3 grid (top-left, middle-center, etc.) which CircuitMind converts to pixel bounding boxes on the video frame.

---

## Contributing

Pull requests are welcome. For major changes please open an issue first to discuss what you'd like to change.

---

## License

MIT — see `LICENSE` for details.
