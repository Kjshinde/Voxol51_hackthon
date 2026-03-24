"""
config.py — Single source of truth for all CircuitMind settings.

Every other script imports from here.
Change a value once → it applies everywhere.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# FiftyOne: redirect MongoDB away from the VirtualBox virtual filesystem.
# This MUST be set before `import fiftyone` anywhere in the project.
# We do it here so every script that imports config gets it automatically.
# ─────────────────────────────────────────────────────────────────────────────
os.environ.setdefault("FIFTYONE_DATABASE_DIR", "/tmp/fiftyone_db")

# ─────────────────────────────────────────────────────────────────────────────
# Camera
# ─────────────────────────────────────────────────────────────────────────────

# Indices confirmed on this VM:
#   /dev/video0 — HP Wide Vision HD Camera (640×480)  [may be unavailable]
#   /dev/video2 — IPEVO V4K               (640×480)  [primary]
# The camera helpers will try these in order and use the first that works.
CAMERA_INDICES = [2, 0]

# ─────────────────────────────────────────────────────────────────────────────
# Gemini
# ─────────────────────────────────────────────────────────────────────────────

GEMINI_MODEL = "gemini-1.5-flash"

# Exact prompt sent to Gemini on every analysed frame.
# The schema here must stay consistent with the parsing code in gemini_utils.py.
GEMINI_PROMPT = """You are an expert electronics engineer.
Identify ALL electronic components visible in this image.
Respond with ONLY valid JSON — no markdown, no explanation, no extra text.
Use this exact schema:

{
  "components": [
    {
      "type": "resistor|capacitor|LED|IC|transistor|diode|inductor|other",
      "value": "e.g. 10k, 100uF, NE555, 2N2222, red, unknown",
      "location": "top-left|top-right|center|bottom-left|bottom-right|top-center|bottom-center",
      "confidence": "high|medium|low"
    }
  ]
}

If no components are visible, return:  {"components": []}"""

# ─────────────────────────────────────────────────────────────────────────────
# Frame sampling
# ─────────────────────────────────────────────────────────────────────────────

# Send 1 out of every N frames to Gemini.
# All other frames are displayed raw (keeps the window smooth).
FRAME_INTERVAL = 30

# ─────────────────────────────────────────────────────────────────────────────
# Storage
# ─────────────────────────────────────────────────────────────────────────────

CAPTURE_DIR  = Path("./captures")   # where JPEG snapshots are saved
DATASET_NAME = "circuitmind"        # FiftyOne persistent dataset name

# ─────────────────────────────────────────────────────────────────────────────
# Drawing / overlay
# ─────────────────────────────────────────────────────────────────────────────

# Per-component-type bounding-box colors in OpenCV BGR format.
# Consistent colors across all scripts (detection window + FiftyOne).
TYPE_COLORS = {
    "resistor":   (  0, 200,   0),   # green
    "capacitor":  (255, 140,   0),   # orange
    "led":        (  0,   0, 220),   # red
    "ic":         (220,   0,   0),   # blue
    "transistor": (  0, 180, 255),   # yellow
    "diode":      (200,   0, 200),   # magenta
    "inductor":   (  0, 220, 220),   # cyan
    "other":      (180, 180, 180),   # grey
}
DEFAULT_COLOR = (200, 200, 200)      # fallback if type not in dict

# 3×3 grid: maps Gemini's location strings to normalised [x, y, w, h] boxes.
# FiftyOne convention: origin top-left, values in [0, 1].
LOCATION_BOXES = {
    "top-left":      [0.00, 0.00, 0.33, 0.33],
    "top-center":    [0.33, 0.00, 0.34, 0.33],
    "top-right":     [0.67, 0.00, 0.33, 0.33],
    "center":        [0.33, 0.33, 0.34, 0.34],
    "bottom-left":   [0.00, 0.67, 0.33, 0.33],
    "bottom-center": [0.33, 0.67, 0.34, 0.33],
    "bottom-right":  [0.67, 0.67, 0.33, 0.33],
}
DEFAULT_BOX = [0.25, 0.25, 0.50, 0.50]   # fallback if location string unknown

# ─────────────────────────────────────────────────────────────────────────────
# Camera open helper  (shared by every milestone + main script)
# ─────────────────────────────────────────────────────────────────────────────

import cv2  # safe to import here; no GUI side effects at import time

def open_camera():
    """
    Try CAMERA_INDICES in order.
    Forces MJPEG + 640x480 @ 15 fps to prevent the green/corrupt-frame
    issue on the IPEVO V4K in VirtualBox (USB bandwidth problem).
    Returns (VideoCapture, index_used) or calls sys.exit(1).
    """
    import sys
    print(f"\n[Camera] Trying indices: {CAMERA_INDICES}")
    for idx in CAMERA_INDICES:
        print(f"[Camera] -> /dev/video{idx} ...", end=" ", flush=True)

        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            print("could not open")
            continue

        # Force MJPEG to avoid raw YUV over USB (causes green half-frame)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)   # lower FPS = less USB bandwidth pressure

        # Discard first few frames so camera can stabilise after format change
        print("warming up...", end=" ", flush=True)
        for _ in range(5):
            cap.read()

        ret, frame = cap.read()
        if not ret or frame is None:
            print("opened but no frame")
            cap.release()
            continue

        w          = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h          = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join([chr((fourcc_int >> (8*i)) & 0xFF) for i in range(4)])
        print("OK")
        print(f"[Camera] OK /dev/video{idx}  {w}x{h}  fmt:{fourcc_str}")
        return cap, idx

    print(
        f"\n[Camera] No working camera (tried {CAMERA_INDICES}).\n"
        "  VirtualBox -> Devices -> USB -> tick IPEVO V4K, then retry.\n"
    )
    sys.exit(1)
