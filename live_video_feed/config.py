"""
config.py — Single source of truth for all CircuitMind settings.

Every other script imports from here.
Change a value once → it applies everywhere.
"""

import os
import re
import sys
import cv2
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

# Using laptop built-in webcam on /dev/video0 (YUYV format, 640x480)
# IPEVO V4K on /dev/video2 was producing corrupt frames in VirtualBox.
CAMERA_INDICES = [0]

# ─────────────────────────────────────────────────────────────────────────────
# Gemini
# ─────────────────────────────────────────────────────────────────────────────

GEMINI_MODEL = "gemini-1.5-flash"

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

CAPTURE_DIR  = Path("./captures")
DATASET_NAME = "circuitmind"

# ─────────────────────────────────────────────────────────────────────────────
# Drawing / overlay colors
# ─────────────────────────────────────────────────────────────────────────────

# Per-component-type colors in OpenCV BGR format.
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
DEFAULT_COLOR = (200, 200, 200)

# 3x3 grid: maps Gemini location strings to normalised [x, y, w, h] boxes.
# FiftyOne convention: origin top-left, all values in [0, 1].
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

def open_camera():
    """
    Try CAMERA_INDICES in order.

    Uses the camera's native format (YUYV for laptop webcam).
    Does NOT force MJPEG — that caused 'Corrupt JPEG data' spam on the
    built-in webcam and green frames on the IPEVO V4K in VirtualBox.

    Returns (VideoCapture, index_used) or calls sys.exit(1).
    """
    print(f"\n[Camera] Trying indices: {CAMERA_INDICES}")

    for idx in CAMERA_INDICES:
        print(f"[Camera] -> /dev/video{idx} ...", end=" ", flush=True)

        # CAP_V4L2 gives us direct V4L2 access (more reliable than auto-detect)
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            print("could not open")
            continue

        # Set resolution — do NOT set FOURCC, let camera use its native format
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Warmup: discard frames so auto-exposure and white balance can settle
        print("warming up...", end=" ", flush=True)
        for _ in range(10):
            cap.read()

        ret, frame = cap.read()
        if not ret or frame is None:
            print("opened but no frame returned")
            cap.release()
            continue

        # Report what we actually got
        w          = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h          = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join([chr((fourcc_int >> (8*i)) & 0xFF) for i in range(4)])
        brightness = round(frame.mean(), 1)

        print("OK")
        print(f"[Camera] ✓ /dev/video{idx}  {w}x{h}  fmt:{fourcc_str}  brightness:{brightness}")
        return cap, idx

    # Nothing worked
    print(
        f"\n[Camera] ✗ No working camera found (tried indices: {CAMERA_INDICES})\n"
        "  Run:  ls /dev/video*  to see available devices\n"
        "  Then update CAMERA_INDICES in config.py\n"
    )
    sys.exit(1)
