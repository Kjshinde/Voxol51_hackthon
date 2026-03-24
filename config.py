"""
config.py — Single source of truth for all CircuitMind settings.

Every other script imports from here.
Change a value once, it applies everywhere.

Mode: PICTURE MODE (no camera)
  Drop .jpg / .png images into the ./pictures/ folder.
  The pipeline processes them one by one.
"""

import os
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# FiftyOne: redirect MongoDB to /tmp so it works inside VirtualBox.
# Must be set BEFORE import fiftyone — importing config first handles this.
# ─────────────────────────────────────────────────────────────────────────────
# Use system MongoDB (FiftyOne bundled mongod requires AVX which VirtualBox blocks)
# MongoDB 6 installed via apt does NOT need AVX and works fine

# ─────────────────────────────────────────────────────────────────────────────
# Input pictures
# ─────────────────────────────────────────────────────────────────────────────

# Drop your electronics images here (any filename, .jpg or .png)
PICTURES_DIR = Path("./pictures")

# Supported extensions (case-insensitive)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# ─────────────────────────────────────────────────────────────────────────────
# Gemini
# ─────────────────────────────────────────────────────────────────────────────

GEMINI_MODEL = "models/gemini-2.5-flash"

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
# Storage
# ─────────────────────────────────────────────────────────────────────────────

DATASET_NAME = "circuitmind"

# ─────────────────────────────────────────────────────────────────────────────
# Drawing / overlay colors  (OpenCV BGR format)
# ─────────────────────────────────────────────────────────────────────────────

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

# 3x3 grid: Gemini location strings -> normalised FiftyOne [x, y, w, h] boxes
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
# Image loader helper  (shared by every milestone + main script)
# ─────────────────────────────────────────────────────────────────────────────

def get_image_paths():
    """
    Scan PICTURES_DIR for all supported image files.
    Returns a sorted list of Path objects.
    Exits with a clear message if the folder is missing or empty.
    """
    if not PICTURES_DIR.exists():
        print("[Images] pictures/ folder not found — creating it for you")
        PICTURES_DIR.mkdir(parents=True)
        print("[Images] Drop your .jpg / .png electronics images into: " + str(PICTURES_DIR.resolve()))
        sys.exit(1)

    paths = sorted([
        p for p in PICTURES_DIR.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ])

    if not paths:
        print("[Images] No images found in: " + str(PICTURES_DIR.resolve()))
        print("[Images] Drop .jpg or .png files there and re-run")
        sys.exit(1)

    print("[Images] Found " + str(len(paths)) + " image(s) in " + str(PICTURES_DIR) + ":")
    for p in paths:
        size_kb = round(p.stat().st_size / 1024, 1)
        print("  " + p.name + "  (" + str(size_kb) + " KB)")

    return paths
