"""
CircuitMind — Part 1
Real-time electronics component detection via Google Gemini Vision + OpenCV.

Usage:
    export GEMINI_API_KEY="your_key_here"
    python3 circuitmind.py
"""

import cv2
import json
import os
import re
import time
from collections import defaultdict

import google.generativeai as genai
import numpy as np
from PIL import Image

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL_NAME     = "gemini-1.5-flash"   # swap to "gemini-1.5-pro" if preferred
PROCESS_EVERY  = 30                   # analyse every Nth frame
WEBCAM_INDEX   = 0                    # 0 = default webcam; try 1, 2 … if needed

# Bounding-box colours per component type  (BGR)
COLOURS = {
    "resistor":   (0,   200, 255),
    "capacitor":  (0,   255, 128),
    "ic":         (255, 80,  80 ),
    "led":        (80,  80,  255),
    "transistor": (255, 200, 0  ),
    "diode":      (180, 0,   255),
    "default":    (200, 200, 200),
}

# ─────────────────────────────────────────────
#  GEMINI SETUP
# ─────────────────────────────────────────────
if not GEMINI_API_KEY:
    raise SystemExit(
        "❌  GEMINI_API_KEY not set.\n"
        "    Run:  export GEMINI_API_KEY='your_key_here'\n"
        "    then re-launch the script."
    )

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

# ─────────────────────────────────────────────
#  PROMPT
# ─────────────────────────────────────────────
DETECTION_PROMPT = """
You are an electronics component recognition system.

Analyse the image and detect all visible electronic components.
For EVERY component you find return a JSON array — and ONLY the JSON array,
no markdown fences, no explanation.

Each element must be an object with exactly these keys:
  "type"   : component category, one of:
             resistor | capacitor | ic | led | transistor | diode | other
  "value"  : specific identifier if visible (e.g. "10kΩ", "100µF", "NE555", "red")
             or "unknown" if not readable
  "region" : one of:
             top-left | top-center | top-right |
             middle-left | middle-center | middle-right |
             bottom-left | bottom-center | bottom-right
  "confidence": high | medium | low

Example output (do NOT copy these values; detect from the actual image):
[
  {"type":"resistor","value":"10kΩ","region":"top-left","confidence":"high"},
  {"type":"led","value":"red","region":"middle-center","confidence":"high"}
]

If no components are visible return an empty array: []
"""

# ─────────────────────────────────────────────
#  REGION → PIXEL BOX
# ─────────────────────────────────────────────
def region_to_box(region: str, w: int, h: int):
    """
    Map a named 3×3 grid region to (x1, y1, x2, y2) pixel coordinates.
    Adds a small random jitter so overlapping labels don't pile up.
    """
    col, row = {
        "top-left":      (0, 0), "top-center":    (1, 0), "top-right":    (2, 0),
        "middle-left":   (0, 1), "middle-center": (1, 1), "middle-right": (2, 1),
        "bottom-left":   (0, 2), "bottom-center": (1, 2), "bottom-right": (2, 2),
    }.get(region.lower().replace("_", "-"), (1, 1))

    cw, ch = w // 3, h // 3
    x1 = col * cw + 4
    y1 = row * ch + 4
    x2 = x1 + cw - 8
    y2 = y1 + ch - 8

    # slight random offset so multiple components in same region separate visually
    jitter = 12
    dx = int((np.random.random() - 0.5) * jitter)
    dy = int((np.random.random() - 0.5) * jitter)
    return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)


# ─────────────────────────────────────────────
#  CALL GEMINI
# ─────────────────────────────────────────────
def detect_components(frame_bgr: np.ndarray) -> list[dict]:
    """Send a BGR frame to Gemini and return parsed component list."""
    # OpenCV BGR → PIL RGB
    pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    try:
        response = model.generate_content([DETECTION_PROMPT, pil_img])
        raw = response.text.strip()

        # Strip possible markdown code fences just in case
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        raw = raw.strip()

        components = json.loads(raw)
        if not isinstance(components, list):
            return []
        return components

    except json.JSONDecodeError as e:
        print(f"[WARN] JSON parse error: {e}\n       Raw response: {raw[:200]}")
        return []
    except Exception as e:
        print(f"[WARN] Gemini error: {e}")
        return []


# ─────────────────────────────────────────────
#  DRAW OVERLAYS
# ─────────────────────────────────────────────
def draw_detections(frame: np.ndarray, components: list[dict]) -> np.ndarray:
    h, w = frame.shape[:2]
    overlay = frame.copy()

    for comp in components:
        ctype  = comp.get("type",   "other").lower()
        value  = comp.get("value",  "unknown")
        region = comp.get("region", "middle-center")
        conf   = comp.get("confidence", "medium")

        colour = COLOURS.get(ctype, COLOURS["default"])
        x1, y1, x2, y2 = region_to_box(region, w, h)

        # Semi-transparent filled rectangle
        cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, -1)

        alpha = 0.15
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Solid border
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

        # Label background
        label = f"{ctype}: {value}"
        if conf == "low":
            label += "?"
        (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - lh - baseline - 4), (x1 + lw + 4, y1), colour, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    return frame


# ─────────────────────────────────────────────
#  PRINT INVENTORY
# ─────────────────────────────────────────────
def print_inventory(components: list[dict], elapsed: float):
    inventory: dict[str, int] = defaultdict(int)
    for comp in components:
        key = f"{comp.get('type','other')} [{comp.get('value','?')}]"
        inventory[key] += 1

    print("\n" + "─" * 50)
    print(f"  CircuitMind — Scan at {time.strftime('%H:%M:%S')}  ({elapsed:.1f}s API)")
    print("─" * 50)
    if inventory:
        for item, count in sorted(inventory.items()):
            print(f"  {count}×  {item}")
    else:
        print("  (no components detected)")
    print("─" * 50)

    # Also print compact one-liner
    if inventory:
        summary = ", ".join(
            f"{cnt}x {itm}" for itm, cnt in sorted(inventory.items())
        )
        print(f"  INVENTORY: {summary}")


# ─────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────
def main():
    print("=" * 50)
    print("  CircuitMind — Part 1  (press Q to quit)")
    print(f"  Model  : {MODEL_NAME}")
    print(f"  Webcam : index {WEBCAM_INDEX}")
    print(f"  Scanning every {PROCESS_EVERY} frames")
    print("=" * 50)

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        raise SystemExit(
            f"❌  Could not open webcam (index {WEBCAM_INDEX}).\n"
            "    Try changing WEBCAM_INDEX to 1 or 2."
        )

    frame_count      = 0
    last_components  : list[dict] = []
    status_msg       = "Initialising…"
    api_in_progress  = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame grab failed — retrying…")
            time.sleep(0.1)
            continue

        frame_count += 1

        # ── Trigger Gemini every PROCESS_EVERY frames ──────────────────
        if frame_count % PROCESS_EVERY == 0:
            status_msg = "Scanning…"
            t0 = time.time()
            last_components = detect_components(frame)
            elapsed = time.time() - t0
            print_inventory(last_components, elapsed)
            status_msg = f"Last scan: {len(last_components)} component(s) @ {time.strftime('%H:%M:%S')}"

        # ── Draw detections from last scan ─────────────────────────────
        display = draw_detections(frame.copy(), last_components)

        # ── HUD overlay ────────────────────────────────────────────────
        h, w = display.shape[:2]
        cv2.rectangle(display, (0, h - 28), (w, h), (20, 20, 20), -1)
        cv2.putText(display, status_msg, (8, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 130), 1, cv2.LINE_AA)

        frames_to_next = PROCESS_EVERY - (frame_count % PROCESS_EVERY)
        counter_txt = f"Next scan in {frames_to_next} frame(s)   [Q] quit"
        cv2.putText(display, counter_txt, (w - 320, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

        cv2.imshow("CircuitMind — Live", display)

        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nCircuitMind session ended.")


if __name__ == "__main__":
    main()
