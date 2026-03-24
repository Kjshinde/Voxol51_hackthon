"""
CircuitMind — Real-Time AI Lab Supervisor for Electronics
Part 1: Live USB camera feed → Gemini Vision → FiftyOne dataset

Cameras detected:
  /dev/video0  HP Wide Vision HD Camera  640×480
  /dev/video2  IPEVO V4K                 640×480

SDK: google-genai  (new, replaces deprecated google-generativeai)

Usage:
    python circuitmind.py

Quit: press  q  in the OpenCV window.
"""

import os
import re
import sys
import json
import time
import textwrap
from pathlib import Path
from collections import defaultdict, Counter

import cv2
import numpy as np
from PIL import Image
import fiftyone as fo

# ── New google-genai SDK ──────────────────────────────────────────────────────
from google import genai
from google.genai import types

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

DATASET_NAME   = "circuitmind"
CAPTURE_DIR    = Path("./captures")
FRAME_INTERVAL = 30          # Analyse every Nth frame; show all others raw
GEMINI_MODEL   = "gemini-1.5-flash"

# Camera indices confirmed working on this VM
CAMERA_INDICES = [0, 2]

# ── Per-type bounding-box colors (BGR for OpenCV) ─────────────────────────────
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

# ── 3×3 location grid → normalised bounding box [x, y, w, h] ─────────────────
LOCATION_BOXES = {
    "top-left":      [0.00, 0.00, 0.33, 0.33],
    "top-center":    [0.33, 0.00, 0.34, 0.33],
    "top-right":     [0.67, 0.00, 0.33, 0.33],
    "center":        [0.33, 0.33, 0.34, 0.34],
    "bottom-left":   [0.00, 0.67, 0.33, 0.33],
    "bottom-center": [0.33, 0.67, 0.34, 0.33],
    "bottom-right":  [0.67, 0.67, 0.33, 0.33],
}
DEFAULT_BOX = [0.25, 0.25, 0.50, 0.50]   # fallback: centre crop

# ─────────────────────────────────────────────────────────────────────────────
# Gemini prompt
# ─────────────────────────────────────────────────────────────────────────────

GEMINI_PROMPT = textwrap.dedent("""
    You are an expert electronics engineer.
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

    If no components are visible, return:  {"components": []}
""").strip()

# ─────────────────────────────────────────────────────────────────────────────
# Camera helpers
# ─────────────────────────────────────────────────────────────────────────────

def open_camera():
    """
    Try CAMERA_INDICES in order.
    Returns (VideoCapture, index_used).
    Exits with a clear message if none work.
    """
    for idx in CAMERA_INDICES:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"[Camera] ✓ Opened /dev/video{idx}  —  resolution: {w}×{h}")
                return cap, idx
            cap.release()

    print(
        f"\n[Camera] ✗ Could not open any camera (tried indices: {CAMERA_INDICES}).\n\n"
        "  VirtualBox USB Passthrough checklist:\n"
        "    1. VM Settings → USB → USB 3.0 (xHCI) Controller enabled\n"
        "    2. Add a USB Device Filter for each camera\n"
        "    3. Devices menu → USB → tick your camera\n"
        "    4. Re-run:  ls /dev/video*\n"
    )
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# FiftyOne helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_or_create_dataset(name):
    if fo.dataset_exists(name):
        ds = fo.load_dataset(name)
        print(f"[FiftyOne] Loaded existing dataset '{name}' ({len(ds)} samples)")
    else:
        ds = fo.Dataset(name=name, persistent=True)
        print(f"[FiftyOne] Created new persistent dataset '{name}'")
    return ds

# ─────────────────────────────────────────────────────────────────────────────
# Gemini helpers  (google-genai SDK)
# ─────────────────────────────────────────────────────────────────────────────

def call_gemini(client, pil_img):
    """
    Send image + prompt to Gemini, parse response JSON.
    Returns list of component dicts.
    Raises ValueError / json.JSONDecodeError on bad output.
    """
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[GEMINI_PROMPT, pil_img],
    )
    raw = response.text

    # Strip markdown fences if Gemini adds them despite instructions
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    # Grab the first {...} block
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON object in Gemini response: {raw[:120]!r}")

    data = json.loads(m.group())
    return data.get("components", [])

# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def color_for_type(ctype):
    return TYPE_COLORS.get(ctype.lower(), DEFAULT_COLOR)


def location_to_bbox(location):
    return LOCATION_BOXES.get(location.lower().strip(), DEFAULT_BOX)


def draw_overlay(frame, detections):
    """
    Draw bounding boxes + labels + INVENTORY HUD onto frame (modifies copy).
    detections: list of fo.Detection objects.
    """
    h, w = frame.shape[:2]
    inventory = Counter()

    for det in detections:
        ctype = det.label.lower()
        value = det.get_field("value") or "unknown"
        inventory[ctype] += 1

        color = color_for_type(ctype)
        bx, by, bw, bh = det.bounding_box   # normalised

        x1, y1 = int(bx * w),        int(by * h)
        x2, y2 = int((bx+bw) * w),   int((by+bh) * h)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label_text = f"{ctype}: {value}"
        (tw, th), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1-th-6), (x1+tw+4, y1), color, -1)
        cv2.putText(
            frame, label_text, (x1+2, y1-4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 255, 255), 1, cv2.LINE_AA,
        )

    # ── INVENTORY HUD (top-left) ──────────────────────────────────────────────
    hud_lines = ["INVENTORY"] + [
        f"{cnt}x {t}" for t, cnt in sorted(inventory.items())
    ]
    y_cur = 24
    for line in hud_lines:
        cv2.putText(
            frame, line, (10, y_cur),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            (0, 255, 255), 2, cv2.LINE_AA,
        )
        y_cur += 26

    return frame


def print_inventory(frame_num, detections):
    print(f"\n=== CIRCUITMIND INVENTORY [frame {frame_num:04d}] ===")
    tally = defaultdict(list)
    for det in detections:
        tally[det.label.lower()].append(det.get_field("value") or "unknown")
    if not tally:
        print("  (no components detected)")
    for ctype, vals in sorted(tally.items()):
        if len(vals) == 1:
            print(f"  1x {vals[0]} {ctype}")
        else:
            print(f"  {len(vals)}x {ctype}  ({', '.join(vals)})")
    print("=" * 42)

# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── API key ───────────────────────────────────────────────────────────────
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("[Error] Set the GOOGLE_API_KEY environment variable first.")
        sys.exit(1)

    # ── Gemini client  (new google-genai SDK) ─────────────────────────────────
    client = genai.Client(api_key=api_key)
    print(f"[Gemini] Using model: {GEMINI_MODEL}  (google-genai SDK)")

    # ── Camera ────────────────────────────────────────────────────────────────
    cap, cam_idx = open_camera()

    # ── FiftyOne dataset ──────────────────────────────────────────────────────
    dataset = load_or_create_dataset(DATASET_NAME)

    # ── Capture directory ─────────────────────────────────────────────────────
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

    # ── State ─────────────────────────────────────────────────────────────────
    frame_counter   = 0
    # Resume numbering if previous captures exist
    capture_counter = len(list(CAPTURE_DIR.glob("frame_*.jpg")))
    last_detections = []   # keep drawing last result between analysis frames
    consecutive_fails = 0
    MAX_CAM_FAILS     = 3

    win_title = "CircuitMind — Live Component Detector"
    cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)

    print(f"\n[CircuitMind] Running on /dev/video{cam_idx} "
          f"— analysing every {FRAME_INTERVAL} frames.\n"
          f"Press  q  in the video window to quit.\n")

    while True:
        ret, frame = cap.read()

        # ── Camera failure handling ───────────────────────────────────────────
        if not ret:
            consecutive_fails += 1
            print(f"[Camera] Read failure "
                  f"({consecutive_fails}/{MAX_CAM_FAILS}) — retrying…")
            time.sleep(0.2)
            if consecutive_fails >= MAX_CAM_FAILS:
                print("[Camera] Too many failures. Exiting.")
                break
            continue
        consecutive_fails = 0
        frame_counter += 1

        display = frame.copy()

        # ── Every FRAME_INTERVAL-th frame: send to Gemini ────────────────────
        if frame_counter % FRAME_INTERVAL == 0:
            capture_counter += 1
            save_path = CAPTURE_DIR / f"frame_{capture_counter:04d}.jpg"
            cv2.imwrite(str(save_path), frame)

            detections = []

            try:
                pil_img    = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                components = call_gemini(client, pil_img)

                for comp in components:
                    ctype      = str(comp.get("type",       "other")).lower()
                    value      = str(comp.get("value",      "unknown"))
                    location   = str(comp.get("location",   "center"))
                    confidence = str(comp.get("confidence", "low"))

                    det = fo.Detection(
                        label            = ctype,
                        bounding_box     = location_to_bbox(location),
                        value            = value,
                        confidence_level = confidence,  # 'confidence' is numeric-only in FO
                    )
                    detections.append(det)

            except json.JSONDecodeError as e:
                print(f"[Gemini] ⚠ JSON parse error (frame {frame_counter}): {e}")
            except ValueError as e:
                print(f"[Gemini] ⚠ Bad response (frame {frame_counter}): {e}")
            except Exception as e:
                print(f"[Gemini] ⚠ API error (frame {frame_counter}): {e}")
            else:
                # ── Persist to FiftyOne only on clean parse ───────────────────
                sample = fo.Sample(filepath=str(save_path.resolve()))
                sample["components"] = fo.Detections(detections=detections)
                dataset.add_sample(sample)
                dataset.save()

                last_detections = detections
                print_inventory(frame_counter, detections)

        # ── Always draw last known detections on the live window ──────────────
        if last_detections:
            display = draw_overlay(display, last_detections)

        cv2.imshow(win_title, display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n[CircuitMind] Quit — shutting down cleanly.")
            break

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    print(f"[FiftyOne] Dataset '{DATASET_NAME}' now has {len(dataset)} samples.")
    print("[CircuitMind] Done.  Run  python view_dataset.py  to browse results.")


if __name__ == "__main__":
    main()
