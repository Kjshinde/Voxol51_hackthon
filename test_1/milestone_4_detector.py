"""
═══════════════════════════════════════════════════════════════════
 MILESTONE 4 — Live detection (no FiftyOne)
═══════════════════════════════════════════════════════════════════
 What this tests:
   • Camera + Gemini working together end-to-end
   • JSON parsing → fo.Detection objects
   • OpenCV overlay: bounding boxes, labels, INVENTORY HUD
   • Frame sampling (every Nth frame sent to Gemini)

 FiftyOne is intentionally NOT used here.
 This lets you verify the visual pipeline before adding DB storage.

 Controls:
   q — quit
   s — force an immediate Gemini analysis on the current frame

 Expected output (terminal):
   [Detector] 🎥 Live feed on /dev/video2 — analysing every 30 frames
   [Detector] Press  q  to quit,  s  to force analysis now
   [Gemini] Analysing frame 0030…
   [Gemini] ✓ 3 component(s) detected
   === INVENTORY [frame 0030] ===
     1x 10k resistor
     2x LED (red, green)

 Run:
   python milestone_4_detector.py
═══════════════════════════════════════════════════════════════════
"""

import os
import re
import sys
import json
import time
import config  # sets FIFTYONE_DATABASE_DIR + all shared settings

from pathlib import Path
from collections import Counter, defaultdict

import cv2
import numpy as np
from PIL import Image
from google import genai

# ── Pull settings from config ─────────────────────────────────────────────────
CAMERA_INDICES = config.CAMERA_INDICES
GEMINI_MODEL   = config.GEMINI_MODEL
GEMINI_PROMPT  = config.GEMINI_PROMPT
FRAME_INTERVAL = config.FRAME_INTERVAL
CAPTURE_DIR    = config.CAPTURE_DIR
TYPE_COLORS    = config.TYPE_COLORS
DEFAULT_COLOR  = config.DEFAULT_COLOR
LOCATION_BOXES = config.LOCATION_BOXES
DEFAULT_BOX    = config.DEFAULT_BOX


# ─────────────────────────────────────────────────────────────────────────────
# Camera
# ─────────────────────────────────────────────────────────────────────────────

def open_camera():
    """Open first working camera from CAMERA_INDICES."""
    print(f"\n[Camera] Trying indices: {CAMERA_INDICES}")
    for idx in CAMERA_INDICES:
        print(f"[Camera] → /dev/video{idx} …", end=" ", flush=True)
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"OK  ({w}×{h})")
                print(f"[Camera] ✓ Using /dev/video{idx}")
                return cap, idx
            cap.release()
        print("failed")

    print("[Camera] ✗ No camera found. Check VirtualBox Devices → USB.")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Gemini
# ─────────────────────────────────────────────────────────────────────────────

def call_gemini(client, pil_img: Image.Image, frame_num: int) -> list[dict]:
    """
    Send image to Gemini. Returns list of component dicts.
    On any error, prints a warning and returns [] so the loop continues.
    """
    print(f"[Gemini] Analysing frame {frame_num:04d}…", end=" ", flush=True)
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[GEMINI_PROMPT, pil_img],
        )
        raw = response.text

        # Strip markdown fences if Gemini adds them
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not m:
            raise ValueError(f"No JSON block found: {raw[:80]!r}")

        data       = json.loads(m.group())
        components = data.get("components", [])
        print(f"✓ {len(components)} component(s)")
        return components

    except json.JSONDecodeError as e:
        print(f"⚠ JSON error: {e}")
    except ValueError as e:
        print(f"⚠ Response error: {e}")
    except Exception as e:
        print(f"⚠ API error: {e}")
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Detection helpers  (simple dicts here — no FiftyOne yet)
# ─────────────────────────────────────────────────────────────────────────────

class SimpleDetection:
    """
    Lightweight stand-in for fo.Detection used in Milestone 4.
    Stores label, normalised bounding_box, value, confidence_level.
    """
    def __init__(self, label, bounding_box, value, confidence_level):
        self.label            = label
        self.bounding_box     = bounding_box   # [x, y, w, h] normalised
        self.value            = value
        self.confidence_level = confidence_level

    def get_field(self, name):
        return getattr(self, name, None)


def components_to_detections(components: list[dict]) -> list[SimpleDetection]:
    """Convert Gemini component dicts → SimpleDetection objects."""
    result = []
    for comp in components:
        ctype    = str(comp.get("type",       "other")).lower()
        value    = str(comp.get("value",      "unknown"))
        location = str(comp.get("location",   "center"))
        conf     = str(comp.get("confidence", "low"))
        bbox     = LOCATION_BOXES.get(location.lower().strip(), DEFAULT_BOX)
        result.append(SimpleDetection(ctype, bbox, value, conf))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Drawing
# ─────────────────────────────────────────────────────────────────────────────

def color_for_type(ctype: str):
    return TYPE_COLORS.get(ctype.lower(), DEFAULT_COLOR)


def draw_detections(frame: np.ndarray, detections: list[SimpleDetection]) -> np.ndarray:
    """Draw bounding boxes, labels, and INVENTORY HUD onto frame."""
    h, w = frame.shape[:2]
    inventory = Counter()

    for det in detections:
        ctype = det.label.lower()
        value = det.value or "unknown"
        inventory[ctype] += 1

        color       = color_for_type(ctype)
        bx, by, bw, bh = det.bounding_box

        x1, y1 = int(bx * w),        int(by * h)
        x2, y2 = int((bx+bw) * w),   int((by+bh) * h)

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label  (colored background + white text)
        label_text       = f"{ctype}: {value}"
        (tw, th), _      = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1-th-6), (x1+tw+4, y1), color, -1)
        cv2.putText(
            frame, label_text, (x1+2, y1-4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )

    # INVENTORY HUD (top-left corner, cyan text)
    hud_lines = ["INVENTORY"] + [f"{cnt}x {t}" for t, cnt in sorted(inventory.items())]
    y_cur = 24
    for line in hud_lines:
        cv2.putText(
            frame, line, (10, y_cur),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA,
        )
        y_cur += 26

    return frame


def print_inventory(frame_num: int, detections: list[SimpleDetection]):
    """Print formatted inventory to terminal."""
    print(f"\n=== INVENTORY [frame {frame_num:04d}] ===")
    tally = defaultdict(list)
    for det in detections:
        tally[det.label.lower()].append(det.value or "unknown")
    if not tally:
        print("  (no components detected)")
    for ctype, vals in sorted(tally.items()):
        if len(vals) == 1:
            print(f"  1x {vals[0]} {ctype}")
        else:
            print(f"  {len(vals)}x {ctype}  ({', '.join(vals)})")
    print("=" * 34)


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(" MILESTONE 4 — Live detector (no FiftyOne)")
    print("=" * 60)

    # ── API key ───────────────────────────────────────────────────────────────
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[Error] Set GOOGLE_API_KEY first.  export GOOGLE_API_KEY='...'")
        sys.exit(1)

    # ── Gemini client ─────────────────────────────────────────────────────────
    print(f"\n[Gemini] Initialising — model: {GEMINI_MODEL}")
    client = genai.Client(api_key=api_key)
    print("[Gemini] ✓ Ready")

    # ── Camera ────────────────────────────────────────────────────────────────
    cap, cam_idx = open_camera()

    # ── Capture dir ───────────────────────────────────────────────────────────
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    capture_counter = len(list(CAPTURE_DIR.glob("frame_*.jpg")))

    # ── State ─────────────────────────────────────────────────────────────────
    frame_counter     = 0
    last_detections   = []   # drawn on every frame until next analysis
    consecutive_fails = 0
    MAX_CAM_FAILS     = 3
    force_analysis    = False  # set True when user presses  s

    win_title = "Milestone 4 — Live Detector  (q=quit, s=analyse now)"
    cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)

    print(f"\n[Detector] 🎥 Live feed on /dev/video{cam_idx}")
    print(f"[Detector] Analysing every {FRAME_INTERVAL} frames")
    print("[Detector] Press  q  to quit,  s  to force analysis now\n")

    while True:
        ret, frame = cap.read()

        # ── Camera failure handling ───────────────────────────────────────────
        if not ret:
            consecutive_fails += 1
            print(f"[Camera] ⚠ Read failure ({consecutive_fails}/{MAX_CAM_FAILS})")
            time.sleep(0.2)
            if consecutive_fails >= MAX_CAM_FAILS:
                print("[Camera] Too many failures — exiting")
                break
            continue
        consecutive_fails = 0
        frame_counter    += 1

        display = frame.copy()

        # ── Decide whether to analyse this frame ──────────────────────────────
        should_analyse = force_analysis or (frame_counter % FRAME_INTERVAL == 0)

        if should_analyse:
            force_analysis   = False
            capture_counter += 1
            save_path = CAPTURE_DIR / f"frame_{capture_counter:04d}.jpg"
            cv2.imwrite(str(save_path), frame)
            print(f"[Detector] 📸 Saved: {save_path}")

            pil_img    = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            components = call_gemini(client, pil_img, frame_counter)
            detections = components_to_detections(components)
            last_detections = detections
            print_inventory(frame_counter, detections)

        # ── Draw last known detections on every frame ─────────────────────────
        if last_detections:
            display = draw_detections(display, last_detections)

        cv2.imshow(win_title, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("\n[Detector] Quit — shutting down")
            break
        elif key == ord("s"):
            print("[Detector] Force analysis requested")
            force_analysis = True

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[Milestone 4] Complete — {capture_counter} frame(s) captured")
    print("  → Ready for the full pipeline (python circuitmind.py)")


if __name__ == "__main__":
    main()
