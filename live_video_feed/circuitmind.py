"""
═══════════════════════════════════════════════════════════════════
 circuitmind.py — Full pipeline  (run AFTER all 4 milestones pass)
═══════════════════════════════════════════════════════════════════
 Combines everything:
   Camera → Gemini Vision → FiftyOne dataset → OpenCV overlay

 Prerequisites (run in order):
   python milestone_1_camera.py    ← camera works
   python milestone_2_gemini.py    ← Gemini API works
   python milestone_3_fiftyone.py  ← FiftyOne DB works
   python milestone_4_detector.py  ← live detection works

 Controls:
   q — quit
   s — force immediate Gemini analysis on current frame

 Run:
   python circuitmind.py
═══════════════════════════════════════════════════════════════════
"""

import os
import re
import sys
import json
import time
import config   # ← MUST be first: sets FIFTYONE_DATABASE_DIR before fiftyone import

from pathlib import Path
from collections import Counter, defaultdict

import cv2
import numpy as np
from PIL import Image
from google import genai

# Import fiftyone AFTER config has set the DB path
import fiftyone as fo

# ── Pull ALL settings from config (single source of truth) ───────────────────
CAMERA_INDICES = config.CAMERA_INDICES
GEMINI_MODEL   = config.GEMINI_MODEL
GEMINI_PROMPT  = config.GEMINI_PROMPT
FRAME_INTERVAL = config.FRAME_INTERVAL
CAPTURE_DIR    = config.CAPTURE_DIR
DATASET_NAME   = config.DATASET_NAME
TYPE_COLORS    = config.TYPE_COLORS
DEFAULT_COLOR  = config.DEFAULT_COLOR
LOCATION_BOXES = config.LOCATION_BOXES
DEFAULT_BOX    = config.DEFAULT_BOX


# ─────────────────────────────────────────────────────────────────────────────
# Camera
# ─────────────────────────────────────────────────────────────────────────────

def open_camera():
    """Delegate to config.open_camera() — single source of truth for the MJPEG fix."""
    return config.open_camera()


# ─────────────────────────────────────────────────────────────────────────────
# FiftyOne
# ─────────────────────────────────────────────────────────────────────────────

def load_or_create_dataset(name: str) -> fo.Dataset:
    """Load existing persistent dataset, or create a new one."""
    if fo.dataset_exists(name):
        ds = fo.load_dataset(name)
        print(f"[FiftyOne] ✓ Loaded existing dataset '{name}' ({len(ds)} samples)")
    else:
        ds = fo.Dataset(name=name, persistent=True)
        print(f"[FiftyOne] ✓ Created new persistent dataset '{name}'")
    return ds


def build_fo_detections(components: list[dict]) -> list[fo.Detection]:
    """
    Convert Gemini component dicts → fo.Detection objects.
    Each Detection gets:
      label            = component type (string, lowercase)
      bounding_box     = normalised [x, y, w, h] from the 3×3 grid
      value            = component value/descriptor from Gemini
      confidence_level = "high" | "medium" | "low"  (string)
                         (can't use 'confidence' — that's a reserved float field in FO)
    """
    result = []
    for comp in components:
        ctype    = str(comp.get("type",       "other")).lower()
        value    = str(comp.get("value",      "unknown"))
        location = str(comp.get("location",   "center"))
        conf     = str(comp.get("confidence", "low"))
        bbox     = LOCATION_BOXES.get(location.lower().strip(), DEFAULT_BOX)

        det = fo.Detection(
            label            = ctype,
            bounding_box     = bbox,
            value            = value,
            confidence_level = conf,
        )
        result.append(det)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Gemini
# ─────────────────────────────────────────────────────────────────────────────

def call_gemini(client, pil_img: Image.Image, frame_num: int) -> list[dict]:
    """
    Send image to Gemini. Returns list of component dicts.
    Never crashes the main loop — returns [] on any error.
    """
    print(f"[Gemini] Analysing frame {frame_num:04d}…", end=" ", flush=True)
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[GEMINI_PROMPT, pil_img],
        )
        raw     = response.text
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        m       = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not m:
            raise ValueError(f"No JSON block: {raw[:80]!r}")
        data       = json.loads(m.group())
        components = data.get("components", [])
        print(f"✓ {len(components)} component(s)")
        return components
    except json.JSONDecodeError as e:
        print(f"⚠ JSON error: {e}")
    except ValueError as e:
        print(f"⚠ Response format error: {e}")
    except Exception as e:
        print(f"⚠ API error: {e}")
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Drawing
# ─────────────────────────────────────────────────────────────────────────────

def color_for_type(ctype: str):
    return TYPE_COLORS.get(ctype.lower(), DEFAULT_COLOR)


def draw_overlay(frame: np.ndarray, detections: list[fo.Detection]) -> np.ndarray:
    """Render bounding boxes, labels, and INVENTORY HUD onto frame."""
    h, w = frame.shape[:2]
    inventory = Counter()

    for det in detections:
        ctype = det.label.lower()
        value = det.get_field("value") or "unknown"
        inventory[ctype] += 1

        color       = color_for_type(ctype)
        bx, by, bw, bh = det.bounding_box

        x1, y1 = int(bx * w),        int(by * h)
        x2, y2 = int((bx+bw) * w),   int((by+bh) * h)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label_text  = f"{ctype}: {value}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1-th-6), (x1+tw+4, y1), color, -1)
        cv2.putText(
            frame, label_text, (x1+2, y1-4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )

    # INVENTORY HUD
    hud_lines = ["INVENTORY"] + [f"{cnt}x {t}" for t, cnt in sorted(inventory.items())]
    y_cur = 24
    for line in hud_lines:
        cv2.putText(
            frame, line, (10, y_cur),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA,
        )
        y_cur += 26

    return frame


def print_inventory(frame_num: int, detections: list[fo.Detection]):
    """Print formatted component inventory to terminal."""
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
    print("=" * 60)
    print(" CircuitMind — Full Pipeline")
    print("=" * 60)

    # ── API key ───────────────────────────────────────────────────────────────
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[Error] Set GOOGLE_API_KEY first.  export GOOGLE_API_KEY='...'")
        sys.exit(1)

    # ── Gemini ────────────────────────────────────────────────────────────────
    print(f"\n[Gemini] Initialising — model: {GEMINI_MODEL}")
    client = genai.Client(api_key=api_key)
    print("[Gemini] ✓ Ready")

    # ── Camera ────────────────────────────────────────────────────────────────
    cap, cam_idx = open_camera()

    # ── FiftyOne ──────────────────────────────────────────────────────────────
    db_dir = os.environ.get("FIFTYONE_DATABASE_DIR", "default")
    print(f"\n[FiftyOne] DB directory: {db_dir}")
    dataset = load_or_create_dataset(DATASET_NAME)

    # ── Capture dir ───────────────────────────────────────────────────────────
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    # Resume numbering if previous captures already exist
    capture_counter = len(list(CAPTURE_DIR.glob("frame_*.jpg")))
    print(f"[Captures] Directory: {CAPTURE_DIR}  (existing: {capture_counter})")

    # ── State ─────────────────────────────────────────────────────────────────
    frame_counter     = 0
    last_detections   = []   # fo.Detection list; drawn on every frame
    consecutive_fails = 0
    MAX_CAM_FAILS     = 3
    force_analysis    = False

    win_title = "CircuitMind — Live Component Detector  (q=quit, s=analyse now)"
    cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)

    print(f"\n[CircuitMind] 🚀 Running on /dev/video{cam_idx}")
    print(f"[CircuitMind] Analysing every {FRAME_INTERVAL} frames")
    print("[CircuitMind] Press  q  to quit,  s  to force analysis now\n")

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
            print(f"[CircuitMind] 📸 Saved: {save_path}")

            # ── Gemini → parse → fo.Detections ───────────────────────────────
            pil_img    = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            components = call_gemini(client, pil_img, frame_counter)
            detections = build_fo_detections(components)

            # ── Save to FiftyOne (only when Gemini returned valid data) ───────
            try:
                sample = fo.Sample(filepath=str(save_path.resolve()))
                sample["components"] = fo.Detections(detections=detections)
                dataset.add_sample(sample)
                dataset.save()
                total = len(dataset)
                print(f"[FiftyOne] ✓ Sample saved — dataset total: {total}")
            except Exception as e:
                print(f"[FiftyOne] ⚠ Save failed: {e}")

            last_detections = detections
            print_inventory(frame_counter, detections)

        # ── Draw last known detections on every frame ─────────────────────────
        if last_detections:
            display = draw_overlay(display, last_detections)

        cv2.imshow(win_title, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("\n[CircuitMind] Quit — shutting down cleanly")
            break
        elif key == ord("s"):
            print("[CircuitMind] Force analysis on next iteration")
            force_analysis = True

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[FiftyOne] Dataset '{DATASET_NAME}' now has {len(dataset)} samples")
    print("[CircuitMind] Done.  Run  python view_dataset.py  to browse results.")


if __name__ == "__main__":
    main()
