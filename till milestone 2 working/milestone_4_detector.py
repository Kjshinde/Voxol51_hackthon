"""
═══════════════════════════════════════════════════════════════════
 MILESTONE 4 — Picture detector  (no FiftyOne)
═══════════════════════════════════════════════════════════════════
 What this tests:
   • Processes every image in pictures/ through Gemini
   • Draws bounding boxes + labels + INVENTORY HUD with OpenCV
   • Saves annotated results to pictures/annotated/
   • No FiftyOne involved yet

 Controls (while viewing annotated image):
   any key — next image
   q       — quit early

 Expected output:
   [Detector] Processing 2 image(s)...
   [Detector] [1/2] board1.jpg
   [Gemini] Sending to Gemini... 3 component(s) detected
   [Detector] Annotated saved: pictures/annotated/board1.jpg
   [Detector] Inventory:
     2x resistor  (10k, 4.7k)
     1x ic  (NE555)
   [Milestone 4] PASSED

 Run:
   python milestone_4_detector.py
═══════════════════════════════════════════════════════════════════
"""

import os
import re
import sys
import json
import config   # sets FIFTYONE_DATABASE_DIR + all shared settings

from pathlib import Path
from collections import Counter, defaultdict

import cv2
import numpy as np
from PIL import Image
from google import genai

GEMINI_MODEL   = config.GEMINI_MODEL
GEMINI_PROMPT  = config.GEMINI_PROMPT
TYPE_COLORS    = config.TYPE_COLORS
DEFAULT_COLOR  = config.DEFAULT_COLOR
LOCATION_BOXES = config.LOCATION_BOXES
DEFAULT_BOX    = config.DEFAULT_BOX

ANNOTATED_DIR  = config.PICTURES_DIR / "annotated"


# ─────────────────────────────────────────────────────────────────────────────
# Gemini
# ─────────────────────────────────────────────────────────────────────────────

def call_gemini(client, pil_img, img_name):
    """Send image to Gemini. Returns list of component dicts, or [] on error."""
    print("[Gemini] Sending " + img_name + " to Gemini...", end="  ", flush=True)
    try:
        response   = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[GEMINI_PROMPT, pil_img],
        )
        raw     = response.text
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        m       = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not m:
            raise ValueError("No JSON block found")
        data       = json.loads(m.group())
        components = data.get("components", [])
        print(str(len(components)) + " component(s) detected")
        return components
    except json.JSONDecodeError as e:
        print("JSON error: " + str(e))
    except ValueError as e:
        print("Response error: " + str(e))
    except Exception as e:
        print("API error: " + str(e))
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Drawing
# ─────────────────────────────────────────────────────────────────────────────

def color_for_type(ctype):
    return TYPE_COLORS.get(ctype.lower(), DEFAULT_COLOR)


def draw_detections(frame, components):
    """
    Draw bounding boxes, labels, and INVENTORY HUD on frame.
    components: list of dicts from Gemini (type, value, location, confidence)
    Returns annotated frame.
    """
    h, w      = frame.shape[:2]
    inventory = Counter()

    for comp in components:
        ctype    = str(comp.get("type",     "other")).lower()
        value    = str(comp.get("value",    "unknown"))
        location = str(comp.get("location", "center"))
        conf     = str(comp.get("confidence", "low"))
        inventory[ctype] += 1

        color       = color_for_type(ctype)
        bx, by, bw, bh = LOCATION_BOXES.get(location.lower().strip(), DEFAULT_BOX)

        x1, y1 = int(bx * w),        int(by * h)
        x2, y2 = int((bx+bw) * w),   int((by+bh) * h)

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label with colored background
        label       = ctype + ": " + value + " [" + conf + "]"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1-th-6), (x1+tw+4, y1), color, -1)
        cv2.putText(
            frame, label, (x1+2, y1-4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )

    # INVENTORY HUD (top-left corner)
    hud_lines = ["INVENTORY"] + [str(cnt) + "x " + t for t, cnt in sorted(inventory.items())]
    y_cur = 28
    for line in hud_lines:
        cv2.putText(
            frame, line, (10, y_cur),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA,
        )
        y_cur += 30

    return frame


def print_inventory(components):
    """Print component inventory to terminal."""
    tally = defaultdict(list)
    for comp in components:
        tally[comp.get("type", "other").lower()].append(
            comp.get("value", "unknown")
        )
    if not tally:
        print("  (no components detected)")
        return
    for ctype, vals in sorted(tally.items()):
        if len(vals) == 1:
            print("  1x " + ctype + "  (" + vals[0] + ")")
        else:
            print("  " + str(len(vals)) + "x " + ctype + "  (" + ", ".join(vals) + ")")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(" MILESTONE 4 — Picture detector  (no FiftyOne)")
    print("=" * 60)
    print()

    # ── API key ───────────────────────────────────────────────────────────────
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[Error] Set GOOGLE_API_KEY first:  export GOOGLE_API_KEY='...'")
        sys.exit(1)

    # ── Gemini client ─────────────────────────────────────────────────────────
    print("[Gemini] Initialising — model: " + GEMINI_MODEL)
    client = genai.Client(api_key=api_key)
    print("[Gemini] Ready")
    print()

    # ── Get images ────────────────────────────────────────────────────────────
    image_paths = config.get_image_paths()
    total       = len(image_paths)
    print()

    # ── Output folder ─────────────────────────────────────────────────────────
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
    print("[Detector] Annotated images will be saved to: " + str(ANNOTATED_DIR))
    print()

    # ── Process each image ────────────────────────────────────────────────────
    win_title = "Milestone 4 — Detections  (any key = next,  q = quit)"
    cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)

    print("[Detector] Processing " + str(total) + " image(s)...")
    print()

    for i, img_path in enumerate(image_paths, 1):
        print("[Detector] [" + str(i) + "/" + str(total) + "] " + img_path.name)
        print("-" * 40)

        # Load with OpenCV for drawing
        frame = cv2.imread(str(img_path))
        if frame is None:
            print("[Detector] Could not load image — skipping")
            continue

        # Load with PIL for Gemini (needs RGB)
        pil_img = Image.open(img_path).convert("RGB")

        # Call Gemini
        components = call_gemini(client, pil_img, img_path.name)

        # Print inventory
        print("[Detector] Inventory:")
        print_inventory(components)

        # Draw on frame
        annotated = draw_detections(frame.copy(), components)

        # Save annotated image
        out_path = ANNOTATED_DIR / img_path.name
        cv2.imwrite(str(out_path), annotated)
        print("[Detector] Annotated saved: " + str(out_path))
        print()

        # Show in window
        h, w = annotated.shape[:2]
        cv2.resizeWindow(win_title, min(w, 1000), min(h, 750))
        cv2.imshow(win_title, annotated)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            print("[Detector] Quit early")
            break

    cv2.destroyAllWindows()

    print("[Milestone 4] PASSED")
    print()
    print("  -> Ready for full pipeline  (python circuitmind.py)")


if __name__ == "__main__":
    main()
