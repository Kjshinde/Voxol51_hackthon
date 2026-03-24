"""
═══════════════════════════════════════════════════════════════════
 circuitmind.py — Full pipeline  (run AFTER all milestones pass)
═══════════════════════════════════════════════════════════════════
 Processes every image in pictures/ through:
   Gemini Vision -> FiftyOne dataset -> Annotated output images

 Prerequisites:
   python milestone_1_images.py    <- images load OK
   python milestone_2_gemini.py    <- Gemini API works
   python milestone_3_fiftyone.py  <- FiftyOne DB works
   python milestone_4_detector.py  <- detection + drawing works

 Output:
   pictures/annotated/  <- images with bounding boxes drawn
   FiftyOne dataset named "circuitmind"  <- browse with view_dataset.py

 Run:
   python circuitmind.py
═══════════════════════════════════════════════════════════════════
"""

import os
import re
import sys
import json
import config   # MUST be first: sets FIFTYONE_DATABASE_DIR before fiftyone import

from pathlib import Path
from collections import Counter, defaultdict

import cv2
from PIL import Image
from google import genai

# Import fiftyone AFTER config has set the DB path
import fiftyone as fo

# ── Pull all settings from config ─────────────────────────────────────────────
GEMINI_MODEL   = config.GEMINI_MODEL
GEMINI_PROMPT  = config.GEMINI_PROMPT
DATASET_NAME   = config.DATASET_NAME
TYPE_COLORS    = config.TYPE_COLORS
DEFAULT_COLOR  = config.DEFAULT_COLOR
LOCATION_BOXES = config.LOCATION_BOXES
DEFAULT_BOX    = config.DEFAULT_BOX
ANNOTATED_DIR  = config.PICTURES_DIR / "annotated"


# ─────────────────────────────────────────────────────────────────────────────
# FiftyOne
# ─────────────────────────────────────────────────────────────────────────────

def load_or_create_dataset(name):
    """Load existing dataset or create a fresh persistent one."""
    if fo.dataset_exists(name):
        ds = fo.load_dataset(name)
        print("[FiftyOne] Loaded existing dataset '" + name + "' (" + str(len(ds)) + " samples)")
    else:
        ds = fo.Dataset(name=name, persistent=True)
        print("[FiftyOne] Created new persistent dataset '" + name + "'")
    return ds


def build_fo_detections(components):
    """
    Convert Gemini component dicts -> fo.Detection objects.
    label            = component type (lowercase string)
    bounding_box     = normalised [x, y, w, h] from the 3x3 grid
    value            = component value/descriptor
    confidence_level = "high"|"medium"|"low"  (string)
                       NOTE: 'confidence' is a reserved float field in FiftyOne,
                       so we use 'confidence_level' for Gemini's string value.
    """
    result = []
    for comp in components:
        ctype    = str(comp.get("type",       "other")).lower()
        value    = str(comp.get("value",      "unknown"))
        location = str(comp.get("location",   "center"))
        conf     = str(comp.get("confidence", "low"))
        bbox     = LOCATION_BOXES.get(location.lower().strip(), DEFAULT_BOX)
        result.append(fo.Detection(
            label            = ctype,
            bounding_box     = bbox,
            value            = value,
            confidence_level = conf,
        ))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Gemini
# ─────────────────────────────────────────────────────────────────────────────

def call_gemini(client, pil_img, img_name):
    """Send image to Gemini. Returns list of component dicts, or [] on error."""
    print("[Gemini] Analysing " + img_name + "...", end="  ", flush=True)
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[GEMINI_PROMPT, pil_img],
        )
        raw     = response.text
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        m       = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not m:
            raise ValueError("No JSON block: " + raw[:80])
        data       = json.loads(m.group())
        components = data.get("components", [])
        print(str(len(components)) + " component(s)")
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


def draw_overlay(frame, components):
    """Draw bounding boxes, labels, and INVENTORY HUD onto frame."""
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

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label       = ctype + ": " + value + " [" + conf + "]"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1-th-6), (x1+tw+4, y1), color, -1)
        cv2.putText(
            frame, label, (x1+2, y1-4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )

    # INVENTORY HUD
    hud_lines = ["INVENTORY"] + [str(cnt) + "x " + t for t, cnt in sorted(inventory.items())]
    y_cur = 28
    for line in hud_lines:
        cv2.putText(
            frame, line, (10, y_cur),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA,
        )
        y_cur += 30
    return frame


def print_inventory(img_name, components):
    """Print formatted inventory to terminal."""
    print()
    print("=== CIRCUITMIND INVENTORY [" + img_name + "] ===")
    tally = defaultdict(list)
    for comp in components:
        tally[comp.get("type", "other").lower()].append(
            comp.get("value", "unknown")
        )
    if not tally:
        print("  (no components detected)")
    for ctype, vals in sorted(tally.items()):
        if len(vals) == 1:
            print("  1x " + ctype + "  (" + vals[0] + ")")
        else:
            print("  " + str(len(vals)) + "x " + ctype + "  (" + ", ".join(vals) + ")")
    print("=" * 44)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(" CircuitMind — Full Pipeline")
    print("=" * 60)
    print()

    # ── API key ───────────────────────────────────────────────────────────────
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[Error] Set GOOGLE_API_KEY first:  export GOOGLE_API_KEY='...'")
        sys.exit(1)

    # ── Gemini ────────────────────────────────────────────────────────────────
    print("[Gemini] Initialising — model: " + GEMINI_MODEL)
    client = genai.Client(api_key=api_key)
    print("[Gemini] Ready")
    print()

    # ── FiftyOne ──────────────────────────────────────────────────────────────
    db_dir = os.environ.get("FIFTYONE_DATABASE_DIR", "default")
    print("[FiftyOne] DB directory: " + db_dir)
    dataset = load_or_create_dataset(DATASET_NAME)
    print()

    # ── Images ────────────────────────────────────────────────────────────────
    image_paths = config.get_image_paths()
    total       = len(image_paths)
    print()

    # ── Output folder ─────────────────────────────────────────────────────────
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
    print("[Output] Annotated images -> " + str(ANNOTATED_DIR))
    print()

    # ── Process each image ────────────────────────────────────────────────────
    for i, img_path in enumerate(image_paths, 1):
        print("[CircuitMind] [" + str(i) + "/" + str(total) + "] " + img_path.name)
        print("-" * 44)

        # Load image
        frame = cv2.imread(str(img_path))
        if frame is None:
            print("[CircuitMind] Could not load image — skipping")
            continue
        pil_img = Image.open(img_path).convert("RGB")

        # Gemini detection
        components = call_gemini(client, pil_img, img_path.name)

        # Build fo.Detections
        fo_detections = build_fo_detections(components)

        # Save to FiftyOne
        try:
            sample = fo.Sample(filepath=str(img_path.resolve()))
            sample["components"] = fo.Detections(detections=fo_detections)
            dataset.add_sample(sample)
            dataset.save()
            print("[FiftyOne] Sample saved — dataset total: " + str(len(dataset)))
        except Exception as e:
            print("[FiftyOne] Save failed: " + str(e))

        # Draw and save annotated image
        annotated = draw_overlay(frame.copy(), components)
        out_path  = ANNOTATED_DIR / img_path.name
        cv2.imwrite(str(out_path), annotated)
        print("[Output] Annotated saved: " + str(out_path))

        # Print inventory
        print_inventory(img_path.name, components)
        print()

    # ── Done ──────────────────────────────────────────────────────────────────
    print("[CircuitMind] All images processed")
    print("[FiftyOne] Dataset '" + DATASET_NAME + "' now has " + str(len(dataset)) + " samples")
    print()
    print("Next steps:")
    print("  View annotated images:  ls pictures/annotated/")
    print("  Browse in FiftyOne App: python view_dataset.py")


if __name__ == "__main__":
    main()
