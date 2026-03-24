"""
view_results.py — Browse CircuitMind detection results
(replaces view_dataset.py / FiftyOne App)

Shows all processed images with their detected components.
Opens each annotated image in a window for visual review.

Run:
  python view_results.py
"""

import sys
import cv2
import config
import storage
from pathlib import Path
from collections import Counter

ANNOTATED_DIR  = config.PICTURES_DIR / "annotated"
TYPE_COLORS    = config.TYPE_COLORS
DEFAULT_COLOR  = config.DEFAULT_COLOR
LOCATION_BOXES = config.LOCATION_BOXES
DEFAULT_BOX    = config.DEFAULT_BOX


def main():
    print("=" * 60)
    print(" CircuitMind — Results Viewer")
    print("=" * 60)

    # ── Load results ──────────────────────────────────────────────────────────
    samples = storage.load_all_samples()

    if not samples:
        print("\n[Viewer] No results found.")
        print("  Run  python circuitmind.py  first to process images.")
        sys.exit(0)

    # ── Print full summary ────────────────────────────────────────────────────
    storage.print_summary()

    # ── Per-image detail ──────────────────────────────────────────────────────
    print("\nPer-image breakdown:")
    print("-" * 44)
    for i, sample in enumerate(samples, 1):
        fname  = sample["filename"]
        comps  = sample.get("components", [])
        ts     = sample.get("processed_at", "unknown")[:19]  # trim microseconds
        print("\n[" + str(i) + "] " + fname + "  (processed: " + ts + ")")
        if comps:
            for comp in comps:
                print("    " + comp.get("type","?") +
                      " — " + comp.get("value","?") +
                      " @ " + comp.get("location","?") +
                      "  [" + comp.get("confidence","?") + "]")
        else:
            print("    (no components detected)")

    # ── Visual review of annotated images ─────────────────────────────────────
    print("\n" + "-" * 44)
    annotated_files = list(ANNOTATED_DIR.glob("*")) if ANNOTATED_DIR.exists() else []

    if not annotated_files:
        print("[Viewer] No annotated images found in " + str(ANNOTATED_DIR))
        print("  Run  python circuitmind.py  to generate them.")
        return

    print("[Viewer] Found " + str(len(annotated_files)) + " annotated image(s)")
    print("[Viewer] Opening for visual review (any key = next, q = quit)...")

    win_title = "CircuitMind Results  (any key = next,  q = quit)"
    cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)

    for img_path in sorted(annotated_files):
        frame = cv2.imread(str(img_path))
        if frame is None:
            print("[Viewer] Could not load: " + img_path.name)
            continue

        h, w = frame.shape[:2]
        cv2.resizeWindow(win_title, min(w, 1000), min(h, 750))
        cv2.imshow(win_title, frame)
        print("[Viewer] Showing: " + img_path.name)

        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            print("[Viewer] Quit")
            break

    cv2.destroyAllWindows()
    print("\n[Viewer] Done.")


if __name__ == "__main__":
    main()
