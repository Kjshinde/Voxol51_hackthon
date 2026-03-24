"""
═══════════════════════════════════════════════════════════════════
 view_dataset.py — Browse CircuitMind results in the FiftyOne App
═══════════════════════════════════════════════════════════════════
 Run AFTER circuitmind.py has captured some frames.

 What it does:
   • Loads the 'circuitmind' FiftyOne dataset
   • Prints a component tally across all frames
   • Launches the FiftyOne App in your browser
   • Stays open until Ctrl+C

 Run:
   python view_dataset.py
═══════════════════════════════════════════════════════════════════
"""

import sys
import time
import config   # ← sets FIFTYONE_DATABASE_DIR before fiftyone import

import fiftyone as fo

DATASET_NAME = config.DATASET_NAME


def main():
    print("=" * 60)
    print(" FiftyOne Dataset Viewer")
    print("=" * 60)

    # ── Check dataset exists ──────────────────────────────────────────────────
    import os
    db_dir = os.environ.get("FIFTYONE_DATABASE_DIR", "NOT SET")
    print(f"\n[FiftyOne] DB directory: {db_dir}")

    if not fo.dataset_exists(DATASET_NAME):
        print(
            f"\n[Error] Dataset '{DATASET_NAME}' does not exist.\n"
            "  Run  python circuitmind.py  first to capture frames."
        )
        sys.exit(1)

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"\n[FiftyOne] Loading dataset '{DATASET_NAME}'…")
    dataset = fo.load_dataset(DATASET_NAME)
    n       = len(dataset)
    print(f"[FiftyOne] ✓ {n} sample(s) found")

    if n == 0:
        print("[FiftyOne] No samples yet — run circuitmind.py first")
        sys.exit(0)

    # ── Component tally ───────────────────────────────────────────────────────
    print("\n[FiftyOne] Component totals across all frames:")
    label_counts: dict[str, int] = {}
    value_examples: dict[str, str] = {}

    for sample in dataset:
        dets = sample.get_field("components")
        if not dets:
            continue
        for det in dets.detections:
            lbl = det.label
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
            if lbl not in value_examples:
                value_examples[lbl] = det.get_field("value") or "unknown"

    if label_counts:
        for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
            example = value_examples.get(label, "")
            print(f"  {count:>4}x  {label:<15}  e.g. {example}")
    else:
        print("  (no detections found in any sample)")

    # ── Launch App ────────────────────────────────────────────────────────────
    print("\n[FiftyOne] Launching App…")
    print("[FiftyOne] → Open the URL shown below in your browser")
    print("[FiftyOne] → Press Ctrl+C here to close\n")

    session = fo.launch_app(dataset)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[FiftyOne] Closing app — goodbye!")
        session.close()


if __name__ == "__main__":
    main()
