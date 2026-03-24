"""
view_dataset.py — Browse the CircuitMind FiftyOne dataset in the App.

Usage:
    python view_dataset.py

Press Ctrl+C to close the App and exit.
"""

import sys
import time
import fiftyone as fo

DATASET_NAME = "circuitmind"


def main() -> None:
    # ── Check the dataset exists ──────────────────────────────────────────────
    if not fo.dataset_exists(DATASET_NAME):
        print(
            f"[Error] Dataset '{DATASET_NAME}' does not exist yet.\n"
            "  Run  python circuitmind.py  first to capture some frames."
        )
        sys.exit(1)

    # ── Load and summarise ────────────────────────────────────────────────────
    dataset = fo.load_dataset(DATASET_NAME)
    n = len(dataset)
    print(f"[FiftyOne] Dataset '{DATASET_NAME}' — {n} sample(s)")

    if n == 0:
        print("[FiftyOne] No samples yet. Run the main detector first.")
        sys.exit(0)

    # ── Quick stats ───────────────────────────────────────────────────────────
    label_counts: dict[str, int] = {}
    for sample in dataset:
        dets = sample.get_field("components")
        if dets:
            for det in dets.detections:
                label_counts[det.label] = label_counts.get(det.label, 0) + 1

    if label_counts:
        print("\n  Component totals across all frames:")
        for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
            print(f"    {count:>4}x  {label}")
    print()

    # ── Launch the FiftyOne App ───────────────────────────────────────────────
    print("[FiftyOne] Launching the App — open the URL shown below in your browser.")
    print("           Press  Ctrl+C  here to close.\n")

    session = fo.launch_app(dataset)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[FiftyOne] Closing the App. Goodbye!")
        session.close()


if __name__ == "__main__":
    main()
