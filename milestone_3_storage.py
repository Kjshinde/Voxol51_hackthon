"""
═══════════════════════════════════════════════════════════════════
 MILESTONE 3 — JSON storage check  (replaces FiftyOne/MongoDB)
═══════════════════════════════════════════════════════════════════
 What this tests:
   • storage.py can create the results/ folder
   • Can write a sample with detections to JSON
   • Can read it back and verify the data is intact
   • Overwrites correctly if same file is processed twice

 Expected output:
   [Storage] Saved new sample: dummy_test.jpg
   [Storage] Read back 1 sample(s)
   [Storage] Sample: dummy_test.jpg
   [Storage]   1. resistor — 10k @ center [high]
   [Storage] Overwrite test...
   [Storage] Updated existing sample: dummy_test.jpg
   [Storage] Still 1 sample after overwrite — correct
   [Storage] Cleaned up test data
   [Milestone 3] PASSED ✓

 Run:
   python milestone_3_storage.py
═══════════════════════════════════════════════════════════════════
"""

import sys
import config   # sets env vars
import storage
from pathlib import Path

TEST_FILEPATH = "pictures/dummy_test.jpg"


def main():
    print("=" * 60)
    print(" MILESTONE 3 — JSON storage check")
    print("=" * 60)
    print()

    # ── 1. Write a sample ────────────────────────────────────────────────────
    print("[Storage] Writing test sample...")
    test_components = [
        {"type": "resistor",  "value": "10k",   "location": "center",      "confidence": "high"},
        {"type": "capacitor", "value": "100uF",  "location": "bottom-left", "confidence": "medium"},
        {"type": "ic",        "value": "NE555",  "location": "top-right",   "confidence": "high"},
    ]

    count = storage.save_sample(TEST_FILEPATH, test_components)
    print("[Storage] Total samples after write: " + str(count))

    # ── 2. Read back and verify ───────────────────────────────────────────────
    print("\n[Storage] Reading back all samples...")
    samples = storage.load_all_samples()
    print("[Storage] Read back " + str(len(samples)) + " sample(s)")

    # Find our test sample
    found = [s for s in samples if s["filepath"] == TEST_FILEPATH]
    if not found:
        print("[Storage] ERROR: test sample not found after write!")
        sys.exit(1)

    sample = found[0]
    print("[Storage] Sample: " + sample["filename"])
    for i, comp in enumerate(sample["components"], 1):
        print("[Storage]   " + str(i) + ". " +
              comp["type"] + " — " + comp["value"] +
              " @ " + comp["location"] +
              " [" + comp["confidence"] + "]")

    # Verify data integrity
    assert len(sample["components"]) == 3, "Expected 3 components"
    assert sample["components"][0]["type"] == "resistor", "First component should be resistor"
    assert sample["components"][2]["value"] == "NE555", "Third component value should be NE555"
    print("[Storage] Data integrity check passed")

    # ── 3. Overwrite test ─────────────────────────────────────────────────────
    print("\n[Storage] Overwrite test (same filepath, different data)...")
    storage.save_sample(TEST_FILEPATH, [
        {"type": "led", "value": "red", "location": "center", "confidence": "high"}
    ])
    samples_after = storage.load_all_samples()
    still_one = len([s for s in samples_after if s["filepath"] == TEST_FILEPATH]) == 1
    if still_one:
        print("[Storage] Still 1 sample after overwrite — correct (no duplicates)")
    else:
        print("[Storage] ERROR: duplicate samples created!")
        sys.exit(1)

    # ── 4. Cleanup ────────────────────────────────────────────────────────────
    print("\n[Storage] Cleaning up test data...")
    # Remove just the test entry, keep any real data
    db_path = storage.RESULTS_FILE
    if db_path.exists():
        import json
        with open(db_path) as f:
            db = json.load(f)
        db["samples"] = [s for s in db["samples"] if s["filepath"] != TEST_FILEPATH]
        with open(db_path, "w") as f:
            json.dump(db, f, indent=2)
    print("[Storage] Test data removed")

    # ── Result ────────────────────────────────────────────────────────────────
    print()
    print("[Milestone 3] PASSED ✓")
    print("  JSON storage is working correctly")
    print("  Results will be saved to: " + str(storage.RESULTS_FILE.resolve()))
    print()
    print("  -> Ready for Milestone 4  (python milestone_4_detector.py)")


if __name__ == "__main__":
    main()
