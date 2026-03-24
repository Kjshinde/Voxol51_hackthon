"""
═══════════════════════════════════════════════════════════════════
 MILESTONE 3 — FiftyOne database check
═══════════════════════════════════════════════════════════════════
 What this tests:
   • FiftyOne can start its embedded MongoDB (in /tmp, not on
     the VirtualBox virtual disk — that's what broke earlier)
   • We can create a persistent dataset
   • We can add a sample with a Detection annotation
   • We can reload the dataset and read back the sample

 Expected output:
   [FiftyOne] DB directory: /tmp/fiftyone_db
   [FiftyOne] ✓ Database connection established
   [FiftyOne] Creating test dataset 'circuitmind_test'…
   [FiftyOne] ✓ Dataset created
   [FiftyOne] Adding a dummy sample with one Detection…
   [FiftyOne] ✓ Sample added — dataset now has 1 sample(s)
   [FiftyOne] Reloading dataset from scratch to verify persistence…
   [FiftyOne] ✓ Reload confirmed — 1 sample(s) found
   [FiftyOne] Cleaning up test dataset…
   [FiftyOne] ✓ Test dataset deleted
   [Milestone 3] PASSED ✓

 Run:
   python milestone_3_fiftyone.py
═══════════════════════════════════════════════════════════════════
"""

import sys
import config   # ← sets FIFTYONE_DATABASE_DIR before fiftyone imports

# Import fiftyone AFTER config has set the env var
import fiftyone as fo
from pathlib import Path

PICTURES_DIR = config.PICTURES_DIR
TEST_DATASET = "circuitmind_test"   # temporary dataset, deleted at end


def main():
    print("=" * 60)
    print(" MILESTONE 3 — FiftyOne database check")
    print("=" * 60)

    # ── 1. Confirm DB path ────────────────────────────────────────────────────
    import os
    db_dir = os.environ.get("FIFTYONE_DATABASE_DIR", "NOT SET")
    print(f"\n[FiftyOne] DB directory: {db_dir}")
    print("[FiftyOne] (must be /tmp/fiftyone_db to avoid VirtualBox lock issue)")

    # ── 2. Test DB connection ─────────────────────────────────────────────────
    print("\n[FiftyOne] Testing database connection…")
    try:
        # Listing datasets triggers a DB connection; if MongoDB fails it throws here
        existing = fo.list_datasets()
        print(f"[FiftyOne] ✓ Database connection established")
        print(f"[FiftyOne] Existing datasets on this machine: {existing or '(none)'}")
    except Exception as e:
        print(f"[FiftyOne] ✗ Database connection failed: {e}")
        print(
            "\n  Fix: make sure FIFTYONE_DATABASE_DIR points to a real local path.\n"
            "  config.py sets it to /tmp/fiftyone_db — check that import config\n"
            "  happens before import fiftyone in every script.\n"
        )
        sys.exit(1)

    # ── 3. Clean up any leftover test dataset from a previous failed run ──────
    if fo.dataset_exists(TEST_DATASET):
        print(f"[FiftyOne] Found leftover '{TEST_DATASET}' — deleting it first")
        fo.delete_dataset(TEST_DATASET)

    # ── 4. Create dataset ─────────────────────────────────────────────────────
    print(f"\n[FiftyOne] Creating test dataset '{TEST_DATASET}'…")
    try:
        ds = fo.Dataset(name=TEST_DATASET, persistent=True)
        print(f"[FiftyOne] ✓ Dataset created")
    except Exception as e:
        print(f"[FiftyOne] ✗ Failed to create dataset: {e}")
        sys.exit(1)

    # ── 5. Create a dummy image file for the sample ───────────────────────────
    # FiftyOne samples need a real filepath that exists on disk.
    PICTURES_DIR.mkdir(parents=True, exist_ok=True)
    dummy_img = PICTURES_DIR / "dummy_sample.jpg"

    # Write a tiny 1×1 black JPEG using raw bytes (no extra lib needed)
    # This is just for the test — real samples use actual camera frames.
    dummy_img.write_bytes(
        b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'
        b'\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t'
        b'\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a'
        b'\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\x1e'
        b'\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4'
        b'\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00'
        b'\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b'
        b'\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05'
        b'\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06'
        b'\x13Qa\x07"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br'
        b'\x82\t\n\x16\x17\x18\x19\x1a%&\'()*456789:CDEFGHIJ'
        b'STUVWXYZ\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xf5\x00\x00\x1f'
        b'\xff\xd9'
    )
    print(f"[FiftyOne] Dummy image created: {dummy_img}")

    # ── 6. Add a sample with a Detection ─────────────────────────────────────
    print("\n[FiftyOne] Adding a dummy sample with one Detection…")
    try:
        det = fo.Detection(
            label            = "resistor",
            bounding_box     = [0.33, 0.33, 0.34, 0.34],   # normalised [x,y,w,h]
            value            = "10k",
            confidence_level = "high",
        )
        sample = fo.Sample(filepath=str(dummy_img.resolve()))
        sample["components"] = fo.Detections(detections=[det])
        ds.add_sample(sample)
        ds.save()
        print(f"[FiftyOne] ✓ Sample added — dataset now has {len(ds)} sample(s)")
    except Exception as e:
        print(f"[FiftyOne] ✗ Failed to add sample: {e}")
        sys.exit(1)

    # ── 7. Reload and verify ──────────────────────────────────────────────────
    print("\n[FiftyOne] Reloading dataset from scratch to verify persistence…")
    try:
        del ds
        ds2 = fo.load_dataset(TEST_DATASET)
        count = len(ds2)
        print(f"[FiftyOne] ✓ Reload confirmed — {count} sample(s) found")

        # Inspect the stored detection
        sample = ds2.first()
        dets   = sample.get_field("components")
        if dets and dets.detections:
            d = dets.detections[0]
            print(f"[FiftyOne] Stored detection: label={d.label!r}  "
                  f"bbox={d.bounding_box}  value={d.get_field('value')!r}")
    except Exception as e:
        print(f"[FiftyOne] ✗ Reload failed: {e}")
        sys.exit(1)

    # ── 8. Cleanup ────────────────────────────────────────────────────────────
    print(f"\n[FiftyOne] Cleaning up test dataset '{TEST_DATASET}'…")
    fo.delete_dataset(TEST_DATASET)
    dummy_img.unlink(missing_ok=True)
    print(f"[FiftyOne] ✓ Test dataset deleted")

    # ── Result ────────────────────────────────────────────────────────────────
    print()
    print("[Milestone 3] PASSED ✓")
    print("  FiftyOne DB is working and Detection round-trip is correct.")
    print("\n  → Ready for Milestone 4 (python milestone_4_detector.py)")


if __name__ == "__main__":
    main()
