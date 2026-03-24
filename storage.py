"""
storage.py — JSON-based storage layer (FiftyOne replacement)

Saves detection results to ./results/circuitmind.json
Each entry mirrors what a fo.Sample would store:
  - filepath
  - filename
  - processed_at  (ISO timestamp)
  - components    (list of detection dicts)

To swap back to FiftyOne later:
  Replace calls to storage.save_sample() with fo.Sample + dataset.add_sample()
  The data structure is identical.

No MongoDB, no AVX, no VirtualBox issues.
"""

import json
from pathlib import Path
from datetime import datetime

RESULTS_DIR  = Path("./results")
RESULTS_FILE = RESULTS_DIR / "circuitmind.json"


def _load_db() -> dict:
    """Load the JSON database from disk. Returns empty structure if not found."""
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {"dataset": "circuitmind", "samples": []}


def _save_db(db: dict) -> None:
    """Write the JSON database to disk."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(db, f, indent=2)


def save_sample(filepath: str, components: list[dict]) -> int:
    """
    Save one image's detection results.

    Args:
        filepath:   absolute or relative path to the source image
        components: list of Gemini component dicts
                    [{"type": ..., "value": ..., "location": ..., "confidence": ...}]

    Returns:
        total number of samples in the dataset after saving
    """
    db = _load_db()

    sample = {
        "filepath":     filepath,
        "filename":     Path(filepath).name,
        "processed_at": datetime.now().isoformat(),
        "components":   components,
    }

    # Overwrite if this filepath was already processed
    existing = [i for i, s in enumerate(db["samples"]) if s["filepath"] == filepath]
    if existing:
        db["samples"][existing[0]] = sample
        print("[Storage] Updated existing sample: " + Path(filepath).name)
    else:
        db["samples"].append(sample)
        print("[Storage] Saved new sample: " + Path(filepath).name)

    _save_db(db)
    return len(db["samples"])


def load_all_samples() -> list[dict]:
    """Return all saved samples."""
    return _load_db()["samples"]


def sample_count() -> int:
    """Return number of saved samples."""
    return len(_load_db()["samples"])


def print_summary() -> None:
    """Print a summary of all stored detections to terminal."""
    db = _load_db()
    samples = db["samples"]

    print("\n=== CircuitMind Dataset Summary ===")
    print("Total images processed: " + str(len(samples)))

    # Tally all components across all samples
    tally: dict[str, list[str]] = {}
    for sample in samples:
        for comp in sample.get("components", []):
            ctype = comp.get("type", "other").lower()
            value = comp.get("value", "unknown")
            if ctype not in tally:
                tally[ctype] = []
            tally[ctype].append(value)

    if tally:
        print("\nComponent totals across all images:")
        for ctype, values in sorted(tally.items(), key=lambda x: -len(x[1])):
            print("  " + str(len(values)) + "x " + ctype +
                  "  (e.g. " + values[0] + ")")
    else:
        print("No components detected yet.")

    print("\nResults file: " + str(RESULTS_FILE.resolve()))
    print("===================================")
