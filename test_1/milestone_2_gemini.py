"""
═══════════════════════════════════════════════════════════════════
 MILESTONE 2 — Gemini API check
═══════════════════════════════════════════════════════════════════
 What this tests:
   • GOOGLE_API_KEY is set and valid
   • google-genai SDK is installed correctly
   • Gemini can receive an image and return component JSON
   • Our JSON parser handles the response without crashing

 Prerequisites:
   • Run Milestone 1 first (saves captures/test_frame.jpg)
   • OR set a custom image path via:  IMAGE_PATH=./my_image.jpg python milestone_2_gemini.py

 Expected output:
   [Gemini] ✓ Client initialised — model: gemini-1.5-flash
   [Gemini] Sending image to Gemini…
   [Gemini] ✓ Response received (312 chars)
   [Gemini] Raw response:
     {"components": [ ... ]}
   [Gemini] Parsed 3 component(s):
     1. IC — NE555 @ center  [confidence: high]
     2. resistor — 10k @ top-left  [confidence: high]
     3. capacitor — 100uF @ bottom-right  [confidence: medium]
   [Milestone 2] PASSED ✓

 Run:
   python milestone_2_gemini.py
═══════════════════════════════════════════════════════════════════
"""

import os
import re
import sys
import json
import config  # loads DB env var + all shared settings

from pathlib import Path
from PIL import Image
from google import genai

# ── Pull settings from config ─────────────────────────────────────────────────
GEMINI_MODEL  = config.GEMINI_MODEL
GEMINI_PROMPT = config.GEMINI_PROMPT
CAPTURE_DIR   = config.CAPTURE_DIR


def get_api_key() -> str:
    """
    Read GOOGLE_API_KEY (or GEMINI_API_KEY as fallback).
    Exit with a helpful message if neither is set.
    """
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        print(
            "\n[Gemini] ✗ No API key found.\n"
            "  Fix:  export GOOGLE_API_KEY='your-key-here'\n"
            "  Then re-run this script.\n"
        )
        sys.exit(1)
    # Show which env var was used (useful for debugging key conflicts)
    var = "GOOGLE_API_KEY" if os.environ.get("GOOGLE_API_KEY") else "GEMINI_API_KEY"
    print(f"[Gemini] Using key from: {var}")
    return key


def parse_gemini_response(raw: str) -> list[dict]:
    """
    Extract JSON from Gemini's response text.
    Strips markdown fences (```json ... ```) if present.
    Returns list of component dicts, or raises on failure.
    """
    # Remove markdown fences Gemini sometimes adds despite instructions
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    # Find the first complete {...} block
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON object found in response:\n{raw[:200]}")

    data = json.loads(m.group())

    if "components" not in data:
        raise ValueError(f"JSON missing 'components' key. Got keys: {list(data.keys())}")

    return data["components"]


def main():
    print("=" * 60)
    print(" MILESTONE 2 — Gemini API check")
    print("=" * 60)

    # ── 1. API key ────────────────────────────────────────────────────────────
    api_key = get_api_key()

    # ── 2. Initialise Gemini client ───────────────────────────────────────────
    print(f"\n[Gemini] Initialising client — model: {GEMINI_MODEL}")
    try:
        client = genai.Client(api_key=api_key)
        print("[Gemini] ✓ Client initialised")
    except Exception as e:
        print(f"[Gemini] ✗ Failed to create client: {e}")
        sys.exit(1)

    # ── 3. Find test image ────────────────────────────────────────────────────
    # Allow override via environment variable, otherwise use Milestone 1 output
    image_path = Path(os.environ.get("IMAGE_PATH", CAPTURE_DIR / "test_frame.jpg"))

    print(f"\n[Gemini] Looking for test image: {image_path}")
    if not image_path.exists():
        print(
            f"[Gemini] ✗ Image not found: {image_path}\n"
            "  Fix: run Milestone 1 first (press  s  to save a frame)\n"
            f"  OR:  export IMAGE_PATH=/path/to/your/image.jpg"
        )
        sys.exit(1)
    print(f"[Gemini] ✓ Image found ({image_path.stat().st_size} bytes)")

    # ── 4. Load image ─────────────────────────────────────────────────────────
    print("[Gemini] Loading image with PIL…")
    try:
        pil_img = Image.open(image_path).convert("RGB")
        print(f"[Gemini] ✓ Image loaded — size: {pil_img.width}×{pil_img.height}")
    except Exception as e:
        print(f"[Gemini] ✗ Failed to open image: {e}")
        sys.exit(1)

    # ── 5. Send to Gemini ─────────────────────────────────────────────────────
    print(f"\n[Gemini] Sending image to Gemini ({GEMINI_MODEL})…")
    print("[Gemini] (this may take 2–5 seconds on first call)")
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[GEMINI_PROMPT, pil_img],
        )
        raw_text = response.text
        print(f"[Gemini] ✓ Response received ({len(raw_text)} chars)")
    except Exception as e:
        print(f"[Gemini] ✗ API call failed: {e}")
        sys.exit(1)

    # ── 6. Print raw response ─────────────────────────────────────────────────
    print("\n[Gemini] Raw response:")
    print("-" * 40)
    print(raw_text)
    print("-" * 40)

    # ── 7. Parse JSON ─────────────────────────────────────────────────────────
    print("\n[Gemini] Parsing JSON…")
    try:
        components = parse_gemini_response(raw_text)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"[Gemini] ✗ Parse failed: {e}")
        print("[Gemini] This means Gemini returned unexpected text. Try again.")
        sys.exit(1)

    # ── 8. Print parsed result ────────────────────────────────────────────────
    print(f"[Gemini] ✓ Parsed {len(components)} component(s):")
    for i, comp in enumerate(components, 1):
        ctype  = comp.get("type",       "?")
        value  = comp.get("value",      "?")
        loc    = comp.get("location",   "?")
        conf   = comp.get("confidence", "?")
        print(f"  {i}. {ctype} — {value} @ {loc}  [confidence: {conf}]")

    if not components:
        print("  (no components detected — try pointing camera at electronics)")

    # ── Result ────────────────────────────────────────────────────────────────
    print()
    print("[Milestone 2] PASSED ✓")
    print("  Gemini API is working and JSON parsing is correct.")
    print("\n  → Ready for Milestone 3 (python milestone_3_fiftyone.py)")


if __name__ == "__main__":
    main()
