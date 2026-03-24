"""
═══════════════════════════════════════════════════════════════════
 MILESTONE 2 — Gemini API check
═══════════════════════════════════════════════════════════════════
 What this tests:
   • GOOGLE_API_KEY is set and valid
   • google-genai SDK installed correctly
   • Gemini can receive an image and return component JSON
   • Our JSON parser handles the response without crashing

 Sends only the FIRST image from pictures/ to Gemini.

 Expected output:
   [Gemini] Using model: gemini-1.5-flash
   [Gemini] Sending pictures/board1.jpg to Gemini...
   [Gemini] Response received (241 chars)
   [Gemini] Raw response:
     {"components": [...]}
   [Gemini] Parsed 3 component(s):
     1. resistor — 10k @ top-left  [confidence: high]
     2. ic — NE555 @ center  [confidence: high]
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
import config   # sets FIFTYONE_DATABASE_DIR + all shared settings

from PIL import Image
from google import genai

GEMINI_MODEL  = config.GEMINI_MODEL
GEMINI_PROMPT = config.GEMINI_PROMPT


def get_api_key():
    """Read API key from environment. Exit with clear message if missing."""
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        print("\n[Gemini] No API key found.")
        print("  Fix:  export GOOGLE_API_KEY='your-key-here'")
        sys.exit(1)
    src = "GOOGLE_API_KEY" if os.environ.get("GOOGLE_API_KEY") else "GEMINI_API_KEY"
    print("[Gemini] API key loaded from: " + src)
    return key


def parse_response(raw):
    """
    Extract JSON from Gemini response text.
    Strips markdown fences if present.
    Returns list of component dicts.
    Raises ValueError / json.JSONDecodeError on failure.
    """
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in response:\n" + raw[:200])
    data = json.loads(m.group())
    if "components" not in data:
        raise ValueError("JSON missing 'components' key. Got: " + str(list(data.keys())))
    return data["components"]


def main():
    print("=" * 60)
    print(" MILESTONE 2 — Gemini API check")
    print("=" * 60)
    print()

    # ── API key ───────────────────────────────────────────────────────────────
    api_key = get_api_key()

    # ── Gemini client ─────────────────────────────────────────────────────────
    print("\n[Gemini] Initialising client — model: " + GEMINI_MODEL)
    try:
        client = genai.Client(api_key=api_key)
        print("[Gemini] Client ready")
    except Exception as e:
        print("[Gemini] Failed to create client: " + str(e))
        sys.exit(1)

    # ── Pick the first image from pictures/ ───────────────────────────────────
    image_paths = config.get_image_paths()
    img_path    = image_paths[0]   # test with first image only
    print()
    print("[Gemini] Test image: " + str(img_path))

    # ── Load image ────────────────────────────────────────────────────────────
    print("[Gemini] Loading image with PIL...")
    try:
        pil_img = Image.open(img_path).convert("RGB")
        print("[Gemini] Image loaded — " + str(pil_img.width) + "x" + str(pil_img.height))
    except Exception as e:
        print("[Gemini] Failed to open image: " + str(e))
        sys.exit(1)

    # ── Send to Gemini ────────────────────────────────────────────────────────
    print("\n[Gemini] Sending image to Gemini (may take 2-5 seconds)...")
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[GEMINI_PROMPT, pil_img],
        )
        raw_text = response.text
        print("[Gemini] Response received (" + str(len(raw_text)) + " chars)")
    except Exception as e:
        print("[Gemini] API call failed: " + str(e))
        sys.exit(1)

    # ── Print raw response ────────────────────────────────────────────────────
    print()
    print("[Gemini] Raw response:")
    print("-" * 40)
    print(raw_text)
    print("-" * 40)

    # ── Parse JSON ────────────────────────────────────────────────────────────
    print()
    print("[Gemini] Parsing JSON...")
    try:
        components = parse_response(raw_text)
    except (json.JSONDecodeError, ValueError) as e:
        print("[Gemini] Parse failed: " + str(e))
        print("[Gemini] Gemini returned unexpected text — try again")
        sys.exit(1)

    # ── Print results ─────────────────────────────────────────────────────────
    print("[Gemini] Parsed " + str(len(components)) + " component(s):")
    for i, comp in enumerate(components, 1):
        ctype = comp.get("type",       "?")
        value = comp.get("value",      "?")
        loc   = comp.get("location",   "?")
        conf  = comp.get("confidence", "?")
        print("  " + str(i) + ". " + ctype + " — " + value +
              " @ " + loc + "  [confidence: " + conf + "]")

    if not components:
        print("  (no components detected — try a clearer image of electronics)")

    # ── Result ────────────────────────────────────────────────────────────────
    print()
    print("[Milestone 2] PASSED ✓")
    print("  Gemini API is working and JSON parsing is correct")
    print()
    print("  -> Ready for Milestone 3  (python milestone_3_fiftyone.py)")


if __name__ == "__main__":
    main()
