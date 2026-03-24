"""
═══════════════════════════════════════════════════════════════════
 MILESTONE 1 — Image loader check
═══════════════════════════════════════════════════════════════════
 What this tests:
   • pictures/ folder exists and has images in it
   • OpenCV can load each image correctly
   • Prints resolution and brightness for each image
   • Shows each image in a window so you can visually confirm

 How to use:
   1. Create a folder called  pictures/  in your project directory
   2. Drop any .jpg or .png electronics photos into it
   3. Run this script

 Expected output:
   [Images] Found 2 image(s) in pictures/
     board1.jpg  (142.3 KB)
     resistors.png  (98.1 KB)
   [Loader] Loading board1.jpg ...  OK  640x480  brightness:112.3
   [Loader] Loading resistors.png ...  OK  1920x1080  brightness:89.7
   [Milestone 1] PASSED ✓

 Controls (while image window is open):
   any key — next image
   q       — quit early

 Run:
   python milestone_1_images.py
═══════════════════════════════════════════════════════════════════
"""

import sys
import config   # sets FIFTYONE_DATABASE_DIR + all shared settings
import cv2

PICTURES_DIR    = config.PICTURES_DIR
IMAGE_EXTENSIONS = config.IMAGE_EXTENSIONS


def main():
    print("=" * 60)
    print(" MILESTONE 1 — Image loader check")
    print("=" * 60)
    print()

    # ── Get image list (exits with helpful message if folder empty) ───────────
    image_paths = config.get_image_paths()
    print()

    # ── Load and display each image ───────────────────────────────────────────
    failed  = []
    passed  = []

    win_title = "Milestone 1 — Image Check  (any key = next,  q = quit)"
    cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)

    for img_path in image_paths:
        print("[Loader] Loading " + img_path.name + " ...", end="  ", flush=True)

        frame = cv2.imread(str(img_path))

        if frame is None:
            print("FAILED — OpenCV could not read this file")
            failed.append(img_path.name)
            continue

        h, w  = frame.shape[:2]
        brightness = round(frame.mean(), 1)
        print("OK  " + str(w) + "x" + str(h) + "  brightness:" + str(brightness))
        passed.append(img_path.name)

        # Show the image — resize window to fit
        cv2.resizeWindow(win_title, min(w, 900), min(h, 700))
        cv2.imshow(win_title, frame)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            print("[Loader] Quit early")
            break

    cv2.destroyAllWindows()

    # ── Result ────────────────────────────────────────────────────────────────
    print()
    print("Summary:")
    print("  Passed: " + str(len(passed)) + "  ->  " + str(passed))
    if failed:
        print("  Failed: " + str(len(failed)) + "  ->  " + str(failed))
        print()
        print("[Milestone 1] PARTIAL — some images failed to load")
        print("  Check that failed files are valid .jpg or .png images")
    else:
        print()
        print("[Milestone 1] PASSED ✓")
        print("  All images loaded successfully")
        print()
        print("  -> Ready for Milestone 2  (python milestone_2_gemini.py)")


if __name__ == "__main__":
    main()
