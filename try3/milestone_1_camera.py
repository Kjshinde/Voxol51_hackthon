"""
═══════════════════════════════════════════════════════════════════
 MILESTONE 1 — Camera check
═══════════════════════════════════════════════════════════════════
 What this tests:
   • USB camera passthrough is working in VirtualBox
   • OpenCV can open the camera and read frames
   • The live feed displays in a window
   • A single test frame is saved to disk

 Expected output:
   [Camera] ✓ Opened /dev/video2  —  resolution: 640×480
   [Camera] Showing live feed — press  s  to save a frame,  q  to quit
   [Camera] ✓ Frame saved → captures/test_frame.jpg
   [Milestone 1] PASSED ✓

 If it fails:
   • "not available" → check VirtualBox Devices → USB → tick IPEVO V4K
   • Window doesn't open → check your display (DISPLAY env var on headless)

 Run:
   python milestone_1_camera.py
═══════════════════════════════════════════════════════════════════
"""

import sys
import config  # loads DB env var + all shared settings
import cv2
from pathlib import Path

# ── Pull settings from config ─────────────────────────────────────────────────
CAMERA_INDICES = config.CAMERA_INDICES
CAPTURE_DIR    = config.CAPTURE_DIR


def open_camera():
    """Delegate to config.open_camera() — single source of truth for the MJPEG fix."""
    return config.open_camera()

    # Nothing worked
    print(
        "\n[Camera] ✗ No working camera found.\n"
        "  Fix: VirtualBox menu → Devices → USB → tick 'IPEVO V4K'\n"
        "  Then re-run this script.\n"
    )
    sys.exit(1)


def main():
    print("=" * 60)
    print(" MILESTONE 1 — Camera check")
    print("=" * 60)

    cap, cam_idx = open_camera()

    # Make sure the captures folder exists
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    save_path  = CAPTURE_DIR / "test_frame.jpg"
    frame_saved = False

    print(f"\n[Camera] Showing live feed from /dev/video{cam_idx}")
    print("[Camera] Press  s  to save a test frame,  q  to quit")

    win_title = "Milestone 1 — Camera Check (s=save, q=quit)"
    cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Camera] ✗ Failed to read frame — check connection")
            break

        # Show a small status banner on the frame
        label = "LIVE FEED — press S to save, Q to quit"
        cv2.putText(
            frame, label, (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 255, 0), 2, cv2.LINE_AA,
        )

        cv2.imshow(win_title, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            cv2.imwrite(str(save_path), frame)
            frame_saved = True
            print(f"[Camera] ✓ Frame saved → {save_path}")

        elif key == ord("q"):
            print("[Camera] Quit key pressed — closing")
            break

    cap.release()
    cv2.destroyAllWindows()

    # ── Result ────────────────────────────────────────────────────────────────
    print()
    if frame_saved and save_path.exists():
        size = save_path.stat().st_size
        print(f"[Milestone 1] PASSED ✓")
        print(f"  Camera /dev/video{cam_idx} works")
        print(f"  Test frame: {save_path}  ({size} bytes)")
        print(f"\n  → Ready for Milestone 2 (python milestone_2_gemini.py)")
    else:
        print("[Milestone 1] WARNING — camera worked but no frame was saved")
        print("  Re-run and press  s  while the window is open")


if __name__ == "__main__":
    main()
