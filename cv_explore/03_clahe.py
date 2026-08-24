"""
03 — CLAHE (Contrast Limited Adaptive Histogram Equalization)
==============================================================
What it does: Enhances local contrast by equalizing the histogram in
small tiles. Unlike global histogram equalization, CLAHE doesn't blow
out bright regions — the "clip limit" caps how much any bin grows.

Why it matters:
  - The boards have uneven lighting (center brighter, edges darker)
  - CLAHE normalizes this so chamber walls have consistent contrast
  - Different clipLimit values trade noise vs contrast
  - Different tileGridSize values trade local vs global equalization
"""

from common import load_image, make_output_dir, save, crop_board, remove_red_overlay

import cv2

img, gray = load_image()
out = make_output_dir("03_clahe")

board = crop_board(remove_red_overlay(img), board_idx=0)
board_gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
save(out, "00_raw_gray.jpg", board_gray)

# Regular histogram equalization (for comparison)
save(out, "hist_eq.jpg", cv2.equalizeHist(board_gray))

# CLAHE with varying clip limits
for clip in [1.0, 2.0, 3.0, 5.0, 10.0, 20.0]:
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    save(out, f"clahe_clip{clip:.0f}.jpg", clahe.apply(board_gray))

# CLAHE with varying tile sizes
for tile in [4, 8, 16, 32]:
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(tile, tile))
    save(out, f"clahe_tile{tile}.jpg", clahe.apply(board_gray))

# CLAHE on LAB L-channel (preserves color)
lab = cv2.cvtColor(board, cv2.COLOR_BGR2LAB)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
lab[:, :, 0] = clahe.apply(lab[:, :, 0])
enhanced_color = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
save(out, "clahe_color.jpg", enhanced_color)

print(f"\nDone! Check {out}/")
