"""
02 — Thresholding Methods
===========================
What it does: Converts grayscale → binary (black/white) using different
strategies. The choice of method dramatically affects what structures
become visible.

Why it matters:
  - Otsu → automatic global threshold, good for board-vs-background
  - Adaptive Mean → local threshold, finds fine structure like chamber walls
  - Adaptive Gaussian → same but weighted, less noisy than mean
  - Binary Inverse → flips what's "object" vs "background"
"""

from common import load_image, make_output_dir, save, crop_board, remove_red_overlay

import cv2

img, gray = load_image()
out = make_output_dir("02_thresholding")

board = crop_board(remove_red_overlay(img), board_idx=0)
board_gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)

# Apply CLAHE first (makes thresholding work better)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
enhanced = clahe.apply(board_gray)
save(out, "00_enhanced.jpg", enhanced)

# Global thresholds
_, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
save(out, "otsu.jpg", otsu)

_, otsu_inv = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
save(out, "otsu_inv.jpg", otsu_inv)

# Fixed thresholds at different levels
for thresh in [80, 120, 160, 200]:
    _, fixed = cv2.threshold(enhanced, thresh, 255, cv2.THRESH_BINARY)
    save(out, f"fixed_{thresh}.jpg", fixed)

# Adaptive thresholds with different block sizes
for block in [11, 21, 51, 101]:
    adapt_mean = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block, 5
    )
    save(out, f"adapt_mean_b{block}.jpg", adapt_mean)

    adapt_gauss = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 5
    )
    save(out, f"adapt_gauss_b{block}.jpg", adapt_gauss)

# Triangle threshold (good for unimodal histograms)
_, triangle = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
save(out, "triangle.jpg", triangle)

print(f"\nDone! Check {out}/")
