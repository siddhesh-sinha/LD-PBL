"""
04 — Edge Detection Methods
==============================
What it does: Finds boundaries where pixel intensity changes sharply.
Different methods emphasize different kinds of edges.

Why it matters:
  - Chamber walls = edges. Better edge detection = better grid finding.
  - Canny → the classic, good at finding clean connected edges
  - Sobel X → vertical edges (= column walls!)
  - Sobel Y → horizontal edges (= row boundaries!)
  - Laplacian → all edges equally, sensitive to noise
  - Scharr → more accurate than Sobel at diagonal edges
"""

from common import load_image, make_output_dir, save, crop_board, remove_red_overlay

import cv2
import numpy as np

img, gray = load_image()
out = make_output_dir("04_edges")

board = crop_board(remove_red_overlay(img), board_idx=0)
board_gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)

# Enhance first
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
enhanced = clahe.apply(board_gray)

# Also try bilateral filter (smooths flat areas, keeps edges)
bilateral = cv2.bilateralFilter(board_gray, 9, 75, 75)

# --- Canny with different thresholds ---
for lo, hi in [(20, 60), (30, 100), (50, 150), (80, 200)]:
    save(out, f"canny_{lo}_{hi}.jpg", cv2.Canny(enhanced, lo, hi))

# Canny on bilateral
save(out, "canny_bilateral.jpg", cv2.Canny(bilateral, 30, 100))

# --- Sobel (directional) ---
sobel_x = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)

save(out, "sobel_x.jpg", np.uint8(np.abs(sobel_x) / np.abs(sobel_x).max() * 255))
save(out, "sobel_y.jpg", np.uint8(np.abs(sobel_y) / np.abs(sobel_y).max() * 255))
save(out, "sobel_mag.jpg", np.uint8(sobel_mag / sobel_mag.max() * 255))

# Sobel with larger kernels (smoother)
for ksize in [3, 5, 7]:
    sx = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=ksize)
    save(out, f"sobel_x_k{ksize}.jpg", np.uint8(np.abs(sx) / np.abs(sx).max() * 255))

# --- Scharr (more accurate than Sobel for 3x3) ---
scharr_x = cv2.Scharr(enhanced, cv2.CV_64F, 1, 0)
scharr_y = cv2.Scharr(enhanced, cv2.CV_64F, 0, 1)
save(out, "scharr_x.jpg", np.uint8(np.abs(scharr_x) / np.abs(scharr_x).max() * 255))
save(out, "scharr_y.jpg", np.uint8(np.abs(scharr_y) / np.abs(scharr_y).max() * 255))

# --- Laplacian ---
for ksize in [1, 3, 5]:
    lap = cv2.Laplacian(enhanced, cv2.CV_64F, ksize=ksize)
    save(out, f"laplacian_k{ksize}.jpg", np.uint8(np.abs(lap) / np.abs(lap).max() * 255))

print(f"\nDone! Check {out}/")
