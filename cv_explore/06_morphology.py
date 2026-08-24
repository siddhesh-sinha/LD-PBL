"""
06 — Morphological Operations
================================
What it does: Modifies shapes in binary/grayscale images using structuring
elements (small kernels). Each operation has a geometric effect:

  - Erode: shrinks white regions (removes thin connections)
  - Dilate: grows white regions (fills small gaps)
  - Open: erode then dilate (removes small white noise)
  - Close: dilate then erode (fills small black holes)
  - Gradient: dilate − erode (outlines of objects)
  - TopHat: original − opening (bright details on dark background)
  - BlackHat: closing − original (dark details on bright background)

Why it matters:
  - A TALL NARROW kernel + opening → extracts only vertical lines
  - A WIDE SHORT kernel + opening → extracts only horizontal lines
  - This is the most direct way to isolate column walls and row walls
"""

from common import load_image, make_output_dir, save, crop_board, remove_red_overlay

import cv2
import numpy as np

img, gray = load_image()
out = make_output_dir("06_morphology")

board = crop_board(remove_red_overlay(img), board_idx=0)
board_gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
bh, bw = board_gray.shape

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
enhanced = clahe.apply(board_gray)

# Threshold for morphology demos
_, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
save(out, "00_binary.jpg", binary)

# --- Basic operations with different kernel sizes ---
for ksize in [3, 5, 9, 15]:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    save(out, f"erode_{ksize}.jpg", cv2.erode(binary, kernel))
    save(out, f"dilate_{ksize}.jpg", cv2.dilate(binary, kernel))
    save(out, f"open_{ksize}.jpg", cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel))
    save(out, f"close_{ksize}.jpg", cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel))

# --- Gradient (edge outline) ---
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
save(out, "gradient.jpg", cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel))

# --- TopHat and BlackHat ---
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
save(out, "tophat.jpg", cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel))
save(out, "blackhat.jpg", cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel))

# --- THE KEY ONE: directional line extraction ---
# Vertical lines: tall narrow kernel
for height_frac in [0.1, 0.2, 0.3, 0.5]:
    kh = max(10, int(bh * height_frac))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kh))
    vert_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vert_kernel)
    save(out, f"vert_lines_h{height_frac}.jpg", vert_lines)

# Horizontal lines: wide short kernel
for width_frac in [0.05, 0.1, 0.15, 0.25]:
    kw = max(10, int(bw * width_frac))
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1))
    horiz_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel)
    save(out, f"horiz_lines_w{width_frac}.jpg", horiz_lines)

# --- Elliptical kernels (for blob isolation) ---
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
save(out, "open_ellipse_7.jpg", cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel))
save(out, "close_ellipse_7.jpg", cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel))

print(f"\nDone! Check {out}/")
