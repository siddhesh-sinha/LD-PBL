"""
01 — Color Space Decomposition
================================
What it does: Splits the image into its component channels across
different color models. Each channel highlights different properties.

Why it matters for us:
  - HSV Saturation → separates the colored cotton plugs from white board
  - LAB L-channel → luminance only, best for dark-fly-on-white-board
  - LAB A-channel → green-red axis, may isolate the reddish plugs
  - HSV Value → brightness, good for finding shadows = chamber walls
"""

from common import load_image, make_output_dir, save, crop_board, remove_red_overlay

import cv2

img, gray = load_image()
out = make_output_dir("01_color_spaces")

# Work on one board (cleaned)
board = crop_board(remove_red_overlay(img), board_idx=0)

# Grayscale
save(out, "gray.jpg", cv2.cvtColor(board, cv2.COLOR_BGR2GRAY))

# HSV channels
hsv = cv2.cvtColor(board, cv2.COLOR_BGR2HSV)
save(out, "hsv_h.jpg", hsv[:, :, 0])  # Hue
save(out, "hsv_s.jpg", hsv[:, :, 1])  # Saturation
save(out, "hsv_v.jpg", hsv[:, :, 2])  # Value (brightness)

# LAB channels
lab = cv2.cvtColor(board, cv2.COLOR_BGR2LAB)
save(out, "lab_L.jpg", lab[:, :, 0])  # Lightness
save(out, "lab_A.jpg", lab[:, :, 1])  # Green-Red
save(out, "lab_B.jpg", lab[:, :, 2])  # Blue-Yellow

# Individual BGR channels
save(out, "bgr_blue.jpg", board[:, :, 0])
save(out, "bgr_green.jpg", board[:, :, 1])
save(out, "bgr_red.jpg", board[:, :, 2])

# YCrCb (used in skin/object detection)
ycrcb = cv2.cvtColor(board, cv2.COLOR_BGR2YCrCb)
save(out, "ycrcb_Y.jpg", ycrcb[:, :, 0])   # Luma
save(out, "ycrcb_Cr.jpg", ycrcb[:, :, 1])  # Red chroma
save(out, "ycrcb_Cb.jpg", ycrcb[:, :, 2])  # Blue chroma

print(f"\nDone! Check {out}/")
