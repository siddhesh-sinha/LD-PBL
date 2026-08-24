"""
17 — Blur Descent: Heavy blur → reduce → structure emerges
===========================================================
Start with massive blur — only coarse structure survives (9 strips).
Reduce blur progressively — plug rows appear, then individual plugs.
The blur IS the noise filter. Dark vs light tells us everything.
"""

from common import load_image, make_output_dir, save, crop_board, remove_red_overlay

import cv2
import numpy as np

img, gray = load_image()
out = make_output_dir("17_blur_descent")

board = crop_board(remove_red_overlay(img), board_idx=0)
board_gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
bh, bw = board_gray.shape
print(f"  Board: {bw}x{bh}")

# Invert: dark plugs → bright (high values)
inv = (255 - board_gray).astype(np.float32)

# ── Blur descent ─────────────────────────────────────────
sigmas = [80, 60, 40, 30, 20, 15, 10, 7, 5, 3]

for i, sigma in enumerate(sigmas):
    # Kernel size must be odd and large enough for sigma
    ksize = int(sigma * 6) | 1  # 6*sigma, forced odd
    blurred = cv2.GaussianBlur(inv, (ksize, ksize), sigma)

    # Normalize to 0-255 for visualization
    bmin, bmax = blurred.min(), blurred.max()
    vis = ((blurred - bmin) / (bmax - bmin + 1e-6) * 255).astype(np.uint8)

    # Also threshold at the mean to get binary dark/light
    mean_val = blurred.mean()
    binary = (blurred > mean_val).astype(np.uint8) * 255

    # Colorize: dark regions (above mean) = red overlay on board
    overlay = board.copy()
    mask = blurred > mean_val
    overlay[mask] = (overlay[mask] * 0.4 + np.array([0, 0, 180]) * 0.6).astype(np.uint8)

    save(out, f"{i:02d}_blur_s{sigma:03d}.jpg", vis)
    save(out, f"{i:02d}_binary_s{sigma:03d}.jpg", binary)
    save(out, f"{i:02d}_overlay_s{sigma:03d}.jpg", overlay)

    # Count dark regions at this scale
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8)
    # Filter tiny components
    big = [j for j in range(1, n_labels) if stats[j, cv2.CC_STAT_AREA] > 100]
    print(f"  σ={sigma:3d}: {len(big)} dark regions, "
          f"range={bmin:.0f}-{bmax:.0f}, mean={mean_val:.0f}")

print(f"\nDone! Check {out}/")
