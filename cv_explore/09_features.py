"""
09 — Feature Detection (Keypoints)
=====================================
What it does: Finds "interesting" points in the image — corners,
blobs, texture changes. These are used in computer vision for matching,
tracking, and recognition.

  - ORB: fast binary descriptor, good for real-time
  - SIFT: scale-invariant, the classic (patented until 2020)
  - Corner detection: Harris, Shi-Tomasi (good features to track)

Why it matters:
  - Keypoints cluster where there's structure (walls, plugs, flies)
  - Corner detectors find wall intersections → grid nodes
  - The DENSITY of keypoints reveals where the action is
"""

from common import load_image, make_output_dir, save, crop_board, remove_red_overlay

import cv2
import numpy as np

img, gray = load_image()
out = make_output_dir("09_features")

board = crop_board(remove_red_overlay(img), board_idx=0)
board_gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
bh, bw = board_gray.shape

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
enhanced = clahe.apply(board_gray)

# --- ORB keypoints ---
orb = cv2.ORB.create(nfeatures=2000)
kp_orb = orb.detect(enhanced, None)
vis_orb = cv2.drawKeypoints(board, kp_orb, None, (0, 255, 0), 0)
save(out, "orb_keypoints.jpg", vis_orb)
print(f"  ORB: {len(kp_orb)} keypoints")

# --- SIFT keypoints ---
try:
    sift = cv2.SIFT.create(nfeatures=2000)
    kp_sift = sift.detect(enhanced, None)
    vis_sift = cv2.drawKeypoints(board, kp_sift, None, (0, 0, 255),
                                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    save(out, "sift_keypoints.jpg", vis_sift)
    print(f"  SIFT: {len(kp_sift)} keypoints")
except Exception as e:
    print(f"  SIFT not available: {e}")

# --- Harris corner detection ---
harris = cv2.cornerHarris(enhanced.astype(np.float32), blockSize=2, ksize=3, k=0.04)
harris_norm = cv2.normalize(harris, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
save(out, "harris_response.jpg", harris_norm)

# Threshold and mark corners
vis_harris = board.copy()
vis_harris[harris > 0.01 * harris.max()] = [0, 0, 255]
save(out, "harris_corners.jpg", vis_harris)

# --- Shi-Tomasi (Good Features to Track) ---
for n_corners in [100, 500, 2000]:
    corners = cv2.goodFeaturesToTrack(enhanced, n_corners, 0.01, 10)
    vis_gft = board.copy()
    if corners is not None:
        for c in corners:
            x, y = c.ravel()
            cv2.circle(vis_gft, (int(x), int(y)), 3, (0, 255, 0), -1)
    save(out, f"shi_tomasi_{n_corners}.jpg", vis_gft)
    n = len(corners) if corners is not None else 0
    print(f"  Shi-Tomasi (max {n_corners}): {n} corners")

# --- Keypoint density heatmap ---
# How many keypoints per 50x50 tile?
tile = 30
density = np.zeros((bh // tile + 1, bw // tile + 1), dtype=np.float32)
for kp in kp_orb:
    tx = int(kp.pt[0]) // tile
    ty = int(kp.pt[1]) // tile
    if 0 <= ty < density.shape[0] and 0 <= tx < density.shape[1]:
        density[ty, tx] += 1

density_vis = cv2.normalize(density, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
density_vis = cv2.resize(density_vis, (bw, bh), interpolation=cv2.INTER_NEAREST)
density_color = cv2.applyColorMap(density_vis, cv2.COLORMAP_JET)
blended = cv2.addWeighted(board, 0.5, density_color, 0.5, 0)
save(out, "keypoint_density_heatmap.jpg", blended)

print(f"\nDone! Check {out}/")
