"""
07 — Blob Detection
=====================
What it does: Finds "blobs" — connected regions of similar pixels.
Two approaches:
  1. SimpleBlobDetector — built-in detector with configurable filters
  2. Contour-based — threshold → findContours → filter by area/shape

Why it matters:
  - Flies are dark blobs on a white background
  - Cotton plugs are colored blobs at the top of each column
  - Detecting fly positions directly gives us the row grid
  - Blob circularity can distinguish flies (round) from wall marks (elongated)
"""

from common import load_image, make_output_dir, save, crop_board, remove_red_overlay

import cv2
import numpy as np

img, gray = load_image()
out = make_output_dir("07_blobs")

board = crop_board(remove_red_overlay(img), board_idx=0)
board_gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
bh, bw = board_gray.shape

# --- SimpleBlobDetector ---
params = cv2.SimpleBlobDetector.Params()
params.filterByColor = True
params.blobColor = 0  # dark blobs
params.filterByArea = True
params.minArea = 30
params.maxArea = 2000
params.filterByCircularity = False
params.filterByConvexity = False
params.filterByInertia = False
params.minThreshold = 10
params.maxThreshold = 200
params.thresholdStep = 10

detector = cv2.SimpleBlobDetector.create(params)
keypoints = detector.detect(board_gray)
vis = cv2.drawKeypoints(board, keypoints, None, (0, 0, 255),
                         cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
save(out, "simple_blob.jpg", vis)
print(f"  SimpleBlobDetector: {len(keypoints)} blobs")

# With circularity filter
params.filterByCircularity = True
params.minCircularity = 0.3
detector2 = cv2.SimpleBlobDetector.create(params)
kp2 = detector2.detect(board_gray)
vis2 = cv2.drawKeypoints(board, kp2, None, (0, 255, 0),
                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
save(out, "simple_blob_circular.jpg", vis2)
print(f"  With circularity filter: {len(kp2)} blobs")

# --- Contour-based blob detection ---
# Method 1: Otsu on inverted
_, thresh = cv2.threshold(board_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
save(out, "otsu_inv.jpg", thresh)

# Clean
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Classify by size
vis_contour = board.copy()
small, medium, large = 0, 0, 0
for c in contours:
    area = cv2.contourArea(c)
    if area < 30:
        continue
    elif area < 200:
        cv2.drawContours(vis_contour, [c], -1, (255, 0, 0), 1)  # blue = small
        small += 1
    elif area < 2000:
        cv2.drawContours(vis_contour, [c], -1, (0, 255, 0), 2)  # green = fly-sized
        medium += 1
    else:
        cv2.drawContours(vis_contour, [c], -1, (0, 0, 255), 2)  # red = large
        large += 1

save(out, "contour_classified.jpg", vis_contour)
print(f"  Contours: {small} small, {medium} fly-sized, {large} large")

# Method 2: LAB L-channel (better for dark objects)
lab = cv2.cvtColor(board, cv2.COLOR_BGR2LAB)
l_ch = lab[:, :, 0]
_, lab_thresh = cv2.threshold(l_ch, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
lab_cleaned = cv2.morphologyEx(lab_thresh, cv2.MORPH_OPEN, kernel)
lab_cleaned = cv2.morphologyEx(lab_cleaned, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
save(out, "lab_L_thresh.jpg", lab_cleaned)

contours_lab, _ = cv2.findContours(lab_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
vis_lab = board.copy()
fly_centroids = []
for c in contours_lab:
    area = cv2.contourArea(c)
    if 30 < area < 2000:
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            fly_centroids.append((cx, cy))
            cv2.circle(vis_lab, (cx, cy), 5, (0, 255, 0), -1)

save(out, "fly_centroids_lab.jpg", vis_lab)
print(f"  LAB-based fly detection: {len(fly_centroids)} flies")

# --- Centroid scatter plot (positions reveal the grid!) ---
scatter = np.zeros((bh, bw, 3), dtype=np.uint8)
for cx, cy in fly_centroids:
    cv2.circle(scatter, (cx, cy), 4, (0, 255, 0), -1)
save(out, "centroid_scatter.jpg", scatter)

print(f"\nDone! Check {out}/")
