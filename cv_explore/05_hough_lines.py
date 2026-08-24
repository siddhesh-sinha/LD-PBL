"""
05 — Hough Line Transform
============================
What it does: Detects straight lines in an edge image. Two variants:
  - Standard Hough: finds infinite lines (rho, theta parameterization)
  - Probabilistic Hough: finds line segments with endpoints

Why it matters:
  - Chamber walls ARE straight lines
  - Vertical Hough lines → column boundaries
  - Horizontal Hough lines → row boundaries
  - Can filter by angle to isolate vertical-only or horizontal-only
"""

from common import load_image, make_output_dir, save, crop_board, remove_red_overlay

import cv2
import numpy as np

img, gray = load_image()
out = make_output_dir("05_hough_lines")

board = crop_board(remove_red_overlay(img), board_idx=0)
board_gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
bh, bw = board_gray.shape

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
enhanced = clahe.apply(board_gray)
edges = cv2.Canny(enhanced, 30, 100)
save(out, "00_edges.jpg", edges)

# --- Standard Hough ---
lines_std = cv2.HoughLines(edges, 1, np.pi / 180, threshold=200)
vis_std = cv2.cvtColor(board_gray, cv2.COLOR_GRAY2BGR)
if lines_std is not None:
    for line in lines_std[:100]:  # limit to first 100
        rho, theta = line[0] if line.ndim > 1 else (line[0], line[1])
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        pt1 = (int(x0 + 2000 * (-b)), int(y0 + 2000 * a))
        pt2 = (int(x0 - 2000 * (-b)), int(y0 - 2000 * a))
        cv2.line(vis_std, pt1, pt2, (0, 0, 255), 1)
    print(f"  Standard Hough: {len(lines_std)} lines")
save(out, "hough_standard.jpg", vis_std)

# --- Probabilistic Hough with different parameters ---
for min_len_frac, max_gap, thresh in [
    (0.3, 10, 50),
    (0.3, 20, 80),
    (0.5, 10, 100),
    (0.2, 30, 40),
]:
    min_len = int(bh * min_len_frac)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, thresh,
                             minLineLength=min_len, maxLineGap=max_gap)
    vis = cv2.cvtColor(board_gray, cv2.COLOR_GRAY2BGR)
    if lines is not None:
        if lines.ndim == 3:
            lines = lines.reshape(-1, 4)
        for x1, y1, x2, y2 in lines:
            dx, dy = abs(x2 - x1), abs(y2 - y1)
            angle = 90 if dx == 0 else abs(np.degrees(np.arctan2(dy, dx)))
            color = (0, 255, 0) if angle > 75 else (0, 0, 255) if angle < 15 else (128, 128, 0)
            cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
    n = len(lines) if lines is not None else 0
    name = f"hough_p_len{min_len_frac}_gap{max_gap}_t{thresh}"
    save(out, f"{name}.jpg", vis)
    print(f"  {name}: {n} lines")

# --- Vertical-only and horizontal-only extractions ---
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80,
                         minLineLength=int(bh * 0.3), maxLineGap=20)
vis_v = cv2.cvtColor(board_gray, cv2.COLOR_GRAY2BGR)
vis_h = cv2.cvtColor(board_gray, cv2.COLOR_GRAY2BGR)

if lines is not None:
    if lines.ndim == 3:
        lines = lines.reshape(-1, 4)
    for x1, y1, x2, y2 in lines:
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        angle = 90 if dx == 0 else abs(np.degrees(np.arctan2(dy, dx)))
        if angle > 80:
            cv2.line(vis_v, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        elif angle < 10:
            cv2.line(vis_h, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

save(out, "vertical_only.jpg", vis_v)
save(out, "horizontal_only.jpg", vis_h)

print(f"\nDone! Green=vertical, Red=horizontal, Yellow=diagonal. Check {out}/")
