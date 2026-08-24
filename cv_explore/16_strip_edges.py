"""
16 — Strip Edge Detection (Blob → Centerline → Edge Scan)
==========================================================
Strategy:
  1. Density profile → 9 strip boundaries (reliable)
  2. Per-strip Otsu → blob centroids (local threshold adapts)
  3. Fit centerline through blob centroids per strip
  4. Scan perpendicular to centerline → column-wise darkness
     → find left/right edge where darkness drops off
  5. Per-strip zoomed visualization for multimodal analysis
"""

from common import load_image, make_output_dir, save, crop_board, remove_red_overlay

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

img, gray = load_image()
out = make_output_dir("16_strip_edges")

board = crop_board(remove_red_overlay(img), board_idx=0)
board_gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
bh, bw = board_gray.shape
print(f"  Board: {bw}x{bh}")

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
inv = 255 - clahe.apply(board_gray)

y_top = int(bh * 0.03)
y_bot = int(bh * 0.92)
MIN_PLUG_AREA = 200

# ── Strip boundaries via density profile ─────────────────
v_profile = gaussian_filter1d(np.mean(inv[y_top:y_bot, :], axis=0), sigma=5)
spx, _ = find_peaks(v_profile, distance=bw // 18, prominence=5)
svl, _ = find_peaks(-v_profile, distance=bw // 18, prominence=3)

all_bounds = sorted(np.concatenate(([0], svl, [bw - 1])))
strip_regions = []
for i in range(len(all_bounds) - 1):
    x0, x1 = int(all_bounds[i]), int(all_bounds[i + 1])
    if x1 - x0 > 15 and any(x0 <= p <= x1 for p in spx):
        strip_regions.append((x0, x1))
print(f"  Strips: {len(strip_regions)}")


def find_plug_centroids(strip_gray_region):
    """Per-strip Otsu blob detection — local threshold adapts."""
    enh = clahe.apply(strip_gray_region)
    _, otsu = cv2.threshold(enh, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        cleaned, connectivity=8)
    plugs = []
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_PLUG_AREA:
            plugs.append((centroids[i][0], centroids[i][1],
                          stats[i, cv2.CC_STAT_AREA]))  # x_local, y_local, area
    plugs.sort(key=lambda p: p[1])  # sort by y
    return plugs


colors = [
    (80, 80, 255), (80, 255, 80), (255, 140, 40),
    (255, 255, 40), (40, 255, 255), (255, 40, 255),
    (160, 255, 40), (255, 160, 40), (40, 160, 255),
]

vis_all = board.copy()

for si, (x0, x1) in enumerate(strip_regions):
    color = colors[si % len(colors)]
    strip_w = x1 - x0

    # Per-strip blob detection (local Otsu)
    plugs = find_plug_centroids(board_gray[y_top:y_bot, x0:x1])
    if len(plugs) < 3:
        print(f"  S{si}: only {len(plugs)} blobs — skip")
        continue

    # Convert to board coordinates
    bxs = np.array([p[0] + x0 for p in plugs])      # x in board coords
    bys = np.array([p[1] + y_top for p in plugs])     # y in board coords

    # ── Fit centerline: x = mx + slope*(y - my) ─────
    mx, my = bxs.mean(), bys.mean()
    dy = bys - my
    slope = np.dot(dy, bxs - mx) / (np.dot(dy, dy) + 1e-9)

    def center_x(y):
        return mx + slope * (y - my)

    y_min_b = int(bys.min()) - 5
    y_max_b = int(bys.max()) + 5

    # ── Perpendicular darkness scan ──────────────────
    # Scan 60px each side from centerline (strips are ~60-80px wide)
    scan_half = 60
    n_cols = scan_half * 2 + 1
    height = y_max_b - y_min_b

    darkness_map = np.zeros((height, n_cols), dtype=float)
    for row_i, y in enumerate(range(y_min_b, y_max_b)):
        cx = center_x(y)
        for di in range(n_cols):
            px = int(round(cx + di - scan_half))
            if 0 <= px < bw and 0 <= y < bh:
                darkness_map[row_i, di] = inv[y, px]

    # Column-mean darkness profile (perpendicular to centerline)
    perp_profile = np.mean(darkness_map, axis=0)
    perp_smooth = gaussian_filter1d(perp_profile, sigma=3)
    mid = scan_half  # blob-centroid center

    # Find the ACTUAL darkness peak (may differ from blob centroid mean)
    peak_idx = int(np.argmax(perp_smooth))
    peak_val = perp_smooth[peak_idx]

    # Find ALL valleys in the profile
    valleys, _ = find_peaks(-perp_smooth, distance=8, prominence=2)

    # Only keep valleys that drop below 60% of peak
    deep = [v for v in valleys if perp_smooth[v] < peak_val * 0.6]

    # Closest deep valley on each side of the DARKNESS PEAK
    left_vals = [v for v in deep if v < peak_idx]
    right_vals = [v for v in deep if v > peak_idx]
    left_edge = int(left_vals[-1]) if left_vals else 0
    right_edge = int(right_vals[0]) if right_vals else n_cols - 1

    # Offsets relative to blob centroid center (for drawing)
    left_off = left_edge - mid
    right_off = right_edge - mid
    # Also track offset from darkness peak
    peak_shift = peak_idx - mid  # how far blob mean is from true center

    strip_width_px = right_edge - left_edge
    print(f"  S{si}: {len(plugs)} blobs, slope={slope:.4f}, "
          f"edges L={left_off} R={right_off} (w={strip_width_px}px), "
          f"peak_shift={peak_shift}")

    # ── Draw on combined vis ─────────────────────────
    # Centerline (thick)
    cv2.line(vis_all, (int(center_x(y_min_b)), y_min_b),
             (int(center_x(y_max_b)), y_max_b), (255, 255, 255), 2)
    # Left edge
    cv2.line(vis_all,
             (int(center_x(y_min_b) + left_off), y_min_b),
             (int(center_x(y_max_b) + left_off), y_max_b), color, 2)
    # Right edge
    cv2.line(vis_all,
             (int(center_x(y_min_b) + right_off), y_min_b),
             (int(center_x(y_max_b) + right_off), y_max_b), color, 2)

    # Blob centroids
    for bx, by in zip(bxs, bys):
        cv2.circle(vis_all, (int(bx), int(by)), 4, (0, 255, 255), -1)

    cv2.putText(vis_all, f"S{si}:{len(plugs)}",
                (int(mx) - 15, max(y_min_b - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    # ── Per-strip zoomed crop ────────────────────────
    pad = 15
    crop_x0 = max(0, int(center_x(my) + left_off) - pad)
    crop_x1 = min(bw, int(center_x(my) + right_off) + pad)
    crop = board[max(0, y_min_b):min(bh, y_max_b),
                 crop_x0:crop_x1].copy()

    # Overlay centerline and edges on crop
    for y in range(y_min_b, y_max_b):
        cx_local = int(center_x(y)) - crop_x0
        lx = cx_local + left_off
        rx = cx_local + right_off
        if 0 <= lx < crop.shape[1]:
            crop[y - y_min_b, lx] = color
        if 0 <= rx < crop.shape[1]:
            crop[y - y_min_b, rx] = color

    for bx, by in zip(bxs, bys):
        lx, ly = int(bx) - crop_x0, int(by) - y_min_b
        if 0 <= lx < crop.shape[1] and 0 <= ly < crop.shape[0]:
            cv2.circle(crop, (lx, ly), 3, (0, 255, 255), -1)

    save(out, f"strip_{si}_zoom.jpg", crop)

    # ── Perpendicular profile plot ───────────────────
    prof_h, prof_w = 100, n_cols
    pvis = np.zeros((prof_h, prof_w, 3), dtype=np.uint8)
    norm = perp_smooth / (perp_smooth.max() + 1e-6)
    for j in range(1, prof_w):
        y1 = prof_h - int(norm[j - 1] * (prof_h - 10))
        y2 = prof_h - int(norm[j] * (prof_h - 10))
        cv2.line(pvis, (j - 1, y1), (j, y2), (0, 255, 0), 1)
    cv2.line(pvis, (mid, 0), (mid, prof_h), (255, 255, 255), 1)
    cv2.line(pvis, (peak_idx, 0), (peak_idx, prof_h), (0, 255, 255), 1)
    cv2.line(pvis, (left_edge, 0), (left_edge, prof_h), (0, 0, 255), 1)
    cv2.line(pvis, (right_edge, 0), (right_edge, prof_h), (0, 0, 255), 1)
    # 60% threshold line
    thresh_y = prof_h - int(0.6 * (prof_h - 10))
    cv2.line(pvis, (0, thresh_y), (prof_w, thresh_y), (80, 80, 80), 1)
    cv2.putText(pvis, f"S{si} w={strip_width_px} shift={peak_shift}",
                (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    save(out, f"strip_{si}_perp.jpg", pvis)

save(out, "00_all_strips_edges.jpg", vis_all)
print(f"\nDone! Check {out}/")
