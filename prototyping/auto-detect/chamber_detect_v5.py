"""
DrosoLab Chamber Auto-Detection — R&D Prototype v5 (FINAL)
===========================================================
Combines best of v1-v4:
  - Red/blue overlay inpainting (v3)
  - Board detection via contour (v1)
  - Column detection: autocorrelation with TIGHT constraint (~9 cols) (v3/v4)
  - Row detection: fly blob histogram (v4) — ALWAYS prefer histogram peaks
  - Grid regularization: extend using median spacing (v4)
  - Final output: numbered cell maps, ghost grid, comparison overlay
"""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, correlate
import os

IMG_PATH = "/root/.claude/uploads/608e2092-c86e-5429-b039-27e3dec388ed/857fc50f-image.jpg"
OUT_DIR = "/tmp/claude-0/-home-user-LD-PBL/608e2092-c86e-5429-b039-27e3dec388ed/scratchpad/cv_output_v5"
os.makedirs(OUT_DIR, exist_ok=True)

img = cv2.imread(IMG_PATH)
h, w = img.shape[:2]
print(f"Image: {w}x{h}")

# ─── Remove overlay ──────────────────────────────────────────────────────────
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask_overlay = (cv2.inRange(hsv, (0, 100, 100), (10, 255, 255)) |
                cv2.inRange(hsv, (160, 100, 100), (180, 255, 255)) |
                cv2.inRange(hsv, (100, 100, 100), (130, 255, 255)))
mask_overlay = cv2.dilate(mask_overlay, np.ones((5, 5), np.uint8), iterations=2)
img_clean = cv2.inpaint(img, mask_overlay, 7, cv2.INPAINT_TELEA)

gray = cv2.cvtColor(img_clean, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
enhanced = clahe.apply(gray)

# ─── Find boards ─────────────────────────────────────────────────────────────
_, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
morph = cv2.morphologyEx(
    cv2.morphologyEx(otsu, cv2.MORPH_CLOSE,
                     cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))),
    cv2.MORPH_OPEN,
    cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
)
contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
board_rects = sorted(
    [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > h * w * 0.03],
    key=lambda b: (0 if b[1] < h // 2 else 1, b[0])
)
print(f"Boards: {len(board_rects)}")

# ─── Helper functions ────────────────────────────────────────────────────────

def autocorr_best_period(signal, min_p, max_p):
    """Find dominant period via autocorrelation."""
    sig = signal - np.mean(signal)
    acf = correlate(sig, sig, mode='full')[len(sig)-1:]
    if acf[0] > 0:
        acf /= acf[0]
    lo, hi = min_p, min(max_p + 1, len(acf))
    if lo >= hi:
        return None
    peaks, props = find_peaks(acf[lo:hi], height=0.05, prominence=0.03)
    if not len(peaks):
        return None
    best = peaks[np.argmax(props['peak_heights'])] + lo
    return int(best)


def fit_grid_offset(signal, spacing):
    """Find best phase offset for a known spacing."""
    best_off, best_sc = 0, -1
    for off in range(spacing):
        pos = np.arange(off, len(signal), spacing)
        sc = sum(np.max(signal[max(0,p-2):min(len(signal),p+3)]) for p in pos)
        if sc > best_sc:
            best_sc = sc
            best_off = off
    return best_off


def detect_fly_centroids(region_bgr):
    """Detect dark blobs (flies) and return their (x, y) centroids."""
    lab = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2LAB)
    l_ch = lab[:, :, 0]
    _, mask = cv2.threshold(l_ch, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts = []
    for c in contours:
        a = cv2.contourArea(c)
        if 30 < a < 2000:
            M = cv2.moments(c)
            if M["m00"] > 0:
                pts.append((int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])))
    return pts


def regularize_and_extend(positions, region_start, region_end, min_count=15):
    """
    Given detected positions, compute median spacing, fill gaps,
    extend to cover the full region.
    """
    if len(positions) < 3:
        return positions

    positions = sorted(positions)

    # Compute median spacing
    spacings = np.diff(positions)
    median_sp = np.median(spacings)

    # Remove positions that are too close (< 0.5× median)
    cleaned = [positions[0]]
    for p in positions[1:]:
        if p - cleaned[-1] > median_sp * 0.5:
            cleaned.append(p)
    positions = cleaned

    # Fill internal gaps (> 1.5× median)
    filled = [positions[0]]
    for p in positions[1:]:
        gap = p - filled[-1]
        if gap > median_sp * 1.5:
            n = round(gap / median_sp)
            step = gap / n
            for j in range(1, n):
                filled.append(int(filled[-1] + step))
        filled.append(p)
    positions = filled

    # Recompute median after filling
    spacings = np.diff(positions)
    median_sp = np.median(spacings)

    # Extend upward
    while positions[0] - median_sp >= region_start:
        positions.insert(0, int(positions[0] - median_sp))

    # Extend downward
    while positions[-1] + median_sp <= region_end:
        positions.append(int(positions[-1] + median_sp))

    return positions


# ─── Process boards ──────────────────────────────────────────────────────────

board_data = []
colors_bgr = [(0, 255, 0), (255, 180, 0), (0, 200, 255), (255, 0, 200)]

for idx, (bx, by, bw, bh) in enumerate(board_rects):
    print(f"\n{'='*55}")
    print(f"Board {idx+1}: pos=({bx},{by}) size={bw}×{bh}")

    board_bgr = img_clean[by:by+bh, bx:bx+bw]
    board_enh = enhanced[by:by+bh, bx:bx+bw]

    # Working region (skip plugs at top, edge at bottom)
    yt, yb = int(bh * 0.06), int(bh * 0.97)
    work_enh = board_enh[yt:yb, :]
    work_bgr = board_bgr[yt:yb, :]
    wh, ww = work_enh.shape[:2]

    # ── COLUMNS ──
    edges = cv2.Canny(work_enh, 30, 100)
    v_proj = gaussian_filter1d(np.sum(edges, axis=0).astype(float) / 255, sigma=2)

    # TIGHT constraint: boards have 9-10 columns → 10-11 boundaries
    # Expected spacing = bw / 9.5 ≈ 100-125px
    target_sp = bw / 9.5
    min_cs = int(target_sp * 0.7)
    max_cs = int(target_sp * 1.3)
    col_sp = autocorr_best_period(v_proj, min_cs, max_cs)

    if col_sp:
        col_off = fit_grid_offset(v_proj, col_sp)
        col_pos = [c for c in range(col_off, ww, col_sp) if 3 < c < ww - 3]
    else:
        # Fallback: divide into 9 equal columns
        col_sp = bw // 9
        col_pos = [int(i * bw / 10) for i in range(1, 10)]

    print(f"  Columns: spacing={col_sp}px, {len(col_pos)} boundaries")

    # ── ROWS via fly detection ──
    flies = detect_fly_centroids(work_bgr)
    print(f"  Flies detected: {len(flies)}")

    if len(flies) >= 10:
        fly_ys = sorted([cy for _, cy in flies])

        # Histogram of fly y-positions
        n_bins = wh // 4  # ~4px per bin
        y_hist, bin_edges = np.histogram(fly_ys, bins=n_bins, range=(0, wh))
        y_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        y_hist_sm = gaussian_filter1d(y_hist.astype(float), sigma=1.5)

        # Find peaks = row centers
        # Expected ~20 rows → min distance ~wh/25
        min_dist_bins = max(3, wh // 25 // 4)  # in bin units
        peaks, _ = find_peaks(y_hist_sm,
                               height=max(1, y_hist_sm.max() * 0.08),
                               distance=min_dist_bins)
        row_centers = [int(y_centers[p]) for p in peaks]
        print(f"  Raw row peaks from flies: {len(row_centers)}")

        # Regularize: fill gaps, extend
        row_pos = regularize_and_extend(row_centers, 0, wh)
        # Offset to board coords
        row_pos = [r + yt for r in row_pos]
    else:
        # Fallback: try edge-based row detection
        h_proj = gaussian_filter1d(np.sum(edges, axis=1).astype(float) / 255, sigma=2)
        target_rsp = wh / 20
        row_sp = autocorr_best_period(h_proj, int(target_rsp * 0.6), int(target_rsp * 1.5))
        if row_sp:
            row_off = fit_grid_offset(h_proj, row_sp)
            row_pos = [r + yt for r in range(row_off, wh, row_sp)]
        else:
            row_pos = []

    print(f"  Final rows: {len(row_pos)}")

    n_cols = max(0, len(col_pos) - 1)
    n_rows = max(0, len(row_pos) - 1)
    cells = n_cols * n_rows
    print(f"  → {n_cols} cols × {n_rows} rows = {cells} cells")

    board_data.append({
        'rect': (bx, by, bw, bh),
        'cols': col_pos,
        'rows': row_pos,
        'yt': yt, 'yb': yb,
        'flies': flies,
        'n_cells': cells,
    })

# ─── Visualizations ──────────────────────────────────────────────────────────

print(f"\n{'='*55}")
print("Drawing outputs...")

# 1. Grid on original
grid_orig = img.copy()
# 2. Grid on cleaned
grid_clean = img_clean.copy()
# 3. Ghost
ghost = np.zeros_like(img)
# 4. Fly detection overlay
fly_img = img_clean.copy()

for idx, bd in enumerate(board_data):
    bx, by, bw, bh = bd['rect']
    yt, yb = bd['yt'], bd['yb']
    col = colors_bgr[idx % 4]
    dim = tuple(c // 2 for c in col)

    for tgt in [grid_orig, grid_clean, ghost]:
        cv2.rectangle(tgt, (bx, by), (bx + bw, by + bh), col, 2)
        for cx in bd['cols']:
            cv2.line(tgt, (bx + cx, by + yt), (bx + cx, by + yb), col, 2)
        for ry in bd['rows']:
            cv2.line(tgt, (bx, by + ry), (bx + bw, by + ry), dim, 1)

    nc = max(0, len(bd['cols']) - 1)
    nr = max(0, len(bd['rows']) - 1)
    label = f"B{idx+1}: {nc}c x {nr}r = {nc*nr}"
    for tgt in [grid_orig, grid_clean]:
        cv2.putText(tgt, label, (bx + 5, by - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

    # Flies
    for fx, fy in bd['flies']:
        cv2.circle(fly_img, (bx + fx, by + yt + fy), 4, col, -1)

cv2.imwrite(os.path.join(OUT_DIR, "20_grid_on_original.jpg"), grid_orig)
cv2.imwrite(os.path.join(OUT_DIR, "21_grid_on_cleaned.jpg"), grid_clean)
cv2.imwrite(os.path.join(OUT_DIR, "22_ghost_grid.jpg"), ghost)
cv2.imwrite(os.path.join(OUT_DIR, "23_flies_detected.jpg"), fly_img)

# Per-board cell maps (on cleaned image)
for bi, bd in enumerate(board_data):
    bx, by, bw, bh = bd['rect']
    col = colors_bgr[bi % 4]
    crop = img_clean[by:by+bh, bx:bx+bw].copy()

    cell = 0
    for ci in range(len(bd['cols']) - 1):
        xl, xr = bd['cols'][ci], bd['cols'][ci + 1]
        for ri in range(len(bd['rows']) - 1):
            rt, rb = bd['rows'][ri], bd['rows'][ri + 1]
            cv2.rectangle(crop, (xl, rt), (xr, rb), col, 1)
            cell += 1
            cx = (xl + xr) // 2 - 8
            cy = (rt + rb) // 2 + 4
            cv2.putText(crop, str(cell), (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

    cv2.imwrite(os.path.join(OUT_DIR, f"30_board{bi+1}_cells.jpg"), crop)

# ─── Comparison montage: original vs detected grid ───────────────────────────
# For each board, put original crop on left, grid crop on right

for bi, bd in enumerate(board_data):
    bx, by, bw, bh = bd['rect']
    col = colors_bgr[bi % 4]
    dim = tuple(c // 2 for c in col)

    orig_crop = img[by:by+bh, bx:bx+bw].copy()
    grid_crop = img_clean[by:by+bh, bx:bx+bw].copy()

    # Draw grid on right half
    for cx in bd['cols']:
        cv2.line(grid_crop, (cx, bd['yt']), (cx, bd['yb']), col, 2)
    for ry in bd['rows']:
        cv2.line(grid_crop, (0, ry), (bw, ry), dim, 1)

    # Stack horizontally with a divider
    divider = np.full((bh, 4, 3), 200, dtype=np.uint8)
    montage = np.hstack([orig_crop, divider, grid_crop])

    # Labels
    cv2.putText(montage, "Original", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(montage, "Auto-detected grid", (bw + 14, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)

    cv2.imwrite(os.path.join(OUT_DIR, f"40_board{bi+1}_comparison.jpg"), montage)

# ─── Summary ────────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print("FINAL RESULTS (v5)")
print(f"{'='*60}")
total = 0
for i, bd in enumerate(board_data):
    nc = max(0, len(bd['cols']) - 1)
    nr = max(0, len(bd['rows']) - 1)
    cells = nc * nr
    total += cells
    print(f"  Board {i+1}: {nc} cols × {nr} rows = {cells:>4d} cells  "
          f"({len(bd['flies'])} flies, col_sp≈{np.mean(np.diff(bd['cols'])):.0f}px"
          f", row_sp≈{np.mean(np.diff(bd['rows'])):.0f}px)")

print(f"\n  TOTAL: {total} cells")
print(f"  Expected (Utpal's labels): ~580")
print(f"  Accuracy: {total/580*100:.0f}%")
print(f"\n  Outputs: {OUT_DIR}")
