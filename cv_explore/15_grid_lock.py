"""
15 — Grid Lock: Precise evenly-spaced grid on cotton plugs
===========================================================
Evenly-spaced linear grid fitted to each strip's plug positions.
RANSAC outlier rejection + global spacing prior + edge trimming.
"""

from common import load_image, make_output_dir, save, crop_board, remove_red_overlay

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

img, gray = load_image()
out = make_output_dir("15_grid_lock")

board = crop_board(remove_red_overlay(img), board_idx=0)
board_gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
bh, bw = board_gray.shape
print(f"  Board: {bw}x{bh}")

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
inv = 255 - clahe.apply(board_gray)

# Trim top 3% and bottom 8% to avoid board-edge false peaks
y_top = int(bh * 0.03)
y_bot = int(bh * 0.92)

colors = [
    (80, 80, 255), (80, 255, 80), (255, 140, 40),
    (255, 255, 40), (40, 255, 255), (255, 40, 255),
    (160, 255, 40), (255, 160, 40), (40, 160, 255),
]

# ── Strip boundaries ───────────────────────────────────────
v_profile = gaussian_filter1d(np.mean(inv[y_top:y_bot, :], axis=0), sigma=5)
strip_peaks_x, _ = find_peaks(v_profile, distance=bw // 18, prominence=5)
strip_valleys, _ = find_peaks(-v_profile, distance=bw // 18, prominence=3)

all_bounds = sorted(np.concatenate(([0], strip_valleys, [bw - 1])))
strip_regions = []
for i in range(len(all_bounds) - 1):
    x0, x1 = int(all_bounds[i]), int(all_bounds[i + 1])
    if x1 - x0 > 15 and any(x0 <= p <= x1 for p in strip_peaks_x):
        strip_regions.append((x0, x1))

print(f"  Strips: {len(strip_regions)}")

# ── Global spacing prior from all strips ───────────────────
all_spacings = []
for x0, x1 in strip_regions:
    h = gaussian_filter1d(np.mean(inv[y_top:y_bot, x0:x1], axis=1), sigma=2)
    pks, _ = find_peaks(h, distance=bh // 50, prominence=3)
    if len(pks) > 3:
        all_spacings.extend(np.diff(pks).tolist())
global_sp = float(np.median(all_spacings)) if all_spacings else 32.0
print(f"  Global spacing: {global_sp:.1f}px")


def fit_grid_ransac(peaks, est_sp, tol=0.25):
    """Fit y(i) = offset + i*spacing with RANSAC outlier rejection.
    Uses np.round for index assignment — simple and stable.
    """
    best_off, best_sp = 0.0, est_sp
    best_inl = np.array([], dtype=int)

    for anchor_idx in range(min(len(peaks), 5)):
        anchor = peaks[anchor_idx]
        indices = np.round((peaks - anchor) / est_sp).astype(int)

        x = indices.astype(float)
        y = peaks.astype(float)
        n = len(x)
        sx, sy, sxx, sxy = x.sum(), y.sum(), (x*x).sum(), (x*y).sum()
        denom = n * sxx - sx * sx
        if abs(denom) < 1e-6:
            continue
        sp = (n * sxy - sx * sy) / denom
        off = (sy - sp * sx) / n

        if sp < est_sp * 0.85 or sp > est_sp * 1.15:
            continue

        resid = np.abs(peaks - (off + indices * sp))
        inl = np.where(resid < tol * sp)[0]

        if len(inl) > len(best_inl):
            xi, yi = x[inl], y[inl]
            ni = len(xi)
            d2 = ni * (xi*xi).sum() - xi.sum()**2
            if abs(d2) > 1e-6:
                best_sp = (ni*(xi*yi).sum() - xi.sum()*yi.sum()) / d2
                best_off = (yi.sum() - best_sp * xi.sum()) / ni
            best_inl = inl

    return best_off, best_sp, best_inl


# ── Per-strip: detect, fit, draw ───────────────────────────
vis_grid = board.copy()
strip_fits = []

for si, (x0, x1) in enumerate(strip_regions):
    color = colors[si % len(colors)]

    # Density profile in trimmed region
    h_smooth = gaussian_filter1d(
        np.mean(inv[y_top:y_bot, x0:x1], axis=1), sigma=2)
    raw_local, _ = find_peaks(h_smooth, distance=bh // 50, prominence=3)
    raw_peaks = raw_local + y_top

    if len(raw_peaks) < 3:
        strip_fits.append(None)
        continue

    offset, spacing, inliers = fit_grid_ransac(raw_peaks, global_sp)
    inlier_peaks = raw_peaks[inliers]

    if len(inlier_peaks) < 2:
        strip_fits.append(None)
        continue

    # Grid covers first to last inlier — no extension
    i_first = round((inlier_peaks.min() - offset) / spacing)
    i_last = round((inlier_peaks.max() - offset) / spacing)
    grid_ys = [offset + i * spacing for i in range(i_first, i_last + 1)
               if y_top <= offset + i * spacing <= y_bot]

    resids = [min(abs(p - g) for g in grid_ys) for p in inlier_peaks]
    rms = np.sqrt(np.mean(np.array(resids) ** 2))
    n_out = len(raw_peaks) - len(inliers)

    strip_fits.append({
        "x0": x0, "x1": x1, "offset": offset, "spacing": spacing,
        "grid_ys": grid_ys, "rms": rms, "raw_peaks": raw_peaks,
        "inliers": inliers,
    })

    print(f"  S{si}: {len(raw_peaks)}pk ({n_out}out) → "
          f"{len(grid_ys)} grid, sp={spacing:.1f}px, rms={rms:.1f}px")

    # Draw strip boundaries
    cv2.line(vis_grid, (x0, 0), (x0, bh), color, 2)
    cv2.line(vis_grid, (x1, 0), (x1, bh), color, 2)

    # Draw evenly-spaced grid lines
    for gy in grid_ys:
        cv2.line(vis_grid, (x0 + 2, int(round(gy))),
                 (x1 - 2, int(round(gy))), color, 1)

    # Dots: white=inlier, red=outlier
    cx = (x0 + x1) // 2
    for pi, p in enumerate(raw_peaks):
        c = (255, 255, 255) if pi in inliers else (0, 0, 255)
        cv2.circle(vis_grid, (cx, int(p)), 3, c, -1)

    cv2.putText(vis_grid, f"S{si}:{len(grid_ys)}",
                (x0 + 2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

save(out, "01_grid_locked.jpg", vis_grid)

# ── Summary ────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  GRID LOCK — {len(strip_regions)} strips, gsp={global_sp:.1f}px")
for si, sf in enumerate(strip_fits):
    if sf is None:
        continue
    n_out = len(sf['raw_peaks']) - len(sf['inliers'])
    print(f"    S{si}: {len(sf['grid_ys'])} lines, "
          f"sp={sf['spacing']:.1f}px, rms={sf['rms']:.1f}px, out={n_out}")
print(f"\nDone! Check {out}/")
