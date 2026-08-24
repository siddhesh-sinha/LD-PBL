"""
15 — Grid Lock: Precise evenly-spaced grid on cotton plugs
===========================================================
Strategy: use BLOB DETECTION (connectedComponents) to find actual
cotton plug centroids — not density profile peaks which hallucinate.
Then fit evenly-spaced linear grid to the blob centroids per strip.

Blob size filter: plugs are 200+ px², flies/noise are <100 px².
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

y_top = int(bh * 0.03)
y_bot = int(bh * 0.92)

colors = [
    (80, 80, 255), (80, 255, 80), (255, 140, 40),
    (255, 255, 40), (40, 255, 255), (255, 40, 255),
    (160, 255, 40), (255, 160, 40), (40, 160, 255),
]

# ── Strip boundaries (density profile — this part works fine) ──
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

# ── Blob detection per strip ───────────────────────────────
MIN_PLUG_AREA = 200  # plugs are 200+ px², flies are <100

def find_plug_centroids(strip_gray_region):
    """Find cotton plug centroids via connected components."""
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
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= MIN_PLUG_AREA:
            plugs.append((centroids[i][1], centroids[i][0], area))  # y, x, area
    plugs.sort(key=lambda p: p[0])  # sort by y
    return plugs


def linreg(x, y):
    """OLS: y = offset + slope*x."""
    n = len(x)
    d = n * (x*x).sum() - x.sum()**2
    if abs(d) < 1e-6:
        return float(y.mean()), 0.0
    slope = (n*(x*y).sum() - x.sum()*y.sum()) / d
    offset = (y.sum() - slope*x.sum()) / n
    return offset, slope


def fit_grid_ransac(ys, est_sp, tol=0.25):
    """RANSAC fit: y(i) = offset + i*spacing."""
    ys = np.array(ys, dtype=float)
    best_off, best_sp = 0.0, est_sp
    best_inl = np.array([], dtype=int)

    for a in range(min(len(ys), 5)):
        idx = np.round((ys - ys[a]) / est_sp).astype(int)
        off, sp = linreg(idx.astype(float), ys)
        if sp < est_sp * 0.85 or sp > est_sp * 1.15:
            continue
        resid = np.abs(ys - (off + idx * sp))
        inl = np.where(resid < tol * sp)[0]
        if len(inl) > len(best_inl):
            xi, yi = idx[inl].astype(float), ys[inl]
            best_off, best_sp = linreg(xi, yi)
            best_inl = inl
    return best_off, best_sp, best_inl


# ── Global spacing from all strips' blob centroids ─────────
all_gaps = []
for x0, x1 in strip_regions:
    plugs = find_plug_centroids(board_gray[y_top:y_bot, x0:x1])
    if len(plugs) > 3:
        plug_ys = [p[0] for p in plugs]
        all_gaps.extend(np.diff(plug_ys).tolist())
gsp = float(np.median(all_gaps)) if all_gaps else 32.0
print(f"  Global spacing (from blobs): {gsp:.1f}px")

# ── Per-strip: detect blobs, fit grid, draw ────────────────
vis_grid = board.copy()
strip_fits = []

for si, (x0, x1) in enumerate(strip_regions):
    color = colors[si % len(colors)]

    plugs = find_plug_centroids(board_gray[y_top:y_bot, x0:x1])
    plug_ys = np.array([p[0] + y_top for p in plugs])  # back to full coords

    if len(plug_ys) < 3:
        print(f"  S{si}: only {len(plug_ys)} plugs — skip")
        strip_fits.append(None)
        continue

    offset, spacing, inliers = fit_grid_ransac(plug_ys, gsp)
    inl_ys = plug_ys[inliers]

    if len(inl_ys) < 2:
        strip_fits.append(None)
        continue

    # Grid: first to last inlier only
    i0 = round((inl_ys.min() - offset) / spacing)
    i1 = round((inl_ys.max() - offset) / spacing)
    grid_ys = [offset + i * spacing for i in range(i0, i1 + 1)
               if y_top <= offset + i * spacing <= y_bot]

    resids = [min(abs(p - g) for g in grid_ys) for p in inl_ys]
    rms = np.sqrt(np.mean(np.array(resids)**2))
    n_out = len(plug_ys) - len(inliers)

    strip_fits.append({
        "x0": x0, "x1": x1, "offset": offset, "spacing": spacing,
        "grid_ys": grid_ys, "rms": rms, "plug_ys": plug_ys,
        "inliers": inliers,
    })

    print(f"  S{si}: {len(plug_ys)} plugs ({n_out}out) → "
          f"{len(grid_ys)} grid, sp={spacing:.2f}px, rms={rms:.1f}px")

    # Draw strip boundaries
    cv2.line(vis_grid, (x0, 0), (x0, bh), color, 2)
    cv2.line(vis_grid, (x1, 0), (x1, bh), color, 2)

    # Draw grid lines
    for gy in grid_ys:
        cv2.line(vis_grid, (x0 + 2, int(round(gy))),
                 (x1 - 2, int(round(gy))), color, 1)

    # Draw blob centroids: yellow=inlier, red=outlier
    for pi, py in enumerate(plug_ys):
        cx_blob = x0 + int(plugs[pi][1])  # blob x within strip + strip offset
        c = (0, 255, 255) if pi in inliers else (0, 0, 255)
        cv2.circle(vis_grid, (cx_blob, int(py)), 4, c, -1)

    cv2.putText(vis_grid, f"S{si}:{len(grid_ys)}",
                (x0 + 2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

save(out, "01_grid_locked.jpg", vis_grid)

# ── Summary ────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  GRID LOCK — {len(strip_regions)} strips, gsp={gsp:.1f}px")
for si, sf in enumerate(strip_fits):
    if sf is None:
        continue
    n_out = len(sf['plug_ys']) - len(sf['inliers'])
    print(f"    S{si}: {len(sf['grid_ys'])} lines, "
          f"sp={sf['spacing']:.2f}px, rms={sf['rms']:.1f}px, out={n_out}")
print(f"\nDone! Check {out}/")
