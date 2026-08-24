"""
18 — Blur Grid: Multi-scale blur → strip boundaries → plug grid
================================================================
σ=15 blur → binary → white channels = strip boundaries
σ=5  blur → per-strip binary → individual plug blobs → grid fit

No density profiles, no peak finding, no valley detection.
The blur IS the feature extraction.
"""

from common import load_image, make_output_dir, save, crop_board, remove_red_overlay

import cv2
import numpy as np

img, gray = load_image()
out = make_output_dir("18_blur_grid")

board = crop_board(remove_red_overlay(img), board_idx=0)
board_gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
bh, bw = board_gray.shape
print(f"  Board: {bw}x{bh}")

inv = (255 - board_gray).astype(np.float32)

# ═══════════════════════════════════════════════════════════
# STAGE 1: σ=12 blur → binary → connected components = strips
#           (crop to plug region first — edges merge otherwise)
# ═══════════════════════════════════════════════════════════
sigma_strips = 12
ksize = int(sigma_strips * 6) | 1
inv_crop = inv[y_top:y_bot, :]
blurred_strips = cv2.GaussianBlur(inv_crop, (ksize, ksize), sigma_strips)

# Binary via Otsu on the blurred cropped region
blur_u8 = ((blurred_strips - blurred_strips.min()) /
           (blurred_strips.max() - blurred_strips.min() + 1e-6) * 255
           ).astype(np.uint8)
_, binary_strips = cv2.threshold(blur_u8, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
save(out, "00_blur_binary.jpg", binary_strips)

# Connected components — each blob is a strip column
n_cc, cc_labels, cc_stats, cc_cents = cv2.connectedComponentsWithStats(
    binary_strips, connectivity=8)

crop_h = y_bot - y_top
strip_ccs = []
for i in range(1, n_cc):
    h = cc_stats[i, cv2.CC_STAT_HEIGHT]
    w = cc_stats[i, cv2.CC_STAT_WIDTH]
    area = cc_stats[i, cv2.CC_STAT_AREA]
    # Strips are tall (>40% of crop) and not too wide (<20% of board)
    if h > crop_h * 0.4 and w < bw * 0.2 and area > 500:
        x0 = cc_stats[i, cv2.CC_STAT_LEFT]
        x1 = x0 + w
        strip_ccs.append((x0, x1, i))

strip_ccs.sort(key=lambda s: s[0])
strip_regions = [(x0, x1) for x0, x1, _ in strip_ccs]

print(f"  σ={sigma_strips}: {len(strip_regions)} strips")
for i, (x0, x1) in enumerate(strip_regions):
    print(f"    S{i}: x={x0}–{x1} (w={x1-x0})")

# Visualize strip boundaries
vis_strips = board.copy()
colors = [
    (80, 80, 255), (80, 255, 80), (255, 140, 40),
    (255, 255, 40), (40, 255, 255), (255, 40, 255),
    (160, 255, 40), (255, 160, 40), (40, 160, 255),
]
for i, (x0, x1) in enumerate(strip_regions):
    c = colors[i % len(colors)]
    cv2.line(vis_strips, (x0, 0), (x0, bh), c, 2)
    cv2.line(vis_strips, (x1, 0), (x1, bh), c, 2)
    cv2.putText(vis_strips, f"S{i}", (x0 + 3, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)
save(out, "01_strip_boundaries.jpg", vis_strips)

# ═══════════════════════════════════════════════════════════
# STAGE 2: σ=5 blur → per-strip plug detection
# ═══════════════════════════════════════════════════════════
sigma_plugs = 5
ksize2 = int(sigma_plugs * 6) | 1
blurred_plugs = cv2.GaussianBlur(inv, (ksize2, ksize2), sigma_plugs)

y_top = int(bh * 0.03)
y_bot = int(bh * 0.92)

vis_grid = board.copy()


def linreg(x, y):
    """OLS: y = offset + slope*x."""
    n = len(x)
    d = n * (x * x).sum() - x.sum()**2
    if abs(d) < 1e-6:
        return float(y.mean()), 0.0
    slope = (n * (x * y).sum() - x.sum() * y.sum()) / d
    offset = (y.sum() - slope * x.sum()) / n
    return offset, slope


# Get global spacing estimate first
all_gaps = []

for si, (x0, x1) in enumerate(strip_regions):
    # Row-mean profile within this strip from the blurred image
    strip_blur = blurred_plugs[y_top:y_bot, x0:x1]
    row_profile = np.mean(strip_blur, axis=1)

    # Threshold at mean of this strip's profile
    row_mean = row_profile.mean()
    above = row_profile > row_mean

    # Find plug blobs: runs of above-mean rows
    plugs_y = []
    start = None
    for y in range(len(row_profile)):
        if above[y] and start is None:
            start = y
        elif not above[y] and start is not None:
            if y - start > 3:  # minimum plug height
                cy = (start + y) / 2 + y_top  # center in board coords
                plugs_y.append(cy)
            start = None
    if start is not None and len(row_profile) - start > 3:
        plugs_y.append((start + len(row_profile)) / 2 + y_top)

    if len(plugs_y) > 2:
        gaps = np.diff(plugs_y)
        all_gaps.extend(gaps.tolist())

gsp = float(np.median(all_gaps)) if all_gaps else 32.0
print(f"\n  Global spacing (from blur plugs): {gsp:.1f}px")

# ── Per-strip: detect plugs from blur, fit grid ─────────
strip_fits = []

for si, (x0, x1) in enumerate(strip_regions):
    color = colors[si % len(colors)]

    strip_blur = blurred_plugs[y_top:y_bot, x0:x1]
    row_profile = np.mean(strip_blur, axis=1)
    row_mean = row_profile.mean()
    above = row_profile > row_mean

    # Find plug centers from runs of above-mean rows
    plugs_y = []
    start = None
    for y in range(len(row_profile)):
        if above[y] and start is None:
            start = y
        elif not above[y] and start is not None:
            if y - start > 3:
                plugs_y.append((start + y) / 2 + y_top)
            start = None
    if start is not None and len(row_profile) - start > 3:
        plugs_y.append((start + len(row_profile)) / 2 + y_top)

    plug_ys = np.array(plugs_y)

    if len(plug_ys) < 3:
        print(f"  S{si}: only {len(plug_ys)} plugs — skip")
        strip_fits.append(None)
        continue

    # RANSAC grid fit
    best_off, best_sp, best_inl = 0.0, gsp, np.array([], dtype=int)
    for a in range(min(len(plug_ys), 5)):
        idx = np.round((plug_ys - plug_ys[a]) / gsp).astype(int)
        off, sp = linreg(idx.astype(float), plug_ys)
        if sp < gsp * 0.85 or sp > gsp * 1.15:
            continue
        resid = np.abs(plug_ys - (off + idx * sp))
        inl = np.where(resid < 0.25 * sp)[0]
        if len(inl) > len(best_inl):
            xi, yi = idx[inl].astype(float), plug_ys[inl]
            best_off, best_sp = linreg(xi, yi)
            best_inl = inl

    inl_ys = plug_ys[best_inl]
    if len(inl_ys) < 2:
        strip_fits.append(None)
        continue

    # Grid: first to last inlier
    i0 = round((inl_ys.min() - best_off) / best_sp)
    i1 = round((inl_ys.max() - best_off) / best_sp)
    grid_ys = [best_off + j * best_sp for j in range(i0, i1 + 1)
               if y_top <= best_off + j * best_sp <= y_bot]

    resids = [min(abs(p - g) for g in grid_ys) for p in inl_ys]
    rms = np.sqrt(np.mean(np.array(resids)**2))
    n_out = len(plug_ys) - len(best_inl)

    strip_fits.append({
        "x0": x0, "x1": x1, "spacing": best_sp,
        "grid_ys": grid_ys, "rms": rms, "n_plugs": len(plug_ys),
        "n_out": n_out,
    })

    print(f"  S{si}: {len(plug_ys)} plugs ({n_out}out) → "
          f"{len(grid_ys)} grid, sp={best_sp:.2f}px, rms={rms:.1f}px")

    # Draw strip boundaries
    cv2.line(vis_grid, (x0, 0), (x0, bh), color, 2)
    cv2.line(vis_grid, (x1, 0), (x1, bh), color, 2)

    # Draw grid lines
    for gy in grid_ys:
        cv2.line(vis_grid, (x0 + 2, int(round(gy))),
                 (x1 - 2, int(round(gy))), color, 1)

    # Mark plug centers: yellow=inlier, red=outlier
    cx_strip = (x0 + x1) // 2
    for pi, py in enumerate(plug_ys):
        c = (0, 255, 255) if pi in best_inl else (0, 0, 255)
        cv2.circle(vis_grid, (cx_strip, int(py)), 4, c, -1)

    cv2.putText(vis_grid, f"S{si}:{len(grid_ys)}",
                (x0 + 2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

save(out, "02_blur_grid.jpg", vis_grid)

# ── Summary ───────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  BLUR GRID — σ_strip={sigma_strips}, σ_plug={sigma_plugs}")
print(f"  {len(strip_regions)} strips, gsp={gsp:.1f}px")
for si, sf in enumerate(strip_fits):
    if sf is None:
        continue
    print(f"    S{si}: {len(sf['grid_ys'])} lines, "
          f"sp={sf['spacing']:.2f}px, rms={sf['rms']:.1f}px, "
          f"out={sf['n_out']}")
print(f"\nDone! Check {out}/")
