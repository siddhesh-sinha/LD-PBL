"""
13 — Edge Density Scan (Border-Inward Detection)
==================================================
Instead of detecting individual plugs/blobs, scan from the EDGES
of the board rectangle inward. The dark-vs-light density profile
tells us where strips are.

  - From the top/bottom edges: scan downward/upward,
    horizontal density profile → finds strip BOUNDARIES (vertical)
  - From the left/right edges: scan inward,
    vertical density profile → finds tube BOUNDARIES (horizontal)

Constraints:
  - Horizontal tube boundaries don't cross vertical strip walls
  - Look for traces of real horizontal lines within each strip
    to confirm tube alignment

This is simpler and more robust than blob detection —
works on ANY image quality because we only need aggregate
density, not individual features.
"""

from common import load_image, make_output_dir, save, crop_board, remove_red_overlay

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

img, gray = load_image()
out = make_output_dir("13_edge_density")

board = crop_board(remove_red_overlay(img), board_idx=0)
board_gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
bh, bw = board_gray.shape
print(f"  Board: {bw}x{bh}")

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
enhanced = clahe.apply(board_gray)

# ============================================================
# STEP 1: Vertical density profile (scan left→right)
#         Each column's mean darkness → strip positions
# ============================================================

# Invert so dark pixels = high values
inv = 255 - enhanced

# Vertical projection: mean intensity per column (across all rows)
v_profile = np.mean(inv, axis=0)
v_smooth = gaussian_filter1d(v_profile, sigma=5)

# Also compute the profile at different vertical slices
# (top 20%, middle, bottom 20% — each might show different things)
top_slice = inv[:int(bh * 0.2), :]
mid_slice = inv[int(bh * 0.3):int(bh * 0.7), :]
bot_slice = inv[int(bh * 0.8):, :]

v_top = gaussian_filter1d(np.mean(top_slice, axis=0), sigma=5)
v_mid = gaussian_filter1d(np.mean(mid_slice, axis=0), sigma=5)
v_bot = gaussian_filter1d(np.mean(bot_slice, axis=0), sigma=5)


def draw_profile_h(profile, width, height=150, color=(0, 255, 0),
                   peaks=None, label=""):
    """Draw a horizontal profile (value vs x-position)."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    norm = profile / (profile.max() + 1e-6)
    pts = [(x, height - int(norm[x] * (height - 20)))
           for x in range(min(len(norm), width))]
    for i in range(1, len(pts)):
        cv2.line(img, pts[i - 1], pts[i], color, 1, cv2.LINE_AA)
    if peaks is not None:
        for p in peaks:
            if p < width:
                cv2.line(img, (p, 0), (p, height), (0, 0, 255), 1)
    if label:
        cv2.putText(img, label, (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return img


# Find peaks in vertical profile = strip centers (dense dark regions)
# Find valleys = gaps between strips (bright areas)
v_peaks, v_peak_props = find_peaks(
    v_smooth, distance=bw // 20, prominence=5
)
v_valleys, _ = find_peaks(
    -v_smooth, distance=bw // 20, prominence=3
)

save(out, "01_v_profile_full.jpg",
     draw_profile_h(v_smooth, bw, 180, (0, 255, 0),
                    v_peaks, "Full height — peaks=strip centers"))
save(out, "01_v_profile_top.jpg",
     draw_profile_h(v_top, bw, 180, (255, 200, 0),
                    label="Top 20% (plug region?)"))
save(out, "01_v_profile_mid.jpg",
     draw_profile_h(v_mid, bw, 180, (0, 200, 255),
                    label="Middle 40%"))
save(out, "01_v_profile_bot.jpg",
     draw_profile_h(v_bot, bw, 180, (200, 0, 255),
                    label="Bottom 20%"))

print(f"\n  Vertical profile peaks (strip centers): {len(v_peaks)}")
print(f"    Positions: {v_peaks.tolist()}")
print(f"  Vertical profile valleys (gaps): {len(v_valleys)}")
print(f"    Positions: {v_valleys.tolist()}")


# ============================================================
# STEP 2: Find strip boundaries from the density profile
# ============================================================

# A strip is a region where density stays HIGH (plateaus)
# A gap is where density drops LOW
# Use the valleys as boundary markers between strips

# Add edges of the board as implicit boundaries
boundaries = sorted(np.concatenate(([0], v_valleys, [bw - 1])))
print(f"\n  Strip boundaries (x): {boundaries}")

# Each pair of consecutive boundaries = one strip (or gap)
# A strip has high average density, a gap has low
strips = []
for i in range(len(boundaries) - 1):
    x_start = int(boundaries[i])
    x_end = int(boundaries[i + 1])
    if x_end - x_start < 10:
        continue  # too narrow

    region_density = np.mean(v_smooth[x_start:x_end])
    overall_mean = np.mean(v_smooth)

    strips.append({
        "x_start": x_start,
        "x_end": x_end,
        "width": x_end - x_start,
        "density": region_density,
        "is_strip": region_density > overall_mean * 0.8,
    })

print(f"\n  Regions between boundaries:")
for s in strips:
    tag = "STRIP" if s["is_strip"] else "gap"
    print(f"    x={s['x_start']:4d}–{s['x_end']:4d} "
          f"(w={s['width']:3d}) "
          f"density={s['density']:.1f}  [{tag}]")

# Visualize strip boundaries on the board
vis_boundaries = board.copy()
for s in strips:
    if s["is_strip"]:
        # Draw strip region
        cv2.rectangle(vis_boundaries,
                      (s["x_start"], 0),
                      (s["x_end"], bh),
                      (0, 255, 0), 2)
    else:
        # Draw gap region (faint)
        cv2.rectangle(vis_boundaries,
                      (s["x_start"], 0),
                      (s["x_end"], bh),
                      (100, 100, 100), 1)

# Draw boundary lines
for b in boundaries[1:-1]:
    cv2.line(vis_boundaries, (int(b), 0), (int(b), bh),
             (0, 0, 255), 1)

save(out, "02_strip_boundaries.jpg", vis_boundaries)


# ============================================================
# STEP 3: Horizontal density profile (scan top→bottom)
#         Each row's mean darkness → tube boundaries
# ============================================================

# Horizontal projection: mean intensity per row
h_profile = np.mean(inv, axis=1)
h_smooth = gaussian_filter1d(h_profile, sigma=3)

# Find peaks (dense rows = rows with lots of flies/plugs)
# Find valleys (sparse rows = gaps between tubes)
h_peaks, _ = find_peaks(h_smooth, distance=bh // 30, prominence=3)
h_valleys, _ = find_peaks(-h_smooth, distance=bh // 30, prominence=2)

save(out, "03_h_profile.jpg",
     draw_profile_h(h_smooth, bh, 180, (0, 200, 255),
                    h_valleys, "Horizontal — valleys=tube gaps"))

print(f"\n  Horizontal profile valleys (tube gaps): {len(h_valleys)}")


# ============================================================
# STEP 4: Per-strip horizontal scanning
#         Within each strip, find tube boundaries independently
# ============================================================

actual_strips = [s for s in strips if s["is_strip"]]
print(f"\n  Processing {len(actual_strips)} strips individually...")

vis_tubes = board.copy()
vis_tubes = cv2.addWeighted(vis_tubes, 0.6,
                             np.zeros_like(vis_tubes), 0, 0)

colors = [
    (0, 0, 255), (0, 255, 0), (255, 100, 0),
    (255, 255, 0), (0, 255, 255), (255, 0, 255),
    (128, 255, 0), (255, 128, 0), (0, 128, 255),
    (128, 0, 255), (255, 0, 128), (0, 255, 128),
]

for si, strip in enumerate(actual_strips):
    color = colors[si % len(colors)]
    x0, x1 = strip["x_start"], strip["x_end"]

    # Extract this strip's region
    strip_region = inv[: , x0:x1]
    strip_w = x1 - x0

    # Horizontal density within this strip only
    h_local = np.mean(strip_region, axis=1)
    h_local_smooth = gaussian_filter1d(h_local, sigma=2)

    # Find local valleys = tube gaps within this strip
    local_valleys, _ = find_peaks(
        -h_local_smooth, distance=bh // 35, prominence=2
    )
    local_peaks, _ = find_peaks(
        h_local_smooth, distance=bh // 35, prominence=2
    )

    # Draw strip outline
    cv2.rectangle(vis_tubes, (x0, 0), (x1, bh), color, 2)

    # Draw horizontal tube boundaries within this strip
    for v in local_valleys:
        cv2.line(vis_tubes, (x0, v), (x1, v), color, 1)

    # Label
    cv2.putText(vis_tubes,
                f"S{si}: {len(local_valleys)} rows",
                (x0 + 2, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    print(f"    Strip {si} (x={x0}–{x1}): "
          f"{len(local_valleys)} tube boundaries, "
          f"{len(local_peaks)} tube centers")

    # Save per-strip profile
    save(out, f"04_strip{si}_h_profile.jpg",
         draw_profile_h(h_local_smooth, bh, 120, color,
                        local_valleys,
                        f"Strip {si} — {len(local_valleys)} boundaries"))

save(out, "05_all_tubes.jpg", vis_tubes)


# ============================================================
# STEP 5: Find actual horizontal LINE traces via edge detection
#         within each strip (for alignment confirmation)
# ============================================================

vis_hlines = board.copy()

for si, strip in enumerate(actual_strips):
    color = colors[si % len(colors)]
    x0, x1 = strip["x_start"], strip["x_end"]

    # Extract strip
    strip_gray = enhanced[: , x0:x1]

    # Sobel Y (horizontal edges) within the strip
    sobel_y = cv2.Sobel(strip_gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_y_abs = np.abs(sobel_y)
    sobel_y_norm = (sobel_y_abs / sobel_y_abs.max() * 255).astype(np.uint8)

    # Threshold strong horizontal edges
    _, strong_edges = cv2.threshold(
        sobel_y_norm, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Horizontal projection of edges within this strip
    h_edge_profile = np.sum(strong_edges, axis=1).astype(float) / 255
    h_edge_smooth = gaussian_filter1d(h_edge_profile, sigma=2)

    # Find peaks = rows with strong horizontal edges
    edge_peaks, _ = find_peaks(
        h_edge_smooth, distance=bh // 35,
        height=h_edge_smooth.max() * 0.15
    )

    # Draw these horizontal edge traces on the board
    for p in edge_peaks:
        cv2.line(vis_hlines, (x0, p), (x1, p), color, 1)

    # Strip boundary
    cv2.rectangle(vis_hlines, (x0, 0), (x1, bh), color, 1)

    print(f"    Strip {si}: {len(edge_peaks)} horizontal edge traces")

    save(out, f"06_strip{si}_sobel_y.jpg", sobel_y_norm)

save(out, "07_horizontal_edge_traces.jpg", vis_hlines)


# ============================================================
# STEP 6: Combined view — strips + tube boundaries + edge traces
# ============================================================

vis_combined = board.copy()

for si, strip in enumerate(actual_strips):
    color = colors[si % len(colors)]
    x0, x1 = strip["x_start"], strip["x_end"]

    # Strip boundary (thick)
    cv2.rectangle(vis_combined, (x0, 0), (x1, bh), color, 2)

    # Per-strip horizontal density valleys (tube boundaries)
    strip_region = inv[:, x0:x1]
    h_local = gaussian_filter1d(np.mean(strip_region, axis=1), sigma=2)
    valleys, _ = find_peaks(-h_local, distance=bh // 35, prominence=2)

    for v in valleys:
        # Tube boundary lines — constrained to this strip only
        cv2.line(vis_combined, (x0, v), (x1, v), color, 1)

    # Label
    cv2.putText(vis_combined,
                f"S{si}",
                (x0 + 2, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

save(out, "08_combined_grid.jpg", vis_combined)


# ============================================================
# Summary
# ============================================================

print(f"\n{'='*50}")
print(f"  SUMMARY")
print(f"{'='*50}")
print(f"  Board: {bw}x{bh}")
print(f"  Vertical profile peaks: {len(v_peaks)} strip centers")
print(f"  Boundaries: {len(boundaries) - 2} internal")
print(f"  Actual strips: {len(actual_strips)}")
for si, s in enumerate(actual_strips):
    x0, x1 = s["x_start"], s["x_end"]
    strip_region = inv[:, x0:x1]
    h_local = gaussian_filter1d(np.mean(strip_region, axis=1), sigma=2)
    valleys, _ = find_peaks(-h_local, distance=bh // 35, prominence=2)
    print(f"    Strip {si}: x={x0}–{x1} "
          f"(w={x1 - x0}), {len(valleys)} tube rows")

print(f"\nDone! Check {out}/")
