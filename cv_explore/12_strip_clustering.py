"""
12 — Strip Clustering (Independent Strips Detection)
=====================================================
The board is NOT one monolithic grid. It's SEPARATE STRIPS of plastic
tubing laid side by side — like rulers thrown on a table.

Each strip:
  - Has its own orientation (angle, position)
  - Has cotton plugs at one end, one per tube
  - Plugs on the SAME strip are vertically aligned (roughly)
  - Plugs on DIFFERENT strips are at different x-positions

Strategy:
  1. Find all cotton plug blobs
  2. Cluster plugs by x-position → each cluster = one strip
     (plugs above/below each other = same strip)
  3. Fit a SEPARATE line per strip
  4. Each strip gets its own tube direction + grid

This handles strips at different angles, positions, and even scales.
"""

from common import load_image, make_output_dir, save, crop_board, remove_red_overlay

import cv2
import numpy as np

img, gray = load_image()
out = make_output_dir("12_strip_clustering")

board = crop_board(remove_red_overlay(img), board_idx=0)
board_gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
bh, bw = board_gray.shape
print(f"  Board: {bw}x{bh}")

# ============================================================
# STEP 1: Find ALL dark blobs — be generous, filter later
# ============================================================

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
enhanced = clahe.apply(board_gray)

# Adaptive threshold handles uneven lighting better than Otsu
# This is an invariant-friendly choice — works even with bad video
adaptive = cv2.adaptiveThreshold(
    enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV, blockSize=51, C=10
)

# Also try Otsu for comparison
_, otsu = cv2.threshold(enhanced, 0, 255,
                        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Clean up both
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

for name, mask in [("adaptive", adaptive), ("otsu", otsu)]:
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_open)
    save(out, f"00_{name}_binary.jpg", cleaned)

# Use Otsu for plug detection (cleaner for high-contrast blobs)
binary = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel_close)
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

# Connected components
n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    binary, connectivity=8
)

# Collect all blobs with their properties
blobs = []
for i in range(1, n_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    cx, cy = centroids[i]

    if area < 20:
        continue

    blobs.append({
        "i": i, "area": area,
        "x": x, "y": y, "w": w, "h": h,
        "cx": cx, "cy": cy,
        "aspect": w / max(h, 1),
    })

print(f"  Total blobs: {len(blobs)}")

# ============================================================
# STEP 2: Identify plugs — they're the biggest blobs near
#         one edge of the board (top or bottom, not middle)
# ============================================================

# First: let's look at blob sizes to understand the distribution
areas = sorted([b["area"] for b in blobs], reverse=True)
print(f"\n  Top 20 blob areas: {areas[:20]}")

# Plugs are BIG relative to flies
# But we don't know the exact cutoff — use the size gap
# Look for a natural break in the sorted areas
log_areas = np.log10(np.array(areas) + 1)

# Find the biggest ratio jump
ratios = []
for i in range(min(30, len(areas) - 1)):
    r = areas[i] / max(areas[i + 1], 1)
    ratios.append((i, r, areas[i], areas[i + 1]))

# Sort by ratio to find the biggest gap
ratios.sort(key=lambda x: x[1], reverse=True)
print(f"\n  Biggest area gaps:")
for idx, ratio, big, small in ratios[:5]:
    print(f"    Position {idx}: {big} / {small} = {ratio:.1f}x")

# Use a combination: plugs are blobs that are
# 1. Larger than the median blob (size filter)
# 2. Within the top 20% of Y positions OR bottom 20%
#    (plugs are at one end of the strips)
# 3. Not too elongated (aspect ratio sanity)

# For now, just separate by a reasonable area threshold
# Plugs in this photo look like they're > 500 px²
area_threshold = 400
plug_blobs = [b for b in blobs if b["area"] > area_threshold]
fly_blobs = [b for b in blobs if b["area"] <= area_threshold]

print(f"\n  Above {area_threshold} px²: {len(plug_blobs)} blobs")
print(f"  Below {area_threshold} px²: {len(fly_blobs)} blobs")

# Visualize plug candidates
vis_plugs = board.copy()
for b in plug_blobs:
    cv2.rectangle(vis_plugs, (b["x"], b["y"]),
                  (b["x"] + b["w"], b["y"] + b["h"]),
                  (0, 0, 255), 2)
    cv2.circle(vis_plugs, (int(b["cx"]), int(b["cy"])), 3,
               (0, 255, 255), -1)
    cv2.putText(vis_plugs, f"{b['area']}",
                (b["x"], b["y"] - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
save(out, "01_plug_candidates.jpg", vis_plugs)


# ============================================================
# STEP 3: Cluster plugs into strips by x-position
# ============================================================

# Plugs on the same strip have similar x-coordinates
# (since strips are roughly vertical — tubes run top-to-bottom)
# Sort by x, then group by proximity

plug_blobs.sort(key=lambda b: b["cx"])
x_coords = np.array([b["cx"] for b in plug_blobs])

# Use a simple 1D clustering: consecutive plugs within
# some x-distance belong to the same strip
# Strip width is roughly bw / 9 (9 columns), so plugs on
# the same strip should be within ~20px of each other
max_x_gap = bw / 15  # generous gap threshold

strips = []
current_strip = [plug_blobs[0]] if plug_blobs else []

for i in range(1, len(plug_blobs)):
    if abs(plug_blobs[i]["cx"] - plug_blobs[i - 1]["cx"]) < max_x_gap:
        current_strip.append(plug_blobs[i])
    else:
        if current_strip:
            strips.append(current_strip)
        current_strip = [plug_blobs[i]]

if current_strip:
    strips.append(current_strip)

print(f"\n  Strips found: {len(strips)}")
for i, s in enumerate(strips):
    xs = [b["cx"] for b in s]
    ys = [b["cy"] for b in s]
    print(f"    Strip {i}: {len(s)} plugs, "
          f"x={min(xs):.0f}–{max(xs):.0f}, "
          f"y={min(ys):.0f}–{max(ys):.0f}")

# Visualize strips with different colors
colors = [
    (0, 0, 255), (0, 255, 0), (255, 0, 0),
    (255, 255, 0), (0, 255, 255), (255, 0, 255),
    (128, 255, 0), (255, 128, 0), (0, 128, 255),
    (128, 0, 255), (255, 0, 128), (0, 255, 128),
]
vis_strips = board.copy()
for si, strip in enumerate(strips):
    color = colors[si % len(colors)]
    for b in strip:
        cv2.rectangle(vis_strips, (b["x"], b["y"]),
                      (b["x"] + b["w"], b["y"] + b["h"]),
                      color, 2)
        cv2.circle(vis_strips, (int(b["cx"]), int(b["cy"])), 4,
                   (255, 255, 255), -1)
    # Label the strip
    avg_x = np.mean([b["cx"] for b in strip])
    min_y = min(b["y"] for b in strip)
    cv2.putText(vis_strips, f"S{si}({len(strip)})",
                (int(avg_x) - 15, max(min_y - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

save(out, "02_strips_clustered.jpg", vis_strips)


# ============================================================
# STEP 4: For each strip, fit a line and derive tube direction
# ============================================================

vis_geometry = board.copy()
vis_geometry = cv2.addWeighted(vis_geometry, 0.5,
                                np.zeros_like(vis_geometry), 0, 0)

strip_data = []

for si, strip in enumerate(strips):
    color = colors[si % len(colors)]

    if len(strip) < 2:
        print(f"\n  Strip {si}: only {len(strip)} plug — skipping line fit")
        # Still draw the single plug
        for b in strip:
            cv2.circle(vis_geometry, (int(b["cx"]), int(b["cy"])),
                       6, color, -1)
        continue

    # Fit line through this strip's plug centroids
    pts = np.array([(b["cx"], b["cy"]) for b in strip],
                   dtype=np.float32).reshape(-1, 1, 2)
    vx, vy, x0, y0 = cv2.fitLine(
        pts, cv2.DIST_HUBER, 0, 0.01, 0.01
    ).flatten()

    plug_dir = np.array([vx, vy])
    tube_dir = np.array([-vy, vx])  # perpendicular
    angle = np.degrees(np.arctan2(vy, vx))

    # Project plugs onto the plug line to get spacing
    projections = []
    for b in strip:
        pt = np.array([b["cx"] - x0, b["cy"] - y0])
        proj = np.dot(pt, plug_dir)
        projections.append((proj, b))
    projections.sort(key=lambda x: x[0])

    spacings = [projections[i][0] - projections[i - 1][0]
                for i in range(1, len(projections))]
    med_spacing = np.median(spacings) if spacings else 0

    strip_info = {
        "idx": si, "plugs": strip, "angle": angle,
        "plug_dir": plug_dir, "tube_dir": tube_dir,
        "origin": (x0, y0), "vx": vx, "vy": vy,
        "projections": projections, "spacings": spacings,
        "median_spacing": med_spacing,
    }
    strip_data.append(strip_info)

    print(f"\n  Strip {si}: {len(strip)} plugs, angle={angle:.1f}°, "
          f"median spacing={med_spacing:.0f}px")

    # Draw plug line
    scale = max(bw, bh)
    pt1 = (int(x0 - vx * scale), int(y0 - vy * scale))
    pt2 = (int(x0 + vx * scale), int(y0 + vy * scale))
    cv2.line(vis_geometry, pt1, pt2, color, 2)

    # Draw tube lines from each plug
    tube_len = bh * 0.85
    for proj_val, b in projections:
        start = (int(b["cx"]), int(b["cy"]))
        end = (int(b["cx"] + tube_dir[0] * tube_len),
               int(b["cy"] + tube_dir[1] * tube_len))
        cv2.line(vis_geometry, start, end, color, 1)
        cv2.circle(vis_geometry, start, 4, (255, 255, 255), -1)

    # Draw wall lines (midpoints between tubes)
    for i in range(len(projections) - 1):
        _, b1 = projections[i]
        _, b2 = projections[i + 1]
        mx = (b1["cx"] + b2["cx"]) / 2
        my = (b1["cy"] + b2["cy"]) / 2
        start = (int(mx), int(my))
        end = (int(mx + tube_dir[0] * tube_len),
               int(my + tube_dir[1] * tube_len))
        cv2.line(vis_geometry, start, end, color, 1,
                 lineType=cv2.LINE_AA)

    # Label
    avg_x = np.mean([b["cx"] for b in strip])
    min_y = min(b["y"] for b in strip)
    cv2.putText(vis_geometry,
                f"S{si}: {angle:.0f}deg, {len(strip)} tubes",
                (int(avg_x) - 30, max(min_y - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

save(out, "03_per_strip_geometry.jpg", vis_geometry)


# ============================================================
# STEP 5: Visual report — annotated board with all strips
# ============================================================

vis_final = board.copy()

for sd in strip_data:
    si = sd["idx"]
    color = colors[si % len(colors)]
    tube_dir = sd["tube_dir"]
    tube_len = bh * 0.85

    # Tube centerlines (solid)
    for _, b in sd["projections"]:
        start = (int(b["cx"]), int(b["cy"]))
        end = (int(b["cx"] + tube_dir[0] * tube_len),
               int(b["cy"] + tube_dir[1] * tube_len))
        cv2.line(vis_final, start, end, color, 2)

    # Plug bboxes
    for b in sd["plugs"]:
        cv2.rectangle(vis_final, (b["x"], b["y"]),
                      (b["x"] + b["w"], b["y"] + b["h"]),
                      (0, 255, 255), 2)

# Also show singleton strips
for si, strip in enumerate(strips):
    if len(strip) < 2:
        color = colors[si % len(colors)]
        for b in strip:
            cv2.rectangle(vis_final, (b["x"], b["y"]),
                          (b["x"] + b["w"], b["y"] + b["h"]),
                          (0, 255, 255), 2)
            cv2.putText(vis_final, f"S{si}?",
                        (b["x"], b["y"] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        color, 1)

save(out, "04_final_annotated.jpg", vis_final)

# Print summary
print(f"\n{'='*50}")
print(f"  SUMMARY")
print(f"{'='*50}")
print(f"  Total blobs found: {len(blobs)}")
print(f"  Plug candidates: {len(plug_blobs)}")
print(f"  Strips detected: {len(strips)}")
for sd in strip_data:
    n = len(sd["plugs"])
    a = sd["angle"]
    ms = sd["median_spacing"]
    print(f"    Strip {sd['idx']}: {n} tubes, "
          f"angle {a:.1f}°, spacing {ms:.0f}px")

print(f"\nDone! Check {out}/")
