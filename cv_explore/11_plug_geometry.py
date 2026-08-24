"""
11 — Cotton Plug Geometry (Anchor-Based Grid Detection)
========================================================
Strategy: Design a system that works no matter how bad the video is.
Work from INVARIANTS — things that are always true about the physical setup:

  1. Cotton plugs are always there — big, dark, high-contrast
  2. Plugs sit at one end of each tube, one per tube
  3. A line through all plugs of a strip → strip orientation
  4. Perpendicular to that line → tube direction
  5. Tube walls create bars perpendicular to the plug line
  6. Plugs are equispaced: plug-wall-plug-wall-plug...

Pipeline:
  - Find the cotton plugs (multiple methods: threshold, color, blobs)
  - Fit a line through each strip's plugs → "plug line"
  - Detect bars/walls perpendicular to that line
  - Lay equispaced grid anchored on plug positions
"""

from common import load_image, make_output_dir, save, crop_board, remove_red_overlay

import cv2
import numpy as np

img, gray = load_image()
out = make_output_dir("11_plug_geometry")

board = crop_board(remove_red_overlay(img), board_idx=0)
board_gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
bh, bw = board_gray.shape
print(f"  Board size: {bw}x{bh}")

# ============================================================
# STEP 1: Find cotton plugs — try multiple methods
# ============================================================

# --- Method A: Dark blob detection via thresholding ---
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
enhanced = clahe.apply(board_gray)

# Otsu on inverted → dark objects become white
_, binary = cv2.threshold(enhanced, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Close small gaps inside plugs, then open to remove noise
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_open)

# connectedComponentsWithStats — one call gives everything
n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    cleaned, connectivity=8
)

# Filter by area: plugs are big blobs (much bigger than flies)
# Flies: 30-200 px², plugs: probably 500+ px²
min_plug_area = 300
max_plug_area = bh * bw * 0.05  # max 5% of board

plug_candidates = []
vis_all = board.copy()
for i in range(1, n_labels):  # skip background (label 0)
    area = stats[i, cv2.CC_STAT_AREA]
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    cx, cy = centroids[i]

    if min_plug_area < area < max_plug_area:
        aspect = w / max(h, 1)
        plug_candidates.append({
            "label": i, "area": area,
            "x": x, "y": y, "w": w, "h": h,
            "cx": cx, "cy": cy, "aspect": aspect,
        })
        # Draw all candidates — color by size
        color = (0, 255, 0) if area > 800 else (255, 200, 0)
        cv2.rectangle(vis_all, (x, y), (x + w, y + h), color, 2)
        cv2.putText(vis_all, f"{int(area)}", (x, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

save(out, "01_all_blob_candidates.jpg", vis_all)
print(f"  connectedComponents: {n_labels - 1} total, "
      f"{len(plug_candidates)} in plug size range")


# --- Method B: HSV color thresholding ---
# Try to find plugs by color, not just brightness
hsv = cv2.cvtColor(board, cv2.COLOR_BGR2HSV)
lab = cv2.cvtColor(board, cv2.COLOR_BGR2LAB)

# Plugs are dark → low V (value) in HSV, low L in LAB
# Try a range of V thresholds to see which captures plugs best
vis_hsv = board.copy()
for max_v in [60, 80, 100, 120]:
    dark_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, max_v))
    dark_cleaned = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel_close)
    dark_cleaned = cv2.morphologyEx(dark_cleaned, cv2.MORPH_OPEN, kernel_open)
    save(out, f"02_dark_mask_v{max_v}.jpg", dark_cleaned)

# Also try L channel from LAB
l_ch = lab[:, :, 0]
for max_l in [60, 80, 100, 120]:
    _, l_mask = cv2.threshold(l_ch, max_l, 255, cv2.THRESH_BINARY_INV)
    l_cleaned = cv2.morphologyEx(l_mask, cv2.MORPH_CLOSE, kernel_close)
    l_cleaned = cv2.morphologyEx(l_cleaned, cv2.MORPH_OPEN, kernel_open)
    save(out, f"02_dark_mask_L{max_l}.jpg", l_cleaned)


# --- Method C: Per-channel analysis ---
# Split into B, G, R and see which channel separates plugs best
b_ch, g_ch, r_ch = cv2.split(board)
for name, ch in [("blue", b_ch), ("green", g_ch), ("red", r_ch)]:
    _, ch_mask = cv2.threshold(ch, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ch_cleaned = cv2.morphologyEx(ch_mask, cv2.MORPH_CLOSE, kernel_close)
    ch_cleaned = cv2.morphologyEx(ch_cleaned, cv2.MORPH_OPEN, kernel_open)
    save(out, f"03_channel_{name}.jpg", ch_cleaned)
    save(out, f"03_channel_{name}_raw.jpg", ch)


# ============================================================
# STEP 2: Separate plugs from flies by size + position
# ============================================================

# Plugs should be in a cluster along one edge of the board
# Sort candidates by area (descending) to see the size distribution
plug_candidates.sort(key=lambda p: p["area"], reverse=True)

# Print size distribution
print("\n  Blob size distribution:")
size_bins = [(0, 100), (100, 300), (300, 800), (800, 2000), (2000, 10000)]
for lo, hi in size_bins:
    count = sum(1 for p in plug_candidates if lo <= p["area"] < hi)
    if count:
        print(f"    {lo}-{hi} px²: {count} blobs")

# Strategy: plugs are the BIGGEST dark blobs
# Find a natural gap in the size distribution
areas = np.array([p["area"] for p in plug_candidates])
if len(areas) > 5:
    # Sort areas descending, look for a big jump
    sorted_areas = np.sort(areas)[::-1]
    ratios = sorted_areas[:-1] / np.maximum(sorted_areas[1:], 1)
    jump_idx = np.argmax(ratios > 2)  # first 2x jump

    if ratios[jump_idx] > 2:
        plug_threshold = sorted_areas[jump_idx]
        plugs = [p for p in plug_candidates if p["area"] >= plug_threshold]
        non_plugs = [p for p in plug_candidates if p["area"] < plug_threshold]
        print(f"\n  Size gap at {plug_threshold:.0f} px²"
              f" (ratio {ratios[jump_idx]:.1f}x)")
    else:
        # Fallback: top 20% by area are plugs (heuristic)
        n_top = max(5, len(plug_candidates) // 5)
        plugs = plug_candidates[:n_top]
        non_plugs = plug_candidates[n_top:]
        print(f"\n  No clear gap — using top {n_top} by area")
else:
    plugs = plug_candidates
    non_plugs = []

print(f"  Plugs: {len(plugs)}, Non-plugs: {len(non_plugs)}")

# Visualize classified blobs
vis_class = board.copy()
for p in non_plugs:
    cv2.rectangle(vis_class,
                  (p["x"], p["y"]),
                  (p["x"] + p["w"], p["y"] + p["h"]),
                  (200, 200, 200), 1)
for p in plugs:
    cv2.rectangle(vis_class,
                  (p["x"], p["y"]),
                  (p["x"] + p["w"], p["y"] + p["h"]),
                  (0, 0, 255), 2)
    cv2.circle(vis_class, (int(p["cx"]), int(p["cy"])), 4,
               (0, 255, 255), -1)

save(out, "04_plugs_classified.jpg", vis_class)


# ============================================================
# STEP 3: Fit a line through the plug centroids
# ============================================================

if len(plugs) >= 2:
    plug_pts = np.array([(p["cx"], p["cy"]) for p in plugs],
                        dtype=np.float32)

    # fitLine: robust line fitting (HUBER handles outliers)
    vx, vy, x0, y0 = cv2.fitLine(
        plug_pts.reshape(-1, 1, 2), cv2.DIST_HUBER, 0, 0.01, 0.01
    ).flatten()

    # Line direction vector and perpendicular
    line_dir = np.array([vx, vy])
    perp_dir = np.array([-vy, vx])  # 90° rotation

    # Compute angle of plug line
    angle_deg = np.degrees(np.arctan2(vy, vx))
    print(f"\n  Plug line angle: {angle_deg:.1f}°")
    print(f"  Direction: ({vx:.3f}, {vy:.3f})")
    print(f"  Perpendicular (tube direction): ({perp_dir[0]:.3f}, "
          f"{perp_dir[1]:.3f})")

    # Draw the plug line and perpendicular direction
    vis_lines = board.copy()

    # Plug line (red) — extend across the board
    scale = max(bw, bh)
    pt1 = (int(x0 - vx * scale), int(y0 - vy * scale))
    pt2 = (int(x0 + vx * scale), int(y0 + vy * scale))
    cv2.line(vis_lines, pt1, pt2, (0, 0, 255), 2)

    # Perpendicular arrows from each plug centroid (green)
    arrow_len = bh * 0.15
    for p in plugs:
        start = (int(p["cx"]), int(p["cy"]))
        end = (int(p["cx"] + perp_dir[0] * arrow_len),
               int(p["cy"] + perp_dir[1] * arrow_len))
        cv2.arrowedLine(vis_lines, start, end, (0, 255, 0), 2,
                        tipLength=0.15)
        cv2.circle(vis_lines, start, 5, (0, 255, 255), -1)

    # Label
    cv2.putText(vis_lines,
                f"Plug line: {angle_deg:.1f} deg",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2)
    cv2.putText(vis_lines,
                f"Tube dir: {angle_deg + 90:.1f} deg",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0), 2)

    save(out, "05_plug_line_and_tube_dir.jpg", vis_lines)

    # ============================================================
    # STEP 4: Measure plug spacing along the plug line
    # ============================================================

    # Project plug centroids onto the plug line
    projections = []
    for p in plugs:
        pt = np.array([p["cx"] - x0, p["cy"] - y0])
        proj = np.dot(pt, line_dir)
        projections.append((proj, p))

    projections.sort(key=lambda x: x[0])

    # Spacing between consecutive plugs
    spacings = []
    for i in range(1, len(projections)):
        d = projections[i][0] - projections[i - 1][0]
        spacings.append(d)

    if spacings:
        median_spacing = np.median(spacings)
        print(f"\n  Plug spacings along line:")
        print(f"    Median: {median_spacing:.1f} px")
        print(f"    Min: {min(spacings):.1f}, Max: {max(spacings):.1f}")
        print(f"    All: {[f'{s:.0f}' for s in spacings]}")

    # ============================================================
    # STEP 5: Lay equispaced grid perpendicular to plug line
    # ============================================================

    vis_grid = board.copy()

    # For each plug, draw the tube centerline going perpendicular
    tube_len = bh * 0.9  # tubes run most of the board height
    for proj_val, p in projections:
        start = (int(p["cx"] - perp_dir[0] * 20),
                 int(p["cy"] - perp_dir[1] * 20))
        end = (int(p["cx"] + perp_dir[0] * tube_len),
               int(p["cy"] + perp_dir[1] * tube_len))
        cv2.line(vis_grid, start, end, (0, 200, 0), 1)

    # Draw plug line
    cv2.line(vis_grid, pt1, pt2, (0, 0, 255), 1)

    # Draw plug bounding boxes
    for p in plugs:
        cv2.rectangle(vis_grid,
                      (p["x"], p["y"]),
                      (p["x"] + p["w"], p["y"] + p["h"]),
                      (0, 255, 255), 1)

    # Now add WALL LINES halfway between each pair of tube lines
    for i in range(len(projections) - 1):
        _, p1 = projections[i]
        _, p2 = projections[i + 1]
        mid_x = (p1["cx"] + p2["cx"]) / 2
        mid_y = (p1["cy"] + p2["cy"]) / 2
        start = (int(mid_x - perp_dir[0] * 20),
                 int(mid_y - perp_dir[1] * 20))
        end = (int(mid_x + perp_dir[0] * tube_len),
               int(mid_y + perp_dir[1] * tube_len))
        cv2.line(vis_grid, start, end, (200, 100, 0), 1)

    cv2.putText(vis_grid,
                f"{len(plugs)} plugs -> {len(plugs)} tubes",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2)

    save(out, "06_grid_from_plugs.jpg", vis_grid)
    print(f"\n  Grid: {len(plugs)} tubes detected from plug positions")

else:
    print("  WARNING: fewer than 2 plugs found, cannot fit line")


# ============================================================
# STEP 6: Find perpendicular bars (walls) via morphology
# ============================================================

# Use the plug line angle to create a ROTATED morphological kernel
# that extracts bars parallel to the tube direction

if len(plugs) >= 2:
    # Angle of tubes (perpendicular to plug line)
    tube_angle = np.degrees(np.arctan2(perp_dir[1], perp_dir[0]))

    # Create a rotated kernel for morphological extraction
    # We want a kernel that is long in the tube direction
    kern_len = max(20, int(bh * 0.1))

    # Simple approach: rotate the binary image to align tubes vertically,
    # apply vertical morphology, rotate back
    center = (bw // 2, bh // 2)
    rotation_needed = -tube_angle  # rotate so tubes point straight down

    M_rot = cv2.getRotationMatrix2D(center, rotation_needed, 1.0)
    rotated = cv2.warpAffine(binary, M_rot, (bw, bh))

    # Now extract vertical lines (which were tube-direction in original)
    vert_kern = cv2.getStructuringElement(cv2.MORPH_RECT,
                                          (1, kern_len))
    tube_walls = cv2.morphologyEx(rotated, cv2.MORPH_OPEN, vert_kern)

    # Also extract horizontal lines (perpendicular = cross walls)
    horiz_kern = cv2.getStructuringElement(cv2.MORPH_RECT,
                                           (max(10, int(bw * 0.05)), 1))
    cross_walls = cv2.morphologyEx(rotated, cv2.MORPH_OPEN, horiz_kern)

    # Rotate back
    M_rot_inv = cv2.getRotationMatrix2D(center, -rotation_needed, 1.0)
    tube_walls_orig = cv2.warpAffine(tube_walls, M_rot_inv, (bw, bh))
    cross_walls_orig = cv2.warpAffine(cross_walls, M_rot_inv, (bw, bh))

    save(out, "07_tube_walls_morph.jpg", tube_walls_orig)
    save(out, "07_cross_walls_morph.jpg", cross_walls_orig)

    # Overlay both on board
    vis_walls = board.copy()
    vis_walls[tube_walls_orig > 128] = [0, 255, 0]    # green = tube walls
    vis_walls[cross_walls_orig > 128] = [0, 100, 255]  # orange = cross walls
    save(out, "08_walls_overlay.jpg", vis_walls)

    print(f"  Morphological wall extraction (tube angle: {tube_angle:.1f}°)")


# ============================================================
# STEP 7: Summary visualization
# ============================================================

vis_summary = board.copy()
# Darken the background slightly
vis_summary = cv2.addWeighted(vis_summary, 0.6,
                               np.zeros_like(vis_summary), 0, 0)

# Draw plug bboxes in yellow
for p in plugs:
    cv2.rectangle(vis_summary,
                  (p["x"], p["y"]),
                  (p["x"] + p["w"], p["y"] + p["h"]),
                  (0, 255, 255), 2)

# Draw tube centerlines in green
if len(plugs) >= 2:
    for proj_val, p in projections:
        start = (int(p["cx"]), int(p["cy"]))
        end = (int(p["cx"] + perp_dir[0] * tube_len),
               int(p["cy"] + perp_dir[1] * tube_len))
        cv2.line(vis_summary, start, end, (0, 200, 0), 2)

    # Draw wall lines (between tubes) in blue
    for i in range(len(projections) - 1):
        _, p1 = projections[i]
        _, p2 = projections[i + 1]
        mid_x = (p1["cx"] + p2["cx"]) / 2
        mid_y = (p1["cy"] + p2["cy"]) / 2
        start = (int(mid_x), int(mid_y))
        end = (int(mid_x + perp_dir[0] * tube_len),
               int(mid_y + perp_dir[1] * tube_len))
        cv2.line(vis_summary, start, end, (255, 100, 0), 1)

    # Plug line in red
    cv2.line(vis_summary, pt1, pt2, (0, 0, 255), 2)

save(out, "09_summary.jpg", vis_summary)

print(f"\nDone! Check {out}/")
