"""
26 — Strip → Plug Detection Pipeline
======================================
Step 1: Blue removal + border trim + adaptive threshold → binary mask
Step 2: Column projection → find 9 vertical strip boundaries
Step 3: Within each strip → find individual plug blobs
Step 4: Estimate plug size, build the grid layout
"""

from common import make_output_dir, save, crop_board

import cv2
import numpy as np
from scipy.signal import find_peaks


def remove_blue_lines(img):
    """Remove blue overlay lines via HSV mask + inpaint."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, (100, 80, 80), (130, 255, 255))
    blue_mask = cv2.dilate(blue_mask, np.ones((3, 3), np.uint8),
                           iterations=1)
    return cv2.inpaint(img, blue_mask, 5, cv2.INPAINT_TELEA)


def make_mask(img):
    """Adaptive threshold → morphology → clean binary mask."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enh = clahe.apply(gray)

    h, w = img.shape[:2]
    bk = max(11, int(max(w, h) * 0.06) | 1)
    adapt = cv2.adaptiveThreshold(enh, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV,
                                   blockSize=bk, C=8)

    k = max(3, min(w, h) // 80)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    clean = cv2.morphologyEx(adapt, cv2.MORPH_CLOSE, kern)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kern)
    return clean


def find_strips(mask, expected=9):
    """Column projection → find vertical strip boundaries.
    Returns list of (x_left, x_right) for each strip."""
    h, w = mask.shape[:2]

    # Sum white pixels per column → projection profile
    col_proj = np.sum(mask > 0, axis=0).astype(float)

    # Smooth to kill noise
    kern_w = max(5, w // 60)
    col_smooth = np.convolve(col_proj,
                              np.ones(kern_w) / kern_w, mode="same")

    # Find peaks = strip centers
    # Min distance between strips ~ width / (expected + 2)
    min_dist = w // (expected + 3)
    # Threshold: at least 15% of max column density
    thresh = np.max(col_smooth) * 0.15

    peaks, props = find_peaks(col_smooth, distance=min_dist,
                               height=thresh, prominence=thresh * 0.3)

    # If we got too few, lower threshold
    if len(peaks) < expected:
        thresh = np.max(col_smooth) * 0.08
        peaks, props = find_peaks(col_smooth, distance=min_dist,
                                   height=thresh)

    # Determine strip boundaries: valley between each pair of peaks
    strips = []
    for i, pk in enumerate(peaks):
        # Left boundary: midpoint to previous peak (or image edge)
        if i == 0:
            left = 0
        else:
            left = (peaks[i - 1] + pk) // 2

        # Right boundary: midpoint to next peak (or image edge)
        if i == len(peaks) - 1:
            right = w
        else:
            right = (pk + peaks[i + 1]) // 2

        strips.append((int(left), int(right), int(pk)))

    return strips, col_smooth


def find_plugs_in_strip(mask, strip_img, x_left, x_right, strip_idx):
    """Within one strip, find individual plug blobs.
    Returns list of plug dicts with cx, cy, area, bbox."""
    strip_mask = mask[:, x_left:x_right]
    sh, sw = strip_mask.shape[:2]

    n_cc, labels, stats, cents = cv2.connectedComponentsWithStats(
        strip_mask, connectivity=8)

    plugs = []
    for i in range(1, n_cc):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 8:
            continue

        cx = float(cents[i][0]) + x_left
        cy = float(cents[i][1])
        bx = stats[i, cv2.CC_STAT_LEFT] + x_left
        by = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]

        plugs.append({
            "strip": strip_idx, "cx": cx, "cy": cy, "area": area,
            "bx": bx, "by": by, "bw": bw, "bh": bh,
        })

    return plugs


def filter_plugs_global(all_strip_plugs):
    """GLOBAL median filter: compute median area across ALL strips,
    then keep blobs within [0.2×, 5×] global median per strip.
    Returns list-of-lists (one per strip) of filtered plugs."""
    every_area = [p["area"] for strip in all_strip_plugs for p in strip]
    if len(every_area) < 5:
        return all_strip_plugs

    global_med = float(np.median(every_area))
    lo, hi = global_med * 0.2, global_med * 5.0

    filtered = []
    for strip in all_strip_plugs:
        filtered.append([p for p in strip if lo <= p["area"] <= hi])
    return filtered, global_med


def assign_rows_global(all_strip_plugs, board_h):
    """Estimate ONE row spacing from the best strips, apply to all.
    Returns list-of-lists with row index added to each plug."""
    # Gather gaps from all strips
    all_gaps = []
    for strip in all_strip_plugs:
        srt = sorted(strip, key=lambda p: p["cy"])
        for i in range(len(srt) - 1):
            dy = srt[i + 1]["cy"] - srt[i]["cy"]
            if dy > 3:
                all_gaps.append(dy)

    if not all_gaps:
        return all_strip_plugs, 0, 0

    # Median gap = typical row spacing
    row_spacing = float(np.median(all_gaps))
    row_tol = row_spacing * 0.45

    # Assign rows per strip using global spacing
    max_row = 0
    for strip in all_strip_plugs:
        srt = sorted(strip, key=lambda p: p["cy"])
        if not srt:
            continue

        # Row 0 starts at first plug, then every row_spacing
        first_y = srt[0]["cy"]
        for p in srt:
            row_idx = max(0, round((p["cy"] - first_y) / row_spacing))
            p["row"] = row_idx
            max_row = max(max_row, row_idx)

    return all_strip_plugs, row_spacing, max_row + 1


# ── Main ────────────────────────────────────────────────
frame_path = ("/root/.claude/uploads/"
              "608e2092-c86e-5429-b039-27e3dec388ed/f95b95fa-image.jpg")
frame = cv2.imread(frame_path)
print(f"  Frame: {frame.shape[1]}x{frame.shape[0]}")

out = make_output_dir("26_strip_plug_detect")
board_raw = crop_board(frame, board_idx=0)

# Clean: blue removal + 5% border trim
board_noblue = remove_blue_lines(board_raw)
bh, bw = board_noblue.shape[:2]
tx, ty = int(bw * 0.05), int(bh * 0.05)
board = board_noblue[ty:bh - ty, tx:bw - tx]
bh, bw = board.shape[:2]
print(f"  Board (trimmed): {bw}x{bh}")
save(out, "00_board.jpg", board)

# Binary mask
mask = make_mask(board)
save(out, "01_mask.jpg", mask)

# ── STEP 2: Find vertical strips ───────────────────────
strips, col_proj = find_strips(mask, expected=9)
print(f"\n  Found {len(strips)} strips")

# Visualize column projection
proj_img = np.zeros((200, bw, 3), dtype=np.uint8)
proj_norm = col_proj / (np.max(col_proj) + 1e-6) * 180
for x in range(bw):
    y = int(proj_norm[x])
    cv2.line(proj_img, (x, 199), (x, 199 - y), (0, 200, 0), 1)
for left, right, center in strips:
    cv2.line(proj_img, (center, 0), (center, 199), (0, 0, 255), 1)
    cv2.line(proj_img, (left, 0), (left, 199), (255, 255, 0), 1)
save(out, "02_col_projection.jpg", proj_img)

# Strip overlay on board
strip_vis = board.copy()
for i, (left, right, center) in enumerate(strips):
    cv2.line(strip_vis, (left, 0), (left, bh), (0, 255, 0), 1)
    cv2.line(strip_vis, (right, 0), (right, bh), (0, 255, 0), 1)
    cv2.line(strip_vis, (center, 0), (center, bh), (0, 0, 255), 1)
    cv2.putText(strip_vis, f"S{i}", (center - 8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
save(out, "03_strips.jpg", strip_vis)

# ── STEP 3: Find plugs in each strip ───────────────────
# First pass: detect raw blobs per strip
raw_per_strip = []
for i, (left, right, center) in enumerate(strips):
    raw_plugs = find_plugs_in_strip(mask, board, left, right, i)
    raw_per_strip.append(raw_plugs)

# Global median filter (not per-strip!)
filt_per_strip, global_med = filter_plugs_global(raw_per_strip)
print(f"\n  Global median area: {global_med:.0f} px²")

# Global row assignment
filt_per_strip, row_spacing, n_rows_global = assign_rows_global(
    filt_per_strip, bh)
print(f"  Row spacing: {row_spacing:.1f} px")
print(f"  Estimated rows: {n_rows_global}")

# Collect results
all_plugs = []
strip_summary = []
for i, (left, right, center) in enumerate(strips):
    plugs = filt_per_strip[i]
    all_plugs.extend(plugs)

    areas = [p["area"] for p in plugs]
    med_area = float(np.median(areas)) if areas else 0
    n_rows_s = max((p.get("row", 0) for p in plugs), default=-1) + 1

    strip_summary.append({
        "strip": i, "x_left": left, "x_right": right,
        "width": right - left,
        "raw": len(raw_per_strip[i]), "filtered": len(plugs),
        "rows": n_rows_s, "med_area": med_area,
    })

    print(f"  Strip {i}: x=[{left},{right}] w={right-left}  "
          f"raw={len(raw_per_strip[i])} filt={len(plugs)} "
          f"rows={n_rows_s} med_area={med_area:.0f}")

# ── STEP 4: Full overlay ───────────────────────────────
plug_vis = board.copy()
# Draw strip boundaries
for left, right, center in strips:
    cv2.line(plug_vis, (left, 0), (left, bh), (0, 255, 0), 1)
    cv2.line(plug_vis, (right, 0), (right, bh), (0, 255, 0), 1)

# Draw plug markers
for p in all_plugs:
    cx, cy = int(p["cx"]), int(p["cy"])
    cv2.circle(plug_vis, (cx, cy), 4, (0, 0, 255), -1)
    # Small bbox
    bx, by, pbw, pbh = p["bx"], p["by"], p["bw"], p["bh"]
    cv2.rectangle(plug_vis, (bx, by), (bx + pbw, by + pbh),
                  (255, 0, 0), 1)
save(out, "04_plugs_detected.jpg", plug_vis)

# ── STEP 5: Stats ──────────────────────────────────────
total_plugs = len(all_plugs)
all_areas = np.array([p["area"] for p in all_plugs])

print(f"\n{'='*60}")
print(f"  STRIP → PLUG DETECTION RESULTS")
print(f"{'='*60}")
print(f"  Strips found:    {len(strips)}")
print(f"  Total plugs:     {total_plugs}")
if len(all_areas):
    print(f"  Area range:      {all_areas.min()}-{all_areas.max()} px²")
    print(f"  Area median:     {np.median(all_areas):.0f} px²")
    print(f"  Area Q1-Q3:      {np.percentile(all_areas, 25):.0f}"
          f"-{np.percentile(all_areas, 75):.0f} px²")

    # Estimate plug dimensions (assume roughly square)
    med_side = np.sqrt(np.median(all_areas))
    print(f"  Est. plug size:  ~{med_side:.0f}×{med_side:.0f} px")

print(f"\n  PER-STRIP BREAKDOWN:")
print(f"  {'Strip':>5} | {'Width':>5} | {'Raw':>4} | "
      f"{'Filt':>4} | {'Rows':>4} | {'MedArea':>7}")
print(f"  {'-'*48}")
for s in strip_summary:
    print(f"  S{s['strip']:<4} | {s['width']:>5} | {s['raw']:>4} | "
          f"{s['filtered']:>4} | {s['rows']:>4} | "
          f"{s['med_area']:>7.0f}")

# Row distribution
row_counts = [s["rows"] for s in strip_summary if s["rows"] > 0]
if row_counts:
    print(f"\n  Rows per strip:  "
          f"min={min(row_counts)} max={max(row_counts)} "
          f"med={np.median(row_counts):.0f}")
    est_grid = f"{len(strips)}×{int(np.median(row_counts))}"
    print(f"  Estimated grid:  {est_grid}")

print(f"\nDone! Check {out}/")
