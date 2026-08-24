"""
14 — Density Scan v2 (Correct Physical Model)
===============================================
9 vertical STRIPS side by side (like rulers on a table).
Each strip: ~20 cotton plugs running DOWN → plug-gap-plug-gap → ~40 rows.
Bottom has extra space; rightmost strip may differ.

Pipeline:
  1. Vertical density profile → 9 strip centers + boundaries
  2. Per-strip horizontal density → ~20 plug positions each
  3. Angle search per strip: rotate scan lines ±3°, maximize
     FFT peak at expected plug frequency → true tube angle
  4. Subdivided grid (plug-gap-plug-gap → ~40 channels)
"""

from common import load_image, make_output_dir, save, crop_board, remove_red_overlay

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def draw_profile(profile, length, height=150, color=(0, 255, 0),
                 markers=None, label=""):
    """Draw a 1D profile graph with optional peak markers."""
    vis = np.zeros((height, length, 3), dtype=np.uint8)
    norm = profile / (profile.max() + 1e-6)
    pts = [(x, height - int(norm[x] * (height - 20)))
           for x in range(min(len(norm), length))]
    for i in range(1, len(pts)):
        cv2.line(vis, pts[i - 1], pts[i], color, 1, cv2.LINE_AA)
    for m in ([] if markers is None else markers):
        if 0 <= int(m) < length:
            cv2.line(vis, (int(m), 0), (int(m), height), (0, 0, 255), 1)
    if label:
        cv2.putText(vis, label, (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return vis


# ── Setup ──────────────────────────────────────────────────
img, gray = load_image()
out = make_output_dir("14_density_v2")

board = crop_board(remove_red_overlay(img), board_idx=0)
board_gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
bh, bw = board_gray.shape
print(f"  Board: {bw}x{bh}")

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
inv = 255 - clahe.apply(board_gray)  # dark pixels → high values

colors = [
    (80, 80, 255), (80, 255, 80), (255, 140, 40),
    (255, 255, 40), (40, 255, 255), (255, 40, 255),
    (160, 255, 40), (255, 160, 40), (40, 160, 255),
]

# ── STEP 1: Vertical profile → 9 strip centers ────────────
v_profile = gaussian_filter1d(np.mean(inv, axis=0), sigma=5)
strip_peaks, _ = find_peaks(v_profile, distance=bw // 18, prominence=5)
strip_valleys, _ = find_peaks(-v_profile, distance=bw // 18, prominence=3)
print(f"  Strip peaks: {len(strip_peaks)}  x={strip_peaks.tolist()}")

save(out, "01_v_profile_strips.jpg",
     draw_profile(v_profile, bw, 200, (0, 255, 0), strip_peaks,
                  f"{len(strip_peaks)} strips"))

# Build strip boundary regions
all_bounds = sorted(np.concatenate(([0], strip_valleys, [bw - 1])))
strip_regions = []
for i in range(len(all_bounds) - 1):
    x0, x1 = int(all_bounds[i]), int(all_bounds[i + 1])
    if x1 - x0 > 15 and any(x0 <= p <= x1 for p in strip_peaks):
        strip_regions.append((x0, x1))

for i, (x0, x1) in enumerate(strip_regions):
    print(f"    Strip {i}: x={x0}–{x1} (w={x1-x0})")

# ── STEP 2: Per-strip horizontal density → plug rows ──────
vis_strips = board.copy()
strip_data = []

for si, (x0, x1) in enumerate(strip_regions):
    color = colors[si % len(colors)]
    cv2.line(vis_strips, (x0, 0), (x0, bh), color, 2)
    cv2.line(vis_strips, (x1, 0), (x1, bh), color, 2)

    h_smooth = gaussian_filter1d(np.mean(inv[:, x0:x1], axis=1), sigma=2)
    row_peaks, _ = find_peaks(h_smooth, distance=bh // 50, prominence=3)
    row_valleys, _ = find_peaks(-h_smooth, distance=bh // 50, prominence=2)

    med_sp = float(np.median(np.diff(row_peaks))) if len(row_peaks) > 1 else 0

    strip_data.append({
        "x0": x0, "x1": x1, "row_peaks": row_peaks,
        "row_valleys": row_valleys, "med_spacing": med_sp,
        "channel_h": med_sp / 2.0,
    })

    for v in row_valleys:
        cv2.line(vis_strips, (x0, v), (x1, v), color, 1)
    cv2.putText(vis_strips, f"S{si}:{len(row_peaks)}p",
                (x0 + 2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    print(f"  S{si}: {len(row_peaks)} plugs, spacing={med_sp:.0f}px")
    save(out, f"02_strip{si}_h_profile.jpg",
         draw_profile(h_smooth, bh, 150, color, row_peaks,
                      f"S{si}: {len(row_peaks)}p, sp={med_sp:.0f}"))

save(out, "03_all_strips_rows.jpg", vis_strips)

# ── STEP 3: Angle search per strip (FFT periodicity) ──────
print(f"\n  --- Angle search ---")
vis_angles = cv2.addWeighted(board.copy(), 0.5,
                              np.zeros_like(board), 0, 0)

for si, (x0, x1) in enumerate(strip_regions):
    color = colors[si % len(colors)]
    strip_w, strip_cx = x1 - x0, (x0 + x1) // 2
    med_sp = strip_data[si]["med_spacing"]
    best_angle, best_score = 0.0, 0.0

    for angle_deg in np.arange(-3.0, 3.25, 0.25):
        tan_a = np.tan(np.radians(angle_deg))
        vals = np.zeros(bh)
        for y in range(bh):
            x_off = int((y - bh / 2) * tan_a)
            lo = max(x0, strip_cx + x_off - strip_w // 4)
            hi = min(x1, strip_cx + x_off + strip_w // 4)
            if hi > lo:
                vals[y] = np.mean(inv[y, lo:hi])
        vals = gaussian_filter1d(vals, sigma=2)

        if med_sp > 5:
            fft = np.abs(np.fft.rfft(vals - vals.mean()))
            fb = round(bh / med_sp)
            score = float(np.max(fft[max(1, fb - 2):min(len(fft), fb + 3)]))
        else:
            score = np.std(vals)
        if score > best_score:
            best_score, best_angle = score, angle_deg

    strip_data[si]["best_angle"] = best_angle
    print(f"    S{si}: angle={best_angle:.2f}° (score={best_score:.0f})")

    tan_a = np.tan(np.radians(best_angle))
    cv2.line(vis_angles, (x0, 0), (x0, bh), color, 2)
    cv2.line(vis_angles, (x1, 0), (x1, bh), color, 2)
    for v in strip_data[si]["row_valleys"]:
        xs = int((v - bh / 2) * tan_a)
        cv2.line(vis_angles, (x0 + xs, v), (x1 + xs, v), color, 1)
    cv2.putText(vis_angles, f"S{si}: {best_angle:.1f}deg",
                (x0 + 2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

save(out, "04_angle_corrected.jpg", vis_angles)

# ── STEP 4: Subdivided grid (plug-gap-plug-gap) ───────────
vis_subdiv = board.copy()

for si, (x0, x1) in enumerate(strip_regions):
    color = colors[si % len(colors)]
    sd = strip_data[si]
    if sd["channel_h"] < 5:
        continue
    tan_a = np.tan(np.radians(sd.get("best_angle", 0)))
    peaks = sd["row_peaks"]

    cv2.line(vis_subdiv, (x0, 0), (x0, bh), color, 2)
    cv2.line(vis_subdiv, (x1, 0), (x1, bh), color, 2)

    # Build plug + gap interleaved list
    expanded = []
    for i, p in enumerate(peaks):
        expanded.append(("plug", int(p)))
        if i < len(peaks) - 1:
            expanded.append(("gap", (int(peaks[i]) + int(peaks[i + 1])) // 2))

    # Draw channel boundaries (between each plug/gap pair)
    for i in range(len(expanded) - 1):
        by = (expanded[i][1] + expanded[i + 1][1]) // 2
        xs = int((by - bh / 2) * tan_a)
        cv2.line(vis_subdiv, (max(0, x0 + xs), by),
                 (min(bw - 1, x1 + xs), by), color, 1)

    # White lines through plug centers
    for ptype, py in expanded:
        if ptype == "plug":
            xs = int((py - bh / 2) * tan_a)
            cv2.line(vis_subdiv, (max(0, x0 + xs), py),
                     (min(bw - 1, x1 + xs), py), (255, 255, 255), 1)

    cv2.putText(vis_subdiv, f"S{si}:{len(expanded)}ch",
                (x0 + 2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

save(out, "05_subdivided_grid.jpg", vis_subdiv)

# ── Summary ────────────────────────────────────────────────
print(f"\n{'='*50}\n  SUMMARY — {bw}x{bh}, {len(strip_regions)} strips")
for si, sd in enumerate(strip_data):
    n = len(sd["row_peaks"])
    a = sd.get("best_angle", 0)
    print(f"    S{si}: x={sd['x0']}-{sd['x1']}, {n} plugs → "
          f"~{n*2-1} ch, angle={a:.1f}°, sp={sd['med_spacing']:.0f}px")
print(f"\nDone! Check {out}/")
