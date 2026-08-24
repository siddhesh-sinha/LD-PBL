"""
10 — Projection Profiles
===========================
What it does: Collapses 2D images into 1D signals by summing along
one axis. The peaks and valleys in the profile directly correspond
to structural features in that direction.

  - Vertical projection (sum columns) → column wall positions
  - Horizontal projection (sum rows) → row boundary positions
  - Can project: raw intensity, edges, binary masks, or blobs

Why it matters:
  - This is the SIMPLEST way to find a repeating grid
  - Column walls = dips in vertical intensity projection
  - Row boundaries = dips in horizontal intensity projection
  - Combined with autocorrelation → robust periodic grid detection
"""

from common import load_image, make_output_dir, save, crop_board, remove_red_overlay

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

img, gray = load_image()
out = make_output_dir("10_projections")

board = crop_board(remove_red_overlay(img), board_idx=0)
board_gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
bh, bw = board_gray.shape

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
enhanced = clahe.apply(board_gray)

# Skip top 8% (plugs)
yt = int(bh * 0.08)
work = enhanced[yt:, :]
wh, ww = work.shape

edges = cv2.Canny(work, 30, 100)

def draw_profile(profile, axis, bh, bw, label="", peaks=None, color=(0, 200, 0)):
    """Draw a projection profile as an image alongside the source."""
    bar_size = 200
    if axis == "vertical":
        prof_img = np.zeros((bar_size, ww, 3), dtype=np.uint8)
        norm = profile / (profile.max() + 1e-6)
        for x in range(min(len(norm), ww)):
            bar = int(norm[x] * bar_size)
            cv2.line(prof_img, (x, bar_size), (x, bar_size - bar), color, 1)
        if peaks is not None:
            for p in peaks:
                cv2.line(prof_img, (p, 0), (p, bar_size), (0, 0, 255), 2)
    else:
        prof_img = np.zeros((wh, bar_size, 3), dtype=np.uint8)
        norm = profile / (profile.max() + 1e-6)
        for y in range(min(len(norm), wh)):
            bar = int(norm[y] * bar_size)
            cv2.line(prof_img, (0, y), (bar, y), color, 1)
        if peaks is not None:
            for p in peaks:
                cv2.line(prof_img, (0, p), (bar_size, p), (0, 0, 255), 2)

    if label:
        cv2.putText(prof_img, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return prof_img


# --- Vertical projections (find columns) ---

# Raw intensity (walls are darker → dips in projection)
v_intensity = np.mean(work, axis=0)
v_int_smooth = gaussian_filter1d(v_intensity, sigma=3)

# Edge density (walls have more edges → peaks)
v_edges = np.sum(edges, axis=0).astype(float) / 255
v_edge_smooth = gaussian_filter1d(v_edges, sigma=3)

# Inverted binary (dark features = columns of flies + walls)
_, binary = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
v_binary = np.sum(binary, axis=0).astype(float) / 255
v_bin_smooth = gaussian_filter1d(v_binary, sigma=3)

# Find peaks in each
v_int_peaks, _ = find_peaks(-v_int_smooth, distance=ww // 15, prominence=2)  # dips = walls
v_edge_peaks, _ = find_peaks(v_edge_smooth, distance=ww // 15, height=v_edge_smooth.max() * 0.2)
v_bin_peaks, _ = find_peaks(-v_bin_smooth, distance=ww // 15, prominence=5)  # dips = bright gaps

save(out, "v_intensity.jpg", draw_profile(v_int_smooth, "vertical", bh, bw,
     f"Intensity (inv) — {len(v_int_peaks)} dips", v_int_peaks, (200, 200, 0)))
save(out, "v_edges.jpg", draw_profile(v_edge_smooth, "vertical", bh, bw,
     f"Edge density — {len(v_edge_peaks)} peaks", v_edge_peaks))
save(out, "v_binary.jpg", draw_profile(v_bin_smooth, "vertical", bh, bw,
     f"Binary (inv) — {len(v_bin_peaks)} dips", v_bin_peaks, (200, 100, 0)))

# --- Horizontal projections (find rows) ---

h_intensity = np.mean(work, axis=1)
h_int_smooth = gaussian_filter1d(h_intensity, sigma=2)

h_edges = np.sum(edges, axis=1).astype(float) / 255
h_edge_smooth = gaussian_filter1d(h_edges, sigma=2)

h_binary = np.sum(binary, axis=1).astype(float) / 255
h_bin_smooth = gaussian_filter1d(h_binary, sigma=2)

h_int_peaks, _ = find_peaks(-h_int_smooth, distance=wh // 30, prominence=1)
h_edge_peaks, _ = find_peaks(h_edge_smooth, distance=wh // 30, height=h_edge_smooth.max() * 0.15)
h_bin_peaks, _ = find_peaks(h_bin_smooth, distance=wh // 30, height=h_bin_smooth.max() * 0.15)

save(out, "h_intensity.jpg", draw_profile(h_int_smooth, "horizontal", bh, bw,
     f"Intens. {len(h_int_peaks)} dips", h_int_peaks, (200, 200, 0)))
save(out, "h_edges.jpg", draw_profile(h_edge_smooth, "horizontal", bh, bw,
     f"Edges {len(h_edge_peaks)} peaks", h_edge_peaks))
save(out, "h_binary.jpg", draw_profile(h_bin_smooth, "horizontal", bh, bw,
     f"Binary {len(h_bin_peaks)} peaks", h_bin_peaks, (200, 100, 0)))

# --- Overlay: draw detected peaks as lines on the board image ---
overlay = cv2.cvtColor(work, cv2.COLOR_GRAY2BGR)
for p in v_edge_peaks:
    cv2.line(overlay, (p, 0), (p, wh), (0, 255, 0), 2)
for p in h_edge_peaks:
    cv2.line(overlay, (0, p), (ww, p), (0, 100, 255), 1)
save(out, "grid_from_projections.jpg", overlay)
print(f"  Grid: {len(v_edge_peaks)} cols × {len(h_edge_peaks)} rows")

print(f"\nDone! Check {out}/")
