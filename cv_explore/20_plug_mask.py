"""
20 — Plug Mask: 1000×1000 binary mask of plug locations
========================================================
Crop board → resize to 1000×1000 → B&W → adaptive threshold → clean.
White = plug, Black = everything else. Red text stays.
Uses LOCAL adaptive threshold to handle lighting gradient.
"""

from common import make_output_dir, save, crop_board

import cv2
import numpy as np

# ── Load fresh frame, crop board ─────────────────────────
frame_path = "/root/.claude/uploads/608e2092-c86e-5429-b039-27e3dec388ed/f95b95fa-image.jpg"
frame = cv2.imread(frame_path)
print(f"  Frame: {frame.shape[1]}x{frame.shape[0]}")

out = make_output_dir("20_plug_mask")

board = crop_board(frame, board_idx=0)
bh, bw = board.shape[:2]
print(f"  Board: {bw}x{bh}")

# ── Resize to 1000×1000 ─────────────────────────────────
SZ = 1000
board_sq = cv2.resize(board, (SZ, SZ), interpolation=cv2.INTER_AREA)
save(out, "00_board_1000.jpg", board_sq)

# ── Grayscale ────────────────────────────────────────────
gray = cv2.cvtColor(board_sq, cv2.COLOR_BGR2GRAY)

# CLAHE for contrast boost
clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
enhanced = clahe.apply(gray)
save(out, "01_clahe.jpg", enhanced)

# ── Adaptive threshold (local mean) ──────────────────────
# Block size ~51 (must be odd) — local neighborhood
# C=8 means "pixel must be 8 darker than local mean to be foreground"
# We threshold the NON-inverted image: plugs are dark → THRESH_BINARY_INV
adapt = cv2.adaptiveThreshold(enhanced, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV,
                               blockSize=51, C=8)
save(out, "02_adaptive_raw.jpg", adapt)

# ── Morphology: erode to separate touching plugs ─────────
kern_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
eroded = cv2.erode(adapt, kern_erode, iterations=2)

# Close small internal holes
kern_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
closed = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, kern_close)

# Open to remove tiny noise dots
kern_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kern_open)
save(out, "03_morph_clean.jpg", cleaned)

# ── Size-filter blobs ────────────────────────────────────
n_cc, labels, stats, cents = cv2.connectedComponentsWithStats(
    cleaned, connectivity=8)

MIN_AREA = 80
MAX_AREA = 5000
mask = np.zeros((SZ, SZ), dtype=np.uint8)
plug_count = 0
areas = []

for i in range(1, n_cc):
    area = stats[i, cv2.CC_STAT_AREA]
    if MIN_AREA <= area <= MAX_AREA:
        mask[labels == i] = 255
        plug_count += 1
        areas.append(area)

save(out, "04_mask_plugs.jpg", mask)
print(f"  Plug blobs: {plug_count}")

# ── Overlay: green on board ──────────────────────────────
overlay = board_sq.copy()
overlay[mask > 0] = (overlay[mask > 0] * 0.3 +
                      np.array([0, 255, 0]) * 0.7).astype(np.uint8)

# Mark centroids
for i in range(1, n_cc):
    area = stats[i, cv2.CC_STAT_AREA]
    if MIN_AREA <= area <= MAX_AREA:
        cx, cy = int(cents[i][0]), int(cents[i][1])
        cv2.circle(overlay, (cx, cy), 3, (0, 0, 255), -1)

save(out, "05_overlay.jpg", overlay)

# ── Side-by-side ─────────────────────────────────────────
mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
sideby = np.hstack([board_sq, mask_3ch])
save(out, "06_side_by_side.jpg", sideby)

# ── Stats ────────────────────────────────────────────────
areas = np.array(areas)
print(f"\n{'='*50}")
print(f"  1000×1000 plug mask (adaptive threshold)")
print(f"  Raw CCs:          {n_cc - 1}")
print(f"  Plugs (filtered): {plug_count}")
if len(areas):
    print(f"  Area range:       {areas.min()}-{areas.max()} px²")
    print(f"  Area median:      {np.median(areas):.0f} px²")
print(f"  Coverage:         "
      f"{mask.sum() / 255 / (SZ*SZ) * 100:.1f}%")
print(f"\nDone! Check {out}/")
