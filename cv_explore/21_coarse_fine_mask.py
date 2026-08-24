"""
21 — Coarse-to-Fine Plug Mask (v3)
====================================
Step 1: 1000×1000 adaptive threshold → harsh B&W mask
Step 2: Size filter only (no circularity — it kills text-overlapped plugs)
Step 3: Local Otsu at original res → precise plug boundary
Step 4: Grid position will filter noise later
"""

from common import make_output_dir, save, crop_board

import cv2
import numpy as np

# ── Load + crop board ────────────────────────────────────
frame_path = "/root/.claude/uploads/608e2092-c86e-5429-b039-27e3dec388ed/f95b95fa-image.jpg"
frame = cv2.imread(frame_path)
print(f"  Frame: {frame.shape[1]}x{frame.shape[0]}")

out = make_output_dir("21_coarse_fine")
board = crop_board(frame, board_idx=0)
bh, bw = board.shape[:2]
print(f"  Board: {bw}x{bh}")

# ══════════════════════════════════════════════════════════
# STEP 1: Harsh 1000×1000 mask
# ══════════════════════════════════════════════════════════
SZ = 1000
board_1k = cv2.resize(board, (SZ, SZ), interpolation=cv2.INTER_AREA)
gray_1k = cv2.cvtColor(board_1k, cv2.COLOR_BGR2GRAY)

clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
enh_1k = clahe.apply(gray_1k)

# Adaptive threshold
adapt = cv2.adaptiveThreshold(enh_1k, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV,
                               blockSize=51, C=8)

# Erode slightly to separate touching blobs, then clean
kern_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
eroded = cv2.erode(adapt, kern_e, iterations=1)
kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
cleaned = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, kern)
cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kern)
save(out, "01_coarse_mask.jpg", cleaned)

# ══════════════════════════════════════════════════════════
# STEP 2: Size filter — just area, keep it simple
# ══════════════════════════════════════════════════════════
n_cc, labels, stats, cents = cv2.connectedComponentsWithStats(
    cleaned, connectivity=8)

MIN_AREA = 100
MAX_AREA = 5000

coarse_plugs = []
for i in range(1, n_cc):
    area = stats[i, cv2.CC_STAT_AREA]
    if MIN_AREA <= area <= MAX_AREA:
        coarse_plugs.append({
            "cx": cents[i][0], "cy": cents[i][1], "area": area,
            "x": stats[i, cv2.CC_STAT_LEFT],
            "y": stats[i, cv2.CC_STAT_TOP],
            "w": stats[i, cv2.CC_STAT_WIDTH],
            "h": stats[i, cv2.CC_STAT_HEIGHT],
        })

print(f"  Coarse plugs (size filter): {len(coarse_plugs)}")

# ══════════════════════════════════════════════════════════
# STEP 3: Fine refinement at ORIGINAL resolution
# ══════════════════════════════════════════════════════════
sx, sy = bw / SZ, bh / SZ
board_gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
clahe_f = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))

fine_plugs = []
PAD = 5

for p in coarse_plugs:
    x0 = max(0, int(p["x"] * sx) - PAD)
    y0 = max(0, int(p["y"] * sy) - PAD)
    x1 = min(bw, int((p["x"] + p["w"]) * sx) + PAD)
    y1 = min(bh, int((p["y"] + p["h"]) * sy) + PAD)

    roi = board_gray[y0:y1, x0:x1]
    if roi.size < 20:
        continue

    roi_enh = clahe_f.apply(roi)
    _, roi_bin = cv2.threshold(roi_enh, 0, 255,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    n2, lab2, st2, cn2 = cv2.connectedComponentsWithStats(
        roi_bin, connectivity=8)
    if n2 < 2:
        continue

    best_i = max(range(1, n2), key=lambda j: st2[j, cv2.CC_STAT_AREA])
    ba = st2[best_i, cv2.CC_STAT_AREA]
    if ba < 30:
        continue

    fine_plugs.append({
        "cx": cn2[best_i][0] + x0,
        "cy": cn2[best_i][1] + y0,
        "area": ba,
        "bx": st2[best_i, cv2.CC_STAT_LEFT] + x0,
        "by": st2[best_i, cv2.CC_STAT_TOP] + y0,
        "bw": st2[best_i, cv2.CC_STAT_WIDTH],
        "bh": st2[best_i, cv2.CC_STAT_HEIGHT],
    })

print(f"  Fine plugs: {len(fine_plugs)}")

# ── Visualize ────────────────────────────────────────────
# Coarse mask (filtered blobs only)
coarse_vis = np.zeros((SZ, SZ), dtype=np.uint8)
for p in coarse_plugs:
    x, y, w, h = int(p["x"]), int(p["y"]), int(p["w"]), int(p["h"])
    x1c, y1c = min(x+w, SZ), min(y+h, SZ)
    coarse_vis[y:y1c, x:x1c] = np.maximum(
        coarse_vis[y:y1c, x:x1c], cleaned[y:y1c, x:x1c])
save(out, "02_coarse_filtered.jpg", coarse_vis)

# Overlay: green blobs + red centroids on board
overlay = board_1k.copy()
overlay[coarse_vis > 0] = (
    overlay[coarse_vis > 0] * 0.4 +
    np.array([0, 220, 0]) * 0.6).astype(np.uint8)

for p in fine_plugs:
    cx = int(p["cx"] / sx)
    cy = int(p["cy"] / sy)
    cv2.circle(overlay, (cx, cy), 3, (0, 0, 255), -1)
save(out, "03_overlay.jpg", overlay)

# Side-by-side: board | coarse mask
sideby = np.hstack([board_1k,
                     cv2.cvtColor(coarse_vis, cv2.COLOR_GRAY2BGR)])
save(out, "04_side_by_side.jpg", sideby)

# ── Stats ────────────────────────────────────────────────
areas = np.array([p["area"] for p in fine_plugs])
print(f"\n{'='*50}")
print(f"  COARSE-TO-FINE (size filter only)")
print(f"  Coarse:  {len(coarse_plugs)} blobs")
print(f"  Fine:    {len(fine_plugs)} plugs")
if len(areas):
    print(f"  Area:    {areas.min()}-{areas.max()} px², "
          f"med={np.median(areas):.0f}")
print(f"\nDone! Check {out}/")
