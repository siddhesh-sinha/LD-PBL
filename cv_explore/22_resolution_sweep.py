"""
22 — Resolution Sweep (v2): median-based blob counting
=======================================================
Resize board to 50×50 → 2000×2000.
At each size: B&W → threshold → count "plug-like" blobs
(within 0.5× to 2× median area, ignoring tiny noise).
Plateaus = structural scales.
"""

from common import make_output_dir, save, crop_board

import cv2
import numpy as np

# ── Load + crop ──────────────────────────────────────────
frame_path = "/root/.claude/uploads/608e2092-c86e-5429-b039-27e3dec388ed/f95b95fa-image.jpg"
frame = cv2.imread(frame_path)
print(f"  Frame: {frame.shape[1]}x{frame.shape[0]}")

out = make_output_dir("22_resolution_sweep")
board = crop_board(frame, board_idx=0)
bh, bw = board.shape[:2]
print(f"  Board: {bw}x{bh}")

# ── Sweep ────────────────────────────────────────────────
sizes = (list(range(50, 300, 25)) +
         list(range(300, 1050, 50)) +
         list(range(1100, 2100, 100)))

results = []

for sz in sizes:
    img = cv2.resize(board, (sz, sz), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enh = clahe.apply(gray)

    bk = max(11, int(sz * 0.05) | 1)
    adapt = cv2.adaptiveThreshold(enh, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV,
                                   blockSize=bk, C=8)

    k = max(3, sz // 200)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    clean = cv2.morphologyEx(adapt, cv2.MORPH_CLOSE, kern)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kern)

    n_cc, labels, stats, cents = cv2.connectedComponentsWithStats(
        clean, connectivity=8)

    if n_cc < 3:
        results.append((sz, 0, 0, 0, 0))
        continue

    areas = np.array([stats[i, cv2.CC_STAT_AREA]
                      for i in range(1, n_cc)])

    # Filter noise: at least 0.01% of image
    min_noise = max(3, sz * sz * 0.0001)
    areas_real = areas[areas >= min_noise]

    if len(areas_real) < 2:
        results.append((sz, len(areas), 0, 0, 0))
        continue

    n_all = len(areas_real)
    med = float(np.median(areas_real))

    # "Plug-like" = within 0.5× to 2× median
    pluglike = areas_real[(areas_real >= med * 0.5) &
                          (areas_real <= med * 2.0)]
    n_plug = len(pluglike)
    med_plug = float(np.median(pluglike)) if len(pluglike) else 0

    results.append((sz, n_all, n_plug, med, med_plug))

    # Save overlay at selected sizes
    if sz % 100 == 0 or sz in (50, 75, 150, 250):
        vis = img.copy()
        mask = clean > 0
        vis[mask] = (vis[mask] * 0.4 +
                     np.array([0, 0, 200]) * 0.6).astype(np.uint8)
        cv2.putText(vis, f"{sz}: all={n_all} plug={n_plug}",
                    (3, max(15, sz // 50)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.25, 0.4 * sz / 500), (255, 255, 255), 1)
        save(out, f"grid_{sz:04d}.jpg", vis)

# ── Print table ──────────────────────────────────────────
print(f"\n{'='*60}")
print(f" {'Size':>5} | {'Real':>5} | {'Plug':>5} | "
      f"{'Median':>7} | {'PlugMed':>7}")
print(f" {'-'*5}-+-{'-'*5}-+-{'-'*5}-+-{'-'*7}-+-{'-'*7}")

prev_plug = -1
for sz, n_all, n_plug, med, med_plug in results:
    tag = ""
    if abs(n_plug - prev_plug) <= 2 and prev_plug > 0:
        tag = " ="  # stable
    elif n_plug > prev_plug + 5:
        tag = " ↑↑"
    elif n_plug < prev_plug - 5:
        tag = " ↓↓"
    print(f" {sz:5d} | {n_all:5d} | {n_plug:5d} | "
          f"{med:7.1f} | {med_plug:7.1f}{tag}")
    prev_plug = n_plug

# ── Find plateaus (plug count stable ±3 for 3+ steps) ───
print(f"\n{'='*60}")
print(f"  PLATEAUS (plug-like count stable ±3 for 3+ steps):")
plugcounts = [r[2] for r in results]
run_start = 0
for i in range(1, len(plugcounts)):
    if abs(plugcounts[i] - plugcounts[run_start]) > 3:
        run_len = i - run_start
        if run_len >= 3:
            s0 = sizes[run_start]
            s1 = sizes[i - 1]
            n = plugcounts[run_start]
            print(f"    {s0:4d}-{s1:4d}: ~{n} plug-like blobs "
                  f"({run_len} steps)")
        run_start = i

run_len = len(plugcounts) - run_start
if run_len >= 3:
    s0 = sizes[run_start]
    s1 = sizes[-1]
    n = plugcounts[run_start]
    print(f"    {s0:4d}-{s1:4d}: ~{n} plug-like blobs "
          f"({run_len} steps)")

print(f"\nDone! Check {out}/")
