"""
23 — Scale Sweep: 1/n² area fractions, blob stats at each
===========================================================
Remove blue lines, trim borders, then resize to 1/10, 1/9, ..., 1/1
of original dimensions (= 1/100, 1/81, ..., 1/1 of area).
At each scale: threshold → blob count + full area stats.
Find where plugs resolve and the count stabilizes.
"""

from common import make_output_dir, save, crop_board

import cv2
import numpy as np


def remove_blue_lines(img):
    """Remove ONLY the blue overlay lines (keep red text)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Blue range in HSV
    blue_mask = cv2.inRange(hsv, (100, 80, 80), (130, 255, 255))
    blue_mask = cv2.dilate(blue_mask, np.ones((3, 3), np.uint8),
                           iterations=1)
    return cv2.inpaint(img, blue_mask, 5, cv2.INPAINT_TELEA)


# ── Load + crop + clean ──────────────────────────────────
frame_path = "/root/.claude/uploads/608e2092-c86e-5429-b039-27e3dec388ed/f95b95fa-image.jpg"
frame = cv2.imread(frame_path)
print(f"  Frame: {frame.shape[1]}x{frame.shape[0]}")

out = make_output_dir("23_scale_sweep")
board_raw = crop_board(frame, board_idx=0)

# Remove blue lines
board_noblue = remove_blue_lines(board_raw)
save(out, "00_no_blue.jpg", board_noblue)

# Trim border — 5% each side to kill edge clutter
bh, bw = board_noblue.shape[:2]
tx, ty = int(bw * 0.05), int(bh * 0.05)
board = board_noblue[ty:bh - ty, tx:bw - tx]
bh, bw = board.shape[:2]
print(f"  Board (trimmed, no blue): {bw}x{bh}")
save(out, "00_trimmed.jpg", board)

# ── Sweep: 1/n of original dims → 1/n² area ─────────────
# n = 10, 9, 8, ..., 1 (and a few upscaled: 1.5×, 2×)
divisors = [10, 9, 8, 7, 6, 5, 4, 3, 2.5, 2, 1.5, 1]
results = []

for div in divisors:
    sw = max(20, int(bw / div))
    sh = max(20, int(bh / div))
    area_frac = 1.0 / (div * div)

    img = cv2.resize(board, (sw, sh), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enh = clahe.apply(gray)

    bk = max(11, int(max(sw, sh) * 0.06) | 1)
    adapt = cv2.adaptiveThreshold(enh, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV,
                                   blockSize=bk, C=8)

    # Morphology scaled to image
    k = max(3, min(sw, sh) // 80)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    clean = cv2.morphologyEx(adapt, cv2.MORPH_CLOSE, kern)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kern)

    n_cc, labels, stats, cents = cv2.connectedComponentsWithStats(
        clean, connectivity=8)

    if n_cc < 2:
        results.append({"div": div, "w": sw, "h": sh,
                         "area_frac": area_frac, "n": 0})
        continue

    areas = np.array([stats[i, cv2.CC_STAT_AREA]
                      for i in range(1, n_cc)])

    # Remove noise floor: < 0.05% of image
    min_noise = max(2, sw * sh * 0.0005)
    real = areas[areas >= min_noise]
    n_real = len(real)

    if n_real < 2:
        results.append({"div": div, "w": sw, "h": sh,
                         "area_frac": area_frac, "n": n_real,
                         "med": 0, "mean": 0, "std": 0,
                         "n_med": 0})
        continue

    med = float(np.median(real))
    mean = float(np.mean(real))
    std = float(np.std(real))

    # Blobs within [0.3×median, 3×median] — "plug-like"
    pluglike = real[(real >= med * 0.3) & (real <= med * 3.0)]
    n_med = len(pluglike)
    med_plug = float(np.median(pluglike)) if len(pluglike) else 0

    # IQR-based: blobs within Q1-1.5*IQR to Q3+1.5*IQR
    q1, q3 = np.percentile(real, 25), np.percentile(real, 75)
    iqr = q3 - q1
    iqr_lo, iqr_hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_iqr = int(np.sum((real >= max(0, iqr_lo)) & (real <= iqr_hi)))

    results.append({
        "div": div, "w": sw, "h": sh, "area_frac": area_frac,
        "n": n_real, "med": med, "mean": mean, "std": std,
        "n_med": n_med, "med_plug": med_plug,
        "n_iqr": n_iqr, "q1": q1, "q3": q3,
    })

    # Save overlay
    vis = img.copy()
    mask = clean > 0
    vis[mask] = (vis[mask] * 0.4 +
                 np.array([0, 0, 200]) * 0.6).astype(np.uint8)
    lbl = f"1/{div:.0f} {sw}x{sh} n={n_real} med={n_med} iqr={n_iqr}"
    cv2.putText(vis, lbl, (3, max(12, sh // 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.2, 0.3 * sw / 300), (255, 255, 255), 1)
    save(out, f"scale_{div:04.1f}.jpg", vis)

# ── Print results ────────────────────────────────────────
print(f"\n{'='*78}")
print(f" {'1/n':>5} | {'WxH':>9} | {'Area%':>5} | {'Real':>4} | "
      f"{'Med#':>4} | {'IQR#':>4} | {'Median':>7} | "
      f"{'Q1':>6} | {'Q3':>6}")
print(f" {'-'*78}")

for r in results:
    div = r["div"]
    n = r.get("n", 0)
    n_med = r.get("n_med", 0)
    n_iqr = r.get("n_iqr", 0)
    med = r.get("med", 0)
    q1 = r.get("q1", 0)
    q3 = r.get("q3", 0)
    pct = r["area_frac"] * 100
    print(f" 1/{div:<3.0f} | {r['w']:>4}x{r['h']:<4} | {pct:5.1f} | "
          f"{n:4d} | {n_med:4d} | {n_iqr:4d} | "
          f"{med:7.1f} | {q1:6.1f} | {q3:6.1f}")

# ── Key finding: where does Real count stabilize? ────────
print(f"\n{'='*78}")
reals = [(r["div"], r.get("n", 0)) for r in results]
print(f"  BLOB COUNT PROGRESSION:")
for div, n in reals:
    bar = "█" * (n // 5)
    print(f"    1/{div:<3.0f}: {n:4d}  {bar}")

print(f"\nDone! Check {out}/")
