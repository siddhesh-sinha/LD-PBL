"""
24 — Center-Crop Sweep: native-resolution chunks from board center
==================================================================
Take center crops at 1/n of board dimensions (= 1/n² area):
  1/10 dims = 1/100 area, 1/9 = 1/81, ..., 1/1 = full board.
Each crop is at NATIVE pixel resolution — no rescaling.
Multiprocessing blasts all scales in parallel.
At each: remove blue → adaptive threshold → blob stats.
Goal: find where plug count plateaus as the window grows.
"""

from common import make_output_dir, save, crop_board

import cv2
import numpy as np
from multiprocessing import Pool, cpu_count
import json
import time


def remove_blue_lines(img):
    """Remove blue overlay lines via HSV mask + inpaint."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, (100, 80, 80), (130, 255, 255))
    blue_mask = cv2.dilate(blue_mask, np.ones((3, 3), np.uint8),
                           iterations=1)
    return cv2.inpaint(img, blue_mask, 5, cv2.INPAINT_TELEA)


def center_crop(img, frac):
    """Crop center frac of each dimension (frac² of area)."""
    h, w = img.shape[:2]
    cw, ch = int(w * frac), int(h * frac)
    cw, ch = max(cw, 20), max(ch, 20)
    x0 = (w - cw) // 2
    y0 = (h - ch) // 2
    return img[y0:y0 + ch, x0:x0 + cw]


def analyze_crop(args):
    """Worker: run blob detection on one center crop. Pickle-safe."""
    board_path, out_dir, div, board_shape = args
    board = cv2.imread(board_path)
    if board is None:
        return {"div": div, "error": "load failed"}

    frac = 1.0 / div
    crop = center_crop(board, frac)
    ch, cw = crop.shape[:2]
    area_frac = frac * frac

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # CLAHE contrast boost
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enh = clahe.apply(gray)

    # Adaptive threshold — block size scales with crop
    bk = max(11, int(max(cw, ch) * 0.06) | 1)
    adapt = cv2.adaptiveThreshold(enh, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV,
                                   blockSize=bk, C=8)

    # Morphology scaled to image
    k = max(3, min(cw, ch) // 80)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    clean = cv2.morphologyEx(adapt, cv2.MORPH_CLOSE, kern)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kern)

    n_cc, labels, stats, cents = cv2.connectedComponentsWithStats(
        clean, connectivity=8)

    if n_cc < 2:
        return {"div": div, "w": cw, "h": ch, "area_frac": area_frac,
                "n": 0, "med": 0, "mean": 0, "std": 0,
                "n_med": 0, "n_iqr": 0, "q1": 0, "q3": 0}

    areas = np.array([stats[i, cv2.CC_STAT_AREA]
                      for i in range(1, n_cc)])

    # Noise floor: fixed 8 px² minimum (don't scale with image!)
    real = areas[areas >= 8]
    n_real = len(real)

    if n_real < 2:
        return {"div": div, "w": cw, "h": ch, "area_frac": area_frac,
                "n": n_real, "med": 0, "mean": 0, "std": 0,
                "n_med": 0, "n_iqr": 0, "q1": 0, "q3": 0}

    med = float(np.median(real))
    mean = float(np.mean(real))
    std = float(np.std(real))

    # Plug-like: within [0.3×median, 3×median]
    pluglike = real[(real >= med * 0.3) & (real <= med * 3.0)]
    n_med = len(pluglike)
    med_plug = float(np.median(pluglike)) if len(pluglike) else 0

    # IQR-based inliers
    q1, q3 = float(np.percentile(real, 25)), float(np.percentile(real, 75))
    iqr = q3 - q1
    iqr_lo, iqr_hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_iqr = int(np.sum((real >= max(0, iqr_lo)) & (real <= iqr_hi)))

    # Density = blobs per 1000 px²
    density = n_real / (cw * ch) * 1000

    result = {
        "div": div, "w": cw, "h": ch, "area_frac": area_frac,
        "n": n_real, "med": med, "mean": mean, "std": std,
        "n_med": n_med, "med_plug": med_plug,
        "n_iqr": n_iqr, "q1": q1, "q3": q3,
        "density": density,
    }

    # Save overlay for this scale
    vis = crop.copy()
    mask = clean > 0
    vis[mask] = (vis[mask] * 0.4 +
                 np.array([0, 0, 200]) * 0.6).astype(np.uint8)
    lbl = f"1/{div:.1f} {cw}x{ch} n={n_real} med={n_med}"
    cv2.putText(vis, lbl, (3, max(14, ch // 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.25, 0.4 * cw / 400), (255, 255, 255), 1)
    cv2.imwrite(f"{out_dir}/crop_{div:04.1f}.jpg", vis)

    return result


# ── Main ────────────────────────────────────────────────
if __name__ == "__main__":
    t0 = time.time()

    frame_path = ("/root/.claude/uploads/"
                  "608e2092-c86e-5429-b039-27e3dec388ed/f95b95fa-image.jpg")
    frame = cv2.imread(frame_path)
    print(f"  Frame: {frame.shape[1]}x{frame.shape[0]}")

    out = make_output_dir("24_center_crop_sweep")
    board_raw = crop_board(frame, board_idx=0)

    # Remove blue lines, trim 5% border
    board_noblue = remove_blue_lines(board_raw)
    bh, bw = board_noblue.shape[:2]
    tx, ty = int(bw * 0.05), int(bh * 0.05)
    board = board_noblue[ty:bh - ty, tx:bw - tx]
    bh, bw = board.shape[:2]
    print(f"  Board (trimmed, no blue): {bw}x{bh}")

    # Save cleaned board for workers to load
    board_path = f"{out}/_board_clean.jpg"
    cv2.imwrite(board_path, board)
    save(out, "00_board_clean.jpg", board)

    # Divisors: 10, 9, 8, ..., 1  plus half-steps
    divisors = [10, 9, 8, 7, 6, 5, 4, 3.5, 3, 2.5, 2, 1.5, 1]

    # Build worker args
    worker_args = [(board_path, out, d, (bh, bw)) for d in divisors]

    # Parallel execution
    ncpu = min(cpu_count(), len(divisors))
    print(f"\n  Launching {len(divisors)} crops on {ncpu} workers...")
    with Pool(ncpu) as pool:
        results = pool.map(analyze_crop, worker_args)

    # Sort by divisor descending (smallest crop first)
    results.sort(key=lambda r: r["div"], reverse=True)
    elapsed = time.time() - t0
    print(f"  All done in {elapsed:.1f}s")

    # ── Print results table ─────────────────────────────
    print(f"\n{'='*85}")
    print(f" {'1/n':>5} | {'WxH':>10} | {'Area%':>5} | "
          f"{'Real':>4} | {'Med#':>4} | {'IQR#':>4} | "
          f"{'Median':>7} | {'Q1':>6} | {'Q3':>6} | "
          f"{'ρ/kpx':>6}")
    print(f" {'-'*85}")

    for r in results:
        div = r["div"]
        n = r.get("n", 0)
        n_med = r.get("n_med", 0)
        n_iqr = r.get("n_iqr", 0)
        med = r.get("med", 0)
        q1 = r.get("q1", 0)
        q3 = r.get("q3", 0)
        pct = r["area_frac"] * 100
        density = r.get("density", 0)
        print(f" 1/{div:<3.0f} | {r['w']:>4}x{r['h']:<5} | "
              f"{pct:5.1f} | {n:4d} | {n_med:4d} | {n_iqr:4d} | "
              f"{med:7.1f} | {q1:6.1f} | {q3:6.1f} | "
              f"{density:6.2f}")

    # ── Blob count + density progression ────────────────
    print(f"\n{'='*85}")
    print(f"  BLOB COUNT + DENSITY PROGRESSION:")
    for r in results:
        div = r["div"]
        n = r.get("n", 0)
        density = r.get("density", 0)
        bar = "█" * (n // 3)
        frac_str = f"1/{div:<3.0f}"
        dims = f"{r['w']}x{r['h']}"
        print(f"    {frac_str:>6} ({dims:>10}): "
              f"{n:4d} blobs  ρ={density:.2f}/kpx²  {bar}")

    # ── Plateau detection ───────────────────────────────
    print(f"\n  DENSITY PLATEAUS (ρ stable ±20% for 3+ steps):")
    densities = [r.get("density", 0) for r in results]
    run_start = 0
    for i in range(1, len(densities)):
        d0 = densities[run_start]
        if d0 == 0 or abs(densities[i] - d0) / d0 > 0.2:
            run_len = i - run_start
            if run_len >= 3:
                r0 = results[run_start]
                r1 = results[i - 1]
                print(f"    1/{r0['div']:.0f} → 1/{r1['div']:.0f}: "
                      f"ρ≈{d0:.2f}/kpx² ({run_len} steps)")
            run_start = i

    run_len = len(densities) - run_start
    if run_len >= 3:
        r0 = results[run_start]
        r1 = results[-1]
        d0 = densities[run_start]
        print(f"    1/{r0['div']:.0f} → 1/{r1['div']:.0f}: "
              f"ρ≈{d0:.2f}/kpx² ({run_len} steps)")

    # ── Save JSON for later use ─────────────────────────
    json_path = f"{out}/results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {json_path}")
    print(f"\nDone! Check {out}/  ({elapsed:.1f}s total)")
