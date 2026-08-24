"""
25 — Fibonacci Grid × Center-Crop Sweep
========================================
For each center crop (1/10 → 1/1 dims) × each Fibonacci grid
(1×1, 2×2, 3×3, 5×5, 8×8, 13×13, 21×21, 34×34):
  - Run blob detection on the crop
  - Overlay N×N grid, assign blobs to cells by centroid
  - Report: % cells empty, % with 1 blob, % with 2+, mean/max per cell
When most cells hold exactly 1 blob → grid matches plug structure.
Multiprocessing blasts the full 2D sweep.
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
    """Crop center frac of each dimension."""
    h, w = img.shape[:2]
    cw, ch = max(int(w * frac), 20), max(int(h * frac), 20)
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    return img[y0:y0 + ch, x0:x0 + cw]


def detect_blobs(img):
    """Adaptive threshold → morphology → blob centroids + areas."""
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

    n_cc, labels, stats, cents = cv2.connectedComponentsWithStats(
        clean, connectivity=8)

    blobs = []
    for i in range(1, n_cc):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= 8:  # fixed noise floor
            blobs.append((float(cents[i][0]), float(cents[i][1]), area))

    return blobs, clean


def grid_stats(blobs, w, h, grid_n):
    """Assign blobs to NxN grid cells, return occupancy stats."""
    cell_w = w / grid_n
    cell_h = h / grid_n
    counts = np.zeros((grid_n, grid_n), dtype=int)

    for cx, cy, _ in blobs:
        gi = min(int(cx / cell_w), grid_n - 1)
        gj = min(int(cy / cell_h), grid_n - 1)
        counts[gj, gi] += 1

    total_cells = grid_n * grid_n
    n_empty = int(np.sum(counts == 0))
    n_one = int(np.sum(counts == 1))
    n_multi = int(np.sum(counts >= 2))

    return {
        "empty": n_empty,
        "one": n_one,
        "multi": n_multi,
        "pct_empty": n_empty / total_cells * 100,
        "pct_one": n_one / total_cells * 100,
        "pct_multi": n_multi / total_cells * 100,
        "mean": float(np.mean(counts)),
        "max": int(np.max(counts)),
        "std": float(np.std(counts)),
        "counts": counts,
    }


def worker(args):
    """Process one (crop_div, grid_n) combo."""
    board_path, out_dir, div, grid_n = args
    board = cv2.imread(board_path)
    if board is None:
        return None

    frac = 1.0 / div
    crop = center_crop(board, frac)
    ch, cw = crop.shape[:2]
    blobs, mask = detect_blobs(crop)
    gs = grid_stats(blobs, cw, ch, grid_n)

    result = {
        "div": div, "grid": grid_n,
        "w": cw, "h": ch,
        "n_blobs": len(blobs),
        "pct_empty": gs["pct_empty"],
        "pct_one": gs["pct_one"],
        "pct_multi": gs["pct_multi"],
        "mean_per_cell": gs["mean"],
        "max_per_cell": gs["max"],
        "std_per_cell": gs["std"],
    }

    # Save overlay for selected combos
    key_divs = [10, 5, 3, 2, 1]
    key_grids = [1, 3, 5, 8, 13, 21, 34]
    if div in key_divs and grid_n in key_grids:
        vis = crop.copy()
        # Draw blobs
        bmask = mask > 0
        vis[bmask] = (vis[bmask] * 0.5 +
                      np.array([0, 0, 180]) * 0.5).astype(np.uint8)
        # Draw grid
        cell_w = cw / grid_n
        cell_h = ch / grid_n
        for i in range(1, grid_n):
            x = int(i * cell_w)
            y = int(i * cell_h)
            cv2.line(vis, (x, 0), (x, ch), (0, 255, 0), 1)
            cv2.line(vis, (0, y), (cw, y), (0, 255, 0), 1)
        # Color cells by occupancy
        for gj in range(grid_n):
            for gi in range(grid_n):
                cnt = gs["counts"][gj, gi]
                if cnt == 1:
                    # green tint — perfect
                    x0 = int(gi * cell_w)
                    y0 = int(gj * cell_h)
                    x1 = int((gi + 1) * cell_w)
                    y1 = int((gj + 1) * cell_h)
                    roi = vis[y0:y1, x0:x1]
                    vis[y0:y1, x0:x1] = (roi * 0.7 +
                        np.array([0, 80, 0]) * 0.3).astype(np.uint8)
                elif cnt >= 2:
                    # yellow tint — overcrowded
                    x0 = int(gi * cell_w)
                    y0 = int(gj * cell_h)
                    x1 = int((gi + 1) * cell_w)
                    y1 = int((gj + 1) * cell_h)
                    roi = vis[y0:y1, x0:x1]
                    vis[y0:y1, x0:x1] = (roi * 0.7 +
                        np.array([0, 60, 80]) * 0.3).astype(np.uint8)

        lbl = (f"1/{div} {grid_n}x{grid_n} "
               f"e={gs['pct_empty']:.0f}% "
               f"1={gs['pct_one']:.0f}% "
               f"2+={gs['pct_multi']:.0f}%")
        cv2.putText(vis, lbl, (2, max(12, ch // 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.2, 0.35 * cw / 400), (255, 255, 255), 1)
        cv2.imwrite(f"{out_dir}/d{div:04.1f}_g{grid_n:02d}.jpg", vis)

    return result


# ── Main ────────────────────────────────────────────────
if __name__ == "__main__":
    t0 = time.time()

    frame_path = ("/root/.claude/uploads/"
                  "608e2092-c86e-5429-b039-27e3dec388ed/f95b95fa-image.jpg")
    frame = cv2.imread(frame_path)
    print(f"  Frame: {frame.shape[1]}x{frame.shape[0]}")

    out = make_output_dir("25_fib_grid_sweep")
    board_raw = crop_board(frame, board_idx=0)
    board_noblue = remove_blue_lines(board_raw)
    bh, bw = board_noblue.shape[:2]
    tx, ty = int(bw * 0.05), int(bh * 0.05)
    board = board_noblue[ty:bh - ty, tx:bw - tx]
    bh, bw = board.shape[:2]
    print(f"  Board (trimmed): {bw}x{bh}")

    board_path = f"{out}/_board_clean.jpg"
    cv2.imwrite(board_path, board)

    # Fibonacci grid sizes
    fibs = [1, 2, 3, 5, 8, 13, 21, 34]
    # Center crop divisors
    divs = [10, 8, 6, 5, 4, 3, 2, 1.5, 1]

    # Build all combos
    combos = [(board_path, out, d, g) for d in divs for g in fibs]
    print(f"  {len(combos)} combos ({len(divs)} crops × {len(fibs)} grids)")

    ncpu = min(cpu_count(), len(combos))
    print(f"  Launching on {ncpu} workers...")
    with Pool(ncpu) as pool:
        raw_results = pool.map(worker, combos)

    results = [r for r in raw_results if r is not None]
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s\n")

    # ── Print: for each crop, sweep grids ───────────────
    print(f"{'='*90}")
    print(f" GRID FIT QUALITY: %cells with exactly 1 blob "
          f"(green = sweet spot)")
    print(f"{'='*90}")
    print(f" {'Crop':>6} {'WxH':>10} {'#blob':>5} | ", end="")
    for g in fibs:
        print(f" {g:>3}×{g:<3}", end="")
    print()
    print(f" {'-'*36}+{'-' * (8 * len(fibs))}")

    for d in divs:
        crop_res = [r for r in results if r["div"] == d]
        if not crop_res:
            continue
        r0 = crop_res[0]
        print(f" 1/{d:<4} {r0['w']:>4}x{r0['h']:<5} {r0['n_blobs']:>5} | ",
              end="")
        for g in fibs:
            gr = next((r for r in crop_res if r["grid"] == g), None)
            if gr:
                p1 = gr["pct_one"]
                # Highlight sweet spots
                if p1 >= 40:
                    marker = "██"
                elif p1 >= 25:
                    marker = "▓▓"
                elif p1 >= 15:
                    marker = "░░"
                else:
                    marker = "  "
                print(f" {p1:4.0f}%{marker}", end="")
            else:
                print(f"    ?  ", end="")
        print()

    # ── Print: %multi (overcrowded cells) ───────────────
    print(f"\n{'='*90}")
    print(f" OVERCROWDED CELLS: %cells with 2+ blobs")
    print(f"{'='*90}")
    print(f" {'Crop':>6} {'#blob':>5} | ", end="")
    for g in fibs:
        print(f" {g:>3}×{g:<3}", end="")
    print()
    print(f" {'-'*14}+{'-' * (8 * len(fibs))}")

    for d in divs:
        crop_res = [r for r in results if r["div"] == d]
        if not crop_res:
            continue
        r0 = crop_res[0]
        print(f" 1/{d:<4} {r0['n_blobs']:>5} | ", end="")
        for g in fibs:
            gr = next((r for r in crop_res if r["grid"] == g), None)
            if gr:
                pm = gr["pct_multi"]
                print(f" {pm:5.1f}%", end="")
            else:
                print(f"     ? ", end="")
        print()

    # ── Best grid per crop ──────────────────────────────
    print(f"\n{'='*90}")
    print(f" BEST GRID per crop (highest %one-blob cells):")
    for d in divs:
        crop_res = [r for r in results if r["div"] == d]
        if not crop_res:
            continue
        best = max(crop_res, key=lambda r: r["pct_one"])
        g = best["grid"]
        print(f"   1/{d:<4} ({best['w']}x{best['h']}): "
              f"best = {g}×{g} grid → "
              f"{best['pct_one']:.0f}% one-blob, "
              f"{best['pct_empty']:.0f}% empty, "
              f"{best['pct_multi']:.0f}% multi "
              f"({best['n_blobs']} blobs)")

    # ── Save JSON ───────────────────────────────────────
    with open(f"{out}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone! Check {out}/  ({elapsed:.1f}s)")
