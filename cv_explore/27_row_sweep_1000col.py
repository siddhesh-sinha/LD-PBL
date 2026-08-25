"""
27 — Row Sweep: 1000 vertical slices, subdivide rows 1→60
==========================================================
Think of it as an Excel sheet:
  - 1000 columns (vertical slices) — FIXED
  - Row 1: just 1000 tall cells (full-height strips)
  - Row 2: each strip cut in half → 2000 cells
  - Row 3: cut in thirds → 3000 cells
  - ...keep going...

At each subdivision: what fraction of cells are DECISIVE
(clearly occupied >50% or clearly empty <5%) vs AMBIGUOUS?
When rows match plug spacing → cells snap to full-or-empty.
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


def sweep_one(args):
    """Worker: compute cell occupancy grid for one row count."""
    mask_path, n_rows, n_cols = args
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    h, w = mask.shape[:2]

    cell_w = w / n_cols
    cell_h = h / n_rows
    total_cells = n_rows * n_cols

    # Pre-compute cumulative sum for fast cell sums
    integral = cv2.integral(mask // 255)

    fracs = np.zeros((n_rows, n_cols), dtype=np.float32)
    for r in range(n_rows):
        y0 = int(r * cell_h)
        y1 = int((r + 1) * cell_h)
        for c in range(n_cols):
            x0 = int(c * cell_w)
            x1 = int((c + 1) * cell_w)
            # Sum from integral image
            s = (integral[y1, x1] - integral[y0, x1]
                 - integral[y1, x0] + integral[y0, x0])
            area = max((y1 - y0) * (x1 - x0), 1)
            fracs[r, c] = s / area

    # Decisive: clearly full (>50%) or clearly empty (<5%)
    full = np.sum(fracs > 0.50)
    empty = np.sum(fracs < 0.05)
    ambiguous = total_cells - full - empty
    pct_decisive = (full + empty) / total_cells * 100

    # Bimodality: how separated are occupied vs empty?
    # Ratio of full to ambiguous (higher = crisper grid)
    crispness = full / max(ambiguous, 1)

    # Entropy of occupancy distribution (low = bimodal = good)
    hist, _ = np.histogram(fracs.ravel(), bins=20, range=(0, 1))
    hist_norm = hist / hist.sum()
    entropy = -np.sum(hist_norm[hist_norm > 0]
                      * np.log2(hist_norm[hist_norm > 0]))

    return {
        "n_rows": n_rows,
        "total_cells": total_cells,
        "full": int(full),
        "empty": int(empty),
        "ambiguous": int(ambiguous),
        "pct_decisive": pct_decisive,
        "crispness": crispness,
        "entropy": entropy,
        "fracs": fracs,
    }


# ── Main ────────────────────────────────────────────────
if __name__ == "__main__":
    t0 = time.time()

    frame_path = ("/root/.claude/uploads/"
                  "608e2092-c86e-5429-b039-27e3dec388ed/f95b95fa-image.jpg")
    frame = cv2.imread(frame_path)
    print(f"  Frame: {frame.shape[1]}x{frame.shape[0]}")

    out = make_output_dir("27_row_sweep_1000col")
    board_raw = crop_board(frame, board_idx=0)
    board_noblue = remove_blue_lines(board_raw)
    bh, bw = board_noblue.shape[:2]
    tx, ty = int(bw * 0.05), int(bh * 0.05)
    board = board_noblue[ty:bh - ty, tx:bw - tx]
    bh, bw = board.shape[:2]
    print(f"  Board (trimmed): {bw}x{bh}")
    save(out, "00_board.jpg", board)

    mask = make_mask(board)
    mask_path = f"{out}/_mask.png"
    cv2.imwrite(mask_path, mask)
    save(out, "01_mask.jpg", mask)

    N_COLS = 1000
    row_range = list(range(1, 61))

    # Parallel sweep
    worker_args = [(mask_path, nr, N_COLS) for nr in row_range]
    ncpu = min(cpu_count(), len(row_range))
    print(f"\n  Sweeping 1000 cols × {len(row_range)} row counts "
          f"on {ncpu} workers...")
    with Pool(ncpu) as pool:
        results = pool.map(sweep_one, worker_args)

    results.sort(key=lambda r: r["n_rows"])
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s\n")

    # ── Print table ─────────────────────────────────────
    print(f"{'='*78}")
    print(f" {'Rows':>4} | {'Cells':>6} | {'Full':>5} | "
          f"{'Empty':>6} | {'Ambig':>5} | {'%Decis':>6} | "
          f"{'Crisp':>6} | {'Entrop':>6}")
    print(f" {'-'*78}")

    for r in results:
        nr = r["n_rows"]
        print(f" {nr:4d} | {r['total_cells']:6d} | "
              f"{r['full']:5d} | {r['empty']:6d} | "
              f"{r['ambiguous']:5d} | {r['pct_decisive']:5.1f}% | "
              f"{r['crispness']:6.2f} | {r['entropy']:6.2f}")

    # ── Key metrics progression ─────────────────────────
    print(f"\n{'='*78}")
    print(f"  CRISPNESS (full ÷ ambiguous — higher = grid snaps):")
    best_crisp = max(results, key=lambda r: r["crispness"])
    for r in results:
        nr = r["n_rows"]
        c = r["crispness"]
        bar = "█" * int(c * 10)
        tag = " ◀ BEST" if nr == best_crisp["n_rows"] else ""
        print(f"    {nr:3d} rows: {c:5.2f}  {bar}{tag}")

    print(f"\n  ENTROPY (lower = more bimodal = grid matches plugs):")
    best_ent = min(results, key=lambda r: r["entropy"])
    for r in results:
        nr = r["n_rows"]
        e = r["entropy"]
        bar = "░" * int(e * 3)
        tag = " ◀ BEST" if nr == best_ent["n_rows"] else ""
        print(f"    {nr:3d} rows: {e:5.2f}  {bar}{tag}")

    # ── Heatmaps at key row counts ──────────────────────
    key_rows = [1, 3, 5, 10, 15, 20, 25, 30, 40, 50]
    for r in results:
        if r["n_rows"] not in key_rows:
            continue
        nr = r["n_rows"]
        fracs = r["fracs"]

        # Scale up for visibility
        vis_h = max(nr * 6, 30)
        frac_u8 = (fracs * 255).astype(np.uint8)
        hmap_vis = cv2.resize(frac_u8, (800, vis_h),
                               interpolation=cv2.INTER_NEAREST)
        hmap_color = cv2.applyColorMap(hmap_vis, cv2.COLORMAP_HOT)
        cv2.putText(hmap_color,
                    f"{nr} rows x 1000 cols  "
                    f"crisp={r['crispness']:.2f}  "
                    f"entropy={r['entropy']:.2f}",
                    (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 255), 1)
        cv2.imwrite(f"{out}/hmap_{nr:02d}.jpg", hmap_color)

    print(f"\n  Heatmaps saved for: {key_rows}")

    # ── Summary ─────────────────────────────────────────
    print(f"\n{'='*78}")
    print(f"  SUMMARY:")
    print(f"  Best crispness:  {best_crisp['n_rows']} rows "
          f"(crisp={best_crisp['crispness']:.2f})")
    print(f"  Best entropy:    {best_ent['n_rows']} rows "
          f"(entropy={best_ent['entropy']:.2f})")
    print(f"  → Row grid snaps at ~{best_crisp['n_rows']} "
          f"horizontal subdivisions")
    print(f"  → Cell height at snap: "
          f"{bh / best_crisp['n_rows']:.1f} px")

    # ── Save JSON (without fracs arrays) ────────────────
    slim = [{k: v for k, v in r.items() if k != "fracs"}
            for r in results]
    with open(f"{out}/results.json", "w") as f:
        json.dump(slim, f, indent=2)

    print(f"\nDone! Check {out}/  ({elapsed:.1f}s)")
