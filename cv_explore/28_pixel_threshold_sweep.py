"""
28 — Pixel-Level Threshold Sweep
=================================
Work at the raw pixel grid (865×604 = 1 pixel = 1 cell).
Start super strict: only the DARKEST pixels pass.
Gradually relax: lower the darkness bar step by step.

At each threshold T: pixel < T → foreground (dark = plug).
  T=10:  only pitch black passes
  T=50:  very dark
  T=100: moderately dark
  T=150: light-ish
  T=200: almost everything

At each step: count connected blobs, measure sizes, track
how plug structure emerges from the darkness.
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


def sweep_one(args):
    """Worker: threshold at value T, count blobs."""
    gray_path, out_dir, thresh = args
    gray = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
    h, w = gray.shape[:2]

    # Pixel < thresh → foreground (dark pixels are plugs)
    binary = (gray < thresh).astype(np.uint8) * 255

    # Count connected components (no morphology — raw pixels)
    n_cc, labels, stats, cents = cv2.connectedComponentsWithStats(
        binary, connectivity=8)

    if n_cc < 2:
        return {"thresh": thresh, "n_fg": 0, "pct_fg": 0,
                "n_blobs": 0, "med_area": 0, "max_area": 0,
                "n_big": 0, "n_med": 0}

    # Foreground pixel count
    n_fg = int(np.sum(binary > 0))
    pct_fg = n_fg / (h * w) * 100

    areas = np.array([stats[i, cv2.CC_STAT_AREA]
                      for i in range(1, n_cc)])

    # Big blobs: >20 px² (not single-pixel noise)
    big = areas[areas > 20]
    n_big = len(big)
    med_area = float(np.median(big)) if len(big) else 0
    max_area = int(np.max(big)) if len(big) else 0

    # Median-filtered: within [0.3×, 3×] median
    if len(big) >= 3:
        med = np.median(big)
        pluglike = big[(big >= med * 0.3) & (big <= med * 3.0)]
        n_med = len(pluglike)
    else:
        n_med = len(big)

    return {
        "thresh": thresh, "n_fg": n_fg, "pct_fg": pct_fg,
        "n_blobs": n_cc - 1, "n_big": n_big, "n_med": n_med,
        "med_area": med_area, "max_area": max_area,
    }


# ── Main ────────────────────────────────────────────────
if __name__ == "__main__":
    t0 = time.time()

    frame_path = ("/root/.claude/uploads/"
                  "608e2092-c86e-5429-b039-27e3dec388ed/f95b95fa-image.jpg")
    frame = cv2.imread(frame_path)
    print(f"  Frame: {frame.shape[1]}x{frame.shape[0]}")

    out = make_output_dir("28_pixel_thresh_sweep")
    board_raw = crop_board(frame, board_idx=0)
    board_noblue = remove_blue_lines(board_raw)
    bh, bw = board_noblue.shape[:2]
    tx, ty = int(bw * 0.05), int(bh * 0.05)
    board = board_noblue[ty:bh - ty, tx:bw - tx]
    bh, bw = board.shape[:2]
    print(f"  Board: {bw}x{bh} = {bw*bh} pixels")
    save(out, "00_board.jpg", board)

    gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    gray_path = f"{out}/_gray.png"
    cv2.imwrite(gray_path, gray)
    save(out, "01_gray.jpg", gray)

    # Print grayscale stats
    print(f"  Gray range: {gray.min()}-{gray.max()}")
    print(f"  Gray mean: {gray.mean():.1f}  median: "
          f"{np.median(gray):.0f}  std: {gray.std():.1f}")

    # Sweep thresholds: 10, 20, 30, ..., 250
    thresholds = list(range(10, 255, 5))

    worker_args = [(gray_path, out, t) for t in thresholds]
    ncpu = min(cpu_count(), len(thresholds))
    print(f"\n  Sweeping {len(thresholds)} thresholds on "
          f"{ncpu} workers...")
    with Pool(ncpu) as pool:
        results = pool.map(sweep_one, worker_args)

    results.sort(key=lambda r: r["thresh"])
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s\n")

    # ── Table ───────────────────────────────────────────
    print(f"{'='*80}")
    print(f" {'T':>3} | {'%FG':>5} | {'Blobs':>6} | "
          f"{'Big':>5} | {'Med#':>5} | {'MedArea':>7} | "
          f"{'MaxArea':>7}")
    print(f" {'-'*80}")

    for r in results:
        t = r["thresh"]
        if t % 10 != 0 and t not in (55, 65, 75, 85, 95):
            continue  # print every 10, plus some in-between
        print(f" {t:3d} | {r['pct_fg']:5.1f} | "
              f"{r['n_blobs']:6d} | {r['n_big']:5d} | "
              f"{r['n_med']:5d} | {r['med_area']:7.0f} | "
              f"{r['max_area']:7d}")

    # ── Blob emergence chart ────────────────────────────
    print(f"\n{'='*80}")
    print(f"  BLOB EMERGENCE (big blobs >20px²):")
    prev_big = -1
    for r in results:
        t = r["thresh"]
        n = r["n_big"]
        bar = "█" * (n // 4)
        delta = ""
        if prev_big >= 0:
            d = n - prev_big
            if d > 5:
                delta = f" ↑{d}"
            elif d < -5:
                delta = f" ↓{abs(d)}"
        print(f"    T={t:3d}: {n:4d} blobs  "
              f"({r['pct_fg']:4.1f}% fg)  {bar}{delta}")
        prev_big = n

    # ── Find the "plug plateau" ─────────────────────────
    print(f"\n{'='*80}")
    print(f"  MEDIAN-FILTERED BLOB PLATEAUS:")
    meds = [r["n_med"] for r in results]
    run_start = 0
    for i in range(1, len(meds)):
        if abs(meds[i] - meds[run_start]) > max(5, meds[run_start] * 0.15):
            run_len = i - run_start
            if run_len >= 3:
                t0v = results[run_start]["thresh"]
                t1v = results[i - 1]["thresh"]
                n = meds[run_start]
                print(f"    T={t0v}-{t1v}: ~{n} plugs "
                      f"({run_len} steps)")
            run_start = i

    run_len = len(meds) - run_start
    if run_len >= 3:
        t0v = results[run_start]["thresh"]
        t1v = results[-1]["thresh"]
        n = meds[run_start]
        print(f"    T={t0v}-{t1v}: ~{n} plugs ({run_len} steps)")

    # ── Save overlays at key thresholds ─────────────────
    key_t = [30, 50, 70, 90, 110, 130, 150, 170, 200]
    for t in key_t:
        binary = (gray < t).astype(np.uint8) * 255
        # Overlay: red on dark pixels
        vis = board.copy()
        vis[binary > 0] = (vis[binary > 0] * 0.3 +
                           np.array([0, 0, 220]) * 0.7).astype(np.uint8)
        cv2.putText(vis, f"T<{t}  {np.sum(binary>0)/(bw*bh)*100:.1f}%",
                    (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)
        cv2.imwrite(f"{out}/thresh_{t:03d}.jpg", vis)

    print(f"  Overlays saved for T={key_t}")

    # ── Save JSON ───────────────────────────────────────
    with open(f"{out}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone! Check {out}/  ({elapsed:.1f}s)")
