"""
29 — Brightness Slices: BW mask per narrow brightness band
===========================================================
Split the grayscale range into thin bands (width=2).
Each slice: show ONLY pixels whose brightness falls in that band.
Linear scan from darkest to lightest reveals every structure
at the brightness level it lives at.

Plug cores → dark bands (~40-70)
Strip edges → mid bands (~80-110)
Background → light bands (~130-160)
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


def analyze_band(args):
    """Worker: extract one brightness band and count blobs."""
    gray_path, lo, hi = args
    gray = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
    h, w = gray.shape[:2]

    # Pixels in this brightness band
    band_mask = ((gray >= lo) & (gray < hi)).astype(np.uint8) * 255
    n_px = int(np.sum(band_mask > 0))
    pct = n_px / (h * w) * 100

    if n_px < 5:
        return {"lo": lo, "hi": hi, "n_px": n_px, "pct": pct,
                "n_blobs": 0, "n_big": 0, "med_area": 0,
                "max_area": 0}

    # Connected components — raw, no morphology
    n_cc, labels, stats, cents = cv2.connectedComponentsWithStats(
        band_mask, connectivity=8)

    areas = np.array([stats[i, cv2.CC_STAT_AREA]
                      for i in range(1, n_cc)])

    big = areas[areas > 10]
    n_big = len(big)
    med_area = float(np.median(big)) if len(big) else 0
    max_area = int(np.max(big)) if len(big) else 0

    return {"lo": lo, "hi": hi, "n_px": n_px, "pct": pct,
            "n_blobs": n_cc - 1, "n_big": n_big,
            "med_area": med_area, "max_area": max_area}


# ── Main ────────────────────────────────────────────────
if __name__ == "__main__":
    t0 = time.time()

    frame_path = ("/root/.claude/uploads/"
                  "608e2092-c86e-5429-b039-27e3dec388ed/f95b95fa-image.jpg")
    frame = cv2.imread(frame_path)
    print(f"  Frame: {frame.shape[1]}x{frame.shape[0]}")

    out = make_output_dir("29_brightness_slices")
    board_raw = crop_board(frame, board_idx=0)
    board_noblue = remove_blue_lines(board_raw)
    bh, bw = board_noblue.shape[:2]
    tx, ty = int(bw * 0.05), int(bh * 0.05)
    board = board_noblue[ty:bh - ty, tx:bw - tx]
    bh, bw = board.shape[:2]
    print(f"  Board: {bw}x{bh}")
    save(out, "00_board.jpg", board)

    gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    gray_path = f"{out}/_gray.png"
    cv2.imwrite(gray_path, gray)

    gmin, gmax = int(gray.min()), int(gray.max())
    print(f"  Gray range: {gmin}-{gmax}")

    # Bands of width 2, covering the full range
    band_w = 2
    bands = list(range(gmin, gmax + 1, band_w))
    worker_args = [(gray_path, lo, min(lo + band_w, 256))
                   for lo in bands]

    ncpu = min(cpu_count(), len(bands))
    print(f"  {len(bands)} bands (width={band_w}) on {ncpu} workers...")
    with Pool(ncpu) as pool:
        results = pool.map(analyze_band, worker_args)

    results.sort(key=lambda r: r["lo"])
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s\n")

    # ── Table ───────────────────────────────────────────
    print(f"{'='*75}")
    print(f" {'Band':>7} | {'Pixels':>6} | {'%Img':>5} | "
          f"{'Blobs':>5} | {'Big':>4} | {'MedA':>5} | "
          f"{'MaxA':>6} | Structure")
    print(f" {'-'*75}")

    for r in results:
        lo, hi = r["lo"], r["hi"]
        n_px = r["n_px"]
        pct = r["pct"]
        nb = r["n_big"]
        ma = r["med_area"]
        mx = r["max_area"]

        # Tag structural features
        tag = ""
        if nb >= 100 and ma > 20:
            tag = "◀ PLUG CORES"
        elif nb >= 50 and ma > 20:
            tag = "◀ plugs"
        elif mx > 5000:
            tag = "◀ big merged"
        elif pct > 10:
            tag = "◀ background"

        if n_px > 0:
            print(f" {lo:3d}-{hi:<3d} | {n_px:6d} | {pct:5.1f} | "
                  f"{r['n_blobs']:5d} | {nb:4d} | {ma:5.0f} | "
                  f"{mx:6d} | {tag}")

    # ── Histogram bar chart ─────────────────────────────
    print(f"\n{'='*75}")
    print(f"  BRIGHTNESS HISTOGRAM (pixels per band):")
    max_px = max(r["n_px"] for r in results)
    for r in results:
        if r["n_px"] == 0:
            continue
        lo = r["lo"]
        n = r["n_px"]
        bar_len = int(n / max_px * 50)
        bar = "█" * bar_len
        print(f"    {lo:3d}: {bar} {n}")

    # ── Blob count per band ─────────────────────────────
    print(f"\n{'='*75}")
    print(f"  BIG BLOBS PER BAND (>10px²):")
    max_big = max(r["n_big"] for r in results)
    for r in results:
        nb = r["n_big"]
        if nb == 0:
            continue
        lo = r["lo"]
        bar_len = int(nb / max(max_big, 1) * 50)
        bar = "█" * bar_len
        print(f"    {lo:3d}: {bar} {nb}")

    # ── Save BW slices for key bands ────────────────────
    for r in results:
        lo, hi = r["lo"], r["hi"]
        band_mask = ((gray >= lo) & (gray < hi)).astype(np.uint8) * 255
        # Save every band as a tiny image
        cv2.imwrite(f"{out}/band_{lo:03d}.jpg", band_mask)

    # ── Composite: color-coded by brightness ────────────
    # Stack all bands into an RGB image where hue = brightness
    composite = np.zeros((bh, bw, 3), dtype=np.uint8)
    for r in results:
        lo, hi = r["lo"], r["hi"]
        mask = (gray >= lo) & (gray < hi)
        # Map brightness to hue: dark=red, mid=green, light=blue
        hue = int((lo - gmin) / max(gmax - gmin, 1) * 120)
        color = cv2.cvtColor(np.array([[[hue, 255, 200]]],
                             dtype=np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
        composite[mask] = color
    cv2.imwrite(f"{out}/composite_hue.jpg", composite)

    print(f"\n  All {len(bands)} band images + composite saved")

    # ── Save JSON ───────────────────────────────────────
    with open(f"{out}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone! Check {out}/  ({elapsed:.1f}s)")
