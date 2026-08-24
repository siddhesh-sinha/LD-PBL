"""
19 — Blur From One Blob (v4): Raw grayscale, no CLAHE, iterative blur
=====================================================================
The problem: CLAHE amplifies edges → extra blobs at high σ.
            Kernel capped by image size → not truly massive blur.
Fix: use RAW grayscale + iterative Gaussian (blur the blur) so very
     high σ actually merges everything. Red text stays in.
"""

from common import make_output_dir, save, crop_board

import cv2
import numpy as np

# ── Load fresh frame, crop board (NO red removal) ───────
frame_path = "/root/.claude/uploads/608e2092-c86e-5429-b039-27e3dec388ed/f95b95fa-image.jpg"
frame = cv2.imread(frame_path)
fh, fw = frame.shape[:2]
print(f"  Frame: {fw}x{fh}")

out = make_output_dir("19_blur_from_one")

# Crop board 0 — raw, red overlays intact
board_full = crop_board(frame, board_idx=0)

# Trim dark border edges (4% each side)
bfh, bfw = board_full.shape[:2]
tx, ty = int(bfw * 0.04), int(bfh * 0.04)
board = board_full[ty:bfh - ty, tx:bfw - tx]
bh, bw = board.shape[:2]
print(f"  Board (trimmed): {bw}x{bh}")
save(out, "00_board.jpg", board)

# ── TWO contrast modes: raw grayscale vs CLAHE ──────────
board_gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
board_clahe = clahe.apply(board_gray)
save(out, "01a_gray.jpg", board_gray)
save(out, "01b_clahe.jpg", board_clahe)


def iterative_blur(img_f32, sigma):
    """Gaussian blur via iteration: split into passes of σ≤50."""
    result = img_f32.copy()
    remaining = sigma
    while remaining > 0:
        s = min(remaining, 50.0)
        k = int(s * 6) | 1
        result = cv2.GaussianBlur(result, (k, k), s)
        remaining -= s
    return result


def run_descent(name, inv):
    """Run blur descent on an inverted float32 image."""
    h, w = inv.shape[:2]
    print(f"\n  === {name} ===")

    sigmas = (list(range(500, 100, -20)) +
              list(range(100, 50, -5)) +
              list(range(50, 20, -2)) +
              list(range(20, 5, -1)) +
              [4, 3, 2, 1])

    prev_n = -1
    transitions = []

    for sigma in sigmas:
        blurred = iterative_blur(inv, sigma)

        # Fixed mean threshold
        binary = (blurred > blurred.mean()).astype(np.uint8) * 255

        # Count blobs
        n_cc, labels, stats, cents = cv2.connectedComponentsWithStats(
            binary, connectivity=8)
        min_area = w * h * 0.001
        big = [(i, stats[i, cv2.CC_STAT_AREA])
               for i in range(1, n_cc)
               if stats[i, cv2.CC_STAT_AREA] > min_area]
        n_big = len(big)

        if n_big != prev_n:
            transitions.append((sigma, n_big))
            print(f"  σ={sigma:4d}: {n_big:3d} blobs ← NEW")

            # Save overlay
            overlay = board.copy()
            mask = binary > 0
            overlay[mask] = (overlay[mask] * 0.4 +
                             np.array([0, 0, 200]) * 0.6).astype(np.uint8)
            for idx, area in big:
                cx, cy = int(cents[idx][0]), int(cents[idx][1])
                cv2.circle(overlay, (cx, cy), 5, (0, 255, 255), -1)
            cv2.putText(overlay, f"s={sigma} n={n_big}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)
            tag = f"{name}_s{sigma:04d}_n{n_big:03d}.jpg"
            save(out, tag, overlay)

        prev_n = n_big

    return transitions


# ── Run both modes ───────────────────────────────────────
inv_raw = (255 - board_gray).astype(np.float32)
inv_clahe = (255 - board_clahe).astype(np.float32)

t_raw = run_descent("RAW", inv_raw)
t_clahe = run_descent("CLAHE", inv_clahe)

# ── Compare ──────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  {'σ':>5}  {'RAW':>5}  {'CLAHE':>6}")
print(f"  {'-'*5}  {'-'*5}  {'-'*6}")
ri, ci = 0, 0
all_sigmas = sorted(set(s for s, _ in t_raw + t_clahe), reverse=True)
raw_n, clahe_n = 0, 0
for s in all_sigmas:
    for sig, n in t_raw:
        if sig == s:
            raw_n = n
    for sig, n in t_clahe:
        if sig == s:
            clahe_n = n
    # Only print when either changed at this sigma
    if any(sig == s for sig, _ in t_raw) or any(sig == s for sig, _ in t_clahe):
        r_mark = "←" if any(sig == s for sig, _ in t_raw) else " "
        c_mark = "←" if any(sig == s for sig, _ in t_clahe) else " "
        print(f"  {s:5d}  {raw_n:4d}{r_mark}  {clahe_n:4d}{c_mark}")

print(f"\nDone! Check {out}/")
