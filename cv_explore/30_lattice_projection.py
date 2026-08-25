"""
30 — Dual-Projection Peak Lattice
==================================
Step 1: dark mask (gray<70) column projection → 9 strip x-centers
Step 2: per-strip row projection of dark → row y-peaks
Step 3: sub-pixel refine each lattice point via darkness-weighted CoM
Step 4: plastic tube-end edge per strip: mid-band (80–115) column peak,
        |Sobel_x| fallback if weak
Step 5: global lattice fit y = y0_s + row*dy (shared dy) → residual RMS
"""

from common import make_output_dir, save, crop_board

import json
import cv2
import numpy as np
from scipy.signal import find_peaks


def remove_blue_lines(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, (100, 80, 80), (130, 255, 255))
    blue_mask = cv2.dilate(blue_mask, np.ones((3, 3), np.uint8), iterations=1)
    return cv2.inpaint(img, blue_mask, 5, cv2.INPAINT_TELEA)


def smooth(x, k):
    return np.convolve(x.astype(float), np.ones(k) / k, mode="same")


# ── Board prep ─────────────────────────────────────────
frame = cv2.imread("/root/.claude/uploads/"
                   "608e2092-c86e-5429-b039-27e3dec388ed/f95b95fa-image.jpg")
board_raw = crop_board(frame, board_idx=0)
board_noblue = remove_blue_lines(board_raw)
bh, bw = board_noblue.shape[:2]
tx, ty = int(bw * 0.05), int(bh * 0.05)
board = board_noblue[ty:bh - ty, tx:bw - tx]
gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
bh, bw = gray.shape
out = make_output_dir("30_lattice_projection")
print(f"  Board: {bw}x{bh}")

# ── Step 1: strip x-centers from dark column projection ─
dark = (gray < 70).astype(np.uint8)
col_proj = smooth(dark.sum(axis=0), 7)
for hfrac in (0.15, 0.08, 0.04):
    peaks, props = find_peaks(col_proj, distance=bw // 12,
                              height=hfrac * col_proj.max())
    if len(peaks) >= 9:
        break
if len(peaks) > 9:  # keep 9 strongest, restore x order
    keep = np.argsort(props["peak_heights"])[-9:]
    peaks = np.sort(peaks[keep])
strip_cx = peaks.astype(float)
half = float(np.median(np.diff(strip_cx))) / 2
bounds = []
for i, cx in enumerate(strip_cx):
    lo = (strip_cx[i - 1] + cx) / 2 if i > 0 else max(0.0, cx - half)
    hi = (cx + strip_cx[i + 1]) / 2 if i < len(strip_cx) - 1 \
        else min(float(bw), cx + half)
    bounds.append((lo, hi))
print(f"  Strips: {len(strip_cx)} at x={[int(c) for c in strip_cx]}")

# ── Step 2+3: rows per strip, then CoM sub-pixel refine ─
DY0 = 24.9
plugs = []
for s, (cx, (lo, hi)) in enumerate(zip(strip_cx, bounds)):
    row_proj = smooth(dark[:, int(lo):int(hi)].sum(axis=1), 5)
    for pfrac in (0.30, 0.15, 0.05):
        ypk, _ = find_peaks(row_proj, distance=int(0.6 * DY0),
                            prominence=pfrac * row_proj.max())
        if len(ypk) >= 15:
            break
    for y in ypk:
        x0, x1 = int(max(0, cx - 12)), int(min(bw, cx + 13))
        y0, y1 = int(max(0, y - 12)), int(min(bh, y + 13))
        win = gray[y0:y1, x0:x1].astype(float)
        wgt = np.clip(110.0 - win, 0, None)
        wsum = wgt.sum()
        if wsum <= 0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        plugs.append({"strip": s, "cx": float((wgt * xx).sum() / wsum),
                      "cy": float((wgt * yy).sum() / wsum),
                      "wsum": float(wsum)})

# row indices per strip via rounding against nominal spacing
gaps = [b["cy"] - a["cy"] for s in range(len(strip_cx))
        for a, b in zip(*(lambda L: (L, L[1:]))(
            sorted([p for p in plugs if p["strip"] == s],
                   key=lambda p: p["cy"])))]
dy_est = float(np.median([g for g in gaps if 3 < g < 2 * DY0]))
for s in range(len(strip_cx)):
    sp = sorted([p for p in plugs if p["strip"] == s], key=lambda p: p["cy"])
    for p in sp:
        p["row"] = int(round((p["cy"] - sp[0]["cy"]) / dy_est))

# conf = darkness mass relative to median plug
med_w = np.median([p["wsum"] for p in plugs])
for p in plugs:
    p["conf"] = float(np.clip(p["wsum"] / med_w, 0, 1))

# ── Step 4: tube-end plastic edge per strip ────────────
sobel = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
mid = ((gray >= 80) & (gray < 115)).astype(np.uint8)
edge_info = []
for s, (lo, hi) in enumerate(bounds):
    a, b = int(max(0, lo - 5)), int(min(bw, hi + 5))
    mp = smooth(mid[:, a:b].sum(axis=0), 3)
    epk, _ = find_peaks(mp, distance=5)
    method = "midband"
    if len(epk) and mp[epk].max() >= 0.35 * bh:
        ex = a + float(epk[np.argmax(mp[epk])])
    else:  # weak mid-band evidence → gradient fallback
        method = "sobel"
        ex = a + float(np.argmax(smooth(sobel[:, a:b].mean(axis=0), 3)))
    edge_info.append((ex, method))

# ── Step 5: lattice fit y = y0_s + row*dy (shared dy) ──
ns = len(strip_cx)
A = np.zeros((len(plugs), ns + 1))
b_vec = np.array([p["cy"] for p in plugs])
for i, p in enumerate(plugs):
    A[i, p["strip"]] = 1.0
    A[i, ns] = p["row"]
sol, *_ = np.linalg.lstsq(A, b_vec, rcond=None)
dy_fit = float(sol[ns])
residual_rms = float(np.sqrt(np.mean((A @ sol - b_vec) ** 2)))

# ── Results table ──────────────────────────────────────
print(f"\n  {'Strip':>5} | {'x_lo':>5} | {'x_hi':>5} | {'edge_x':>6} | "
      f"{'meth':>7} | {'plugs':>5}")
print(f"  {'-' * 48}")
per_strip = []
for s, ((lo, hi), (ex, meth)) in enumerate(zip(bounds, edge_info)):
    n = sum(1 for p in plugs if p["strip"] == s)
    per_strip.append(n)
    print(f"  S{s:<4} | {lo:>5.0f} | {hi:>5.0f} | {ex:>6.1f} | "
          f"{meth:>7} | {n:>5}")
print(f"\n  Total plugs:  {len(plugs)}")
print(f"  Row spacing:  {dy_fit:.2f} px (median-gap est {dy_est:.2f})")
print(f"  Residual RMS: {residual_rms:.3f} px")

# ── Overlay + JSON ─────────────────────────────────────
vis = board.copy()
for s, ((lo, hi), (ex, _)) in enumerate(zip(bounds, edge_info)):
    cv2.line(vis, (int(ex), 0), (int(ex), bh), (0, 255, 255), 1)
    cv2.line(vis, (int(lo), 0), (int(lo), bh), (0, 128, 0), 1)
    cv2.putText(vis, f"S{s}", (int(strip_cx[s]) - 8, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
for p in plugs:
    cv2.circle(vis, (int(round(p["cx"])), int(round(p["cy"]))), 3,
               (0, 0, 255), -1)
save(out, "05_final_overlay.jpg", vis)

data = {"approach": "dual_projection_peak_lattice",
        "board": {"w": int(bw), "h": int(bh)},
        "row_spacing": dy_fit,
        "strips": [{"idx": s, "x_left": float(lo), "x_right": float(hi),
                    "x_center": float(strip_cx[s]),
                    "edge_x": float(edge_info[s][0])}
                   for s, (lo, hi) in enumerate(bounds)],
        "plugs": [{"strip": p["strip"], "row": p["row"], "cx": p["cx"],
                   "cy": p["cy"], "conf": p["conf"]} for p in plugs]}
with open(f"{out}/positions.json", "w") as f:
    json.dump(data, f, indent=1)
print(f"  → positions.json\nDone! Check {out}/")
