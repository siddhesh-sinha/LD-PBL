"""
48 — Strip & Tube Structure Extraction
=======================================
Layer-by-layer geometric model of the board:
  1. Per-plug enclosing geometry: circumcircle + bounding square
     (from script 47's outlines) → size statistics
  2. Vertical spacing statistics per strip → predicted horizontal
     tube-separator lines (midpoints; big gaps get filled at the
     median pitch)
  3. Per-strip inscribed rectangle (plastic extent from mid-band)
  4. Jagged TRUE boundaries: per-row edge scan down each side of
     the strip (the plastic is cut roughly — same refinement idea
     as the plug outlines, but vertical)
Output: structure.json — the full skeleton of strips and tubes.
"""

from common import make_output_dir, save, crop_board
from crux import (FRAME_PATH, remove_blue_lines, lock_lattice,
                  mid_mask, pick_threshold)

import cv2
import numpy as np
import json
import os

out = make_output_dir("48_strip_structure")
frame = cv2.imread(FRAME_PATH)
board_full = remove_blue_lines(crop_board(frame, board_idx=0))
h0, w0 = board_full.shape[:2]
tx, ty = int(w0 * 0.05), int(h0 * 0.05)
board = board_full[ty:h0 - ty, tx:w0 - tx]
gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
h, w = gray.shape[:2]

lat = lock_lattice(gray)
T, dy = lat["T"], lat["dy"]
mid = mid_mask(gray, T)

# ── 1. Per-plug enclosing geometry ──────────────────────
shapes_path = os.path.join(os.path.dirname(out),
                           "47_plug_shapes", "shapes.json")
with open(shapes_path) as f:
    shapes = json.load(f)

for s in shapes:
    poly = np.array(s["poly"], dtype=np.int32)
    (ccx, ccy), cr = cv2.minEnclosingCircle(poly)
    bx, by, bw, bh = cv2.boundingRect(poly)
    s["circ"] = (int(round(ccx)), int(round(ccy)), int(np.ceil(cr)))
    s["sq"] = int(max(bw, bh))            # circumscribed square side

crs = [s["circ"][2] for s in shapes]
sqs = [s["sq"] for s in shapes]
print(f"  Plug circumcircles: r med={np.median(crs):.0f} "
      f"q1-q3={np.percentile(crs,25):.0f}-{np.percentile(crs,75):.0f}")
print(f"  Circumscribed square side: med={np.median(sqs):.0f} "
      f"max={max(sqs)}")

# ── 2. Spacing statistics → tube separator lines ────────
by_strip = {}
for s in shapes:
    by_strip.setdefault(s["strip"], []).append(s)

all_gaps = []
strip_model = []
for si in sorted(by_strip):
    plugs = sorted(by_strip[si], key=lambda s: s["cy"])
    cys = [p["cy"] for p in plugs]
    gaps = np.diff(cys)
    med_gap = float(np.median(gaps)) if len(gaps) else dy
    all_gaps.extend(gaps)

    # Separators at midpoints; fill big gaps at the median pitch
    seps = []
    for a, b in zip(cys[:-1], cys[1:]):
        g = b - a
        n = max(1, int(round(g / med_gap)))
        for j in range(1, n + 1):
            seps.append(int(round(a + g * (2 * j - 1) / (2 * n))))
    strip_model.append({"strip": si, "plugs": plugs,
                        "med_gap": med_gap, "gap_std":
                        float(np.std(gaps)) if len(gaps) else 0,
                        "separators": seps})

print(f"\n  Vertical spacing: global med={np.median(all_gaps):.1f}px "
      f"std={np.std(all_gaps):.1f}")
for m in strip_model:
    print(f"    S{m['strip']}: {len(m['plugs'])} plugs  "
          f"gap {m['med_gap']:.1f}±{m['gap_std']:.1f}px  "
          f"{len(m['separators'])} tube lines")

# ── 3+4. Strip rectangles + jagged true boundaries ──────
# The strip's flat surface reflects as bright as the gap — only
# its cut BEVEL edges show as thin mid-tone lines (script 29's
# 80-115 band). Strip boundary = the leftmost/rightmost bevel line
# in the strip window, traced per-row for the jagged truth.
from scipy.signal import find_peaks as _fp


def edge_scan(x_lo, x_hi, y_top, y_bot):
    """Find the two bevel lines from the mid-band column profile,
    then trace each per-row (nearest mid pixel within ±6px) →
    jagged left/right polylines, median-filtered."""
    prof = mid[y_top:y_bot, x_lo:x_hi].sum(axis=0).astype(float)
    prof = np.convolve(prof, np.ones(5) / 5, mode="same")
    pk, _ = _fp(prof, height=prof.max() * 0.25, distance=8)
    if len(pk) >= 2:
        lx, rx = x_lo + int(pk[0]), x_lo + int(pk[-1])
    else:
        lx, rx = x_lo + 3, x_hi - 3

    def trace(line_x):
        ys = []
        for y in range(y_top, y_bot):
            seg = mid[y, max(0, line_x - 6):line_x + 7]
            on = np.where(seg > 0)[0]
            ys.append(max(0, line_x - 6) + int(on[len(on) // 2])
                      if len(on) else line_x)
        k = 9
        return np.array([np.median(ys[max(0, i - k):i + k + 1])
                         for i in range(len(ys))], dtype=int)

    return trace(lx), trace(rx)


structure = []
for m in strip_model:
    si = m["strip"]
    st = lat["strips"][si]
    yt, yb = st["y_top"], st["y_bot"]
    Lf, Rf = edge_scan(st["x0"], st["x1"], yt, yb)

    # Robust inscribed rectangle: inside the jagged edges for 90%
    # of rows (strict max/min collapses on a single outlier row)
    rect = (int(np.percentile(Lf, 90)), yt,
            int(np.percentile(Rf, 10)), yb)
    jag_l = float(np.std(Lf))
    jag_r = float(np.std(Rf))

    structure.append({
        "strip": si, "rect": rect,
        "y_top": yt, "y_bot": yb,
        "left_edge": Lf.tolist(), "right_edge": Rf.tolist(),
        "jaggedness": (round(jag_l, 1), round(jag_r, 1)),
        "separators": m["separators"],
        "med_gap": round(m["med_gap"], 1),
        "plugs": [{"cx": p["cx"], "cy": p["cy"],
                   "circ": p["circ"], "sq": p["sq"]}
                  for p in m["plugs"]],
    })
    print(f"    S{si} rect x=[{rect[0]},{rect[2]}] "
          f"w={rect[2]-rect[0]} jag L/R={jag_l:.1f}/{jag_r:.1f}px")

# ── Overlay ─────────────────────────────────────────────
vis = board.copy()
for s in structure:
    x0r, ytr, x1r, ybr = s["rect"]
    yt = s["y_top"]
    # jagged boundaries (cyan)
    for i in range(1, len(s["left_edge"])):
        cv2.line(vis, (s["left_edge"][i - 1], yt + i - 1),
                 (s["left_edge"][i], yt + i), (255, 255, 0), 1)
        cv2.line(vis, (s["right_edge"][i - 1], yt + i - 1),
                 (s["right_edge"][i], yt + i), (255, 255, 0), 1)
    # inscribed rectangle (blue)
    cv2.rectangle(vis, (x0r, ytr), (x1r, ybr), (255, 0, 0), 1)
    # tube separator lines (yellow)
    for sy in s["separators"]:
        cv2.line(vis, (x0r, sy), (x1r, sy), (0, 220, 220), 1)
    # plug circumcircles (green)
    for p in s["plugs"]:
        ccx, ccy, cr = p["circ"]
        cv2.circle(vis, (ccx, ccy), cr, (0, 255, 0), 1)
save(out, "01_structure_overlay.jpg", vis)

with open(f"{out}/structure.json", "w") as f:
    json.dump(structure, f)

n_tubes = sum(len(s["separators"]) + 1 for s in structure)
print(f"\n  STRUCTURE: {len(structure)} strips, {n_tubes} tubes, "
      f"{sum(len(s['plugs']) for s in structure)} plugs")
print(f"\nDone! Check {out}/")
