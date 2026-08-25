"""
51 — Auto-Calibrating Two-Ended Structure Finder
=================================================
Takes ANY backlit frame. Per board:
  DARK sweep (rising):  track compact-blob population — count,
    size consistency. The plug threshold = center of the most
    stable plateau (count steady, sizes tight).
  BRIGHT sweep (falling): track LONG VERTICAL blob population
    (the gaps between strips). The gap threshold = where the
    count of clean vertical lines peaks before surfaces flood.
Labels both, derives strip rectangles between gap lines.
No hand-tuned thresholds — statistics pick everything.
"""

from common import make_output_dir, save

import cv2
import numpy as np
import json
import sys

FRAME = ("/root/.claude/uploads/608e2092-c86e-5429-b039-27e3dec388ed/"
         "6ef21155-image.png")


def _valley(sums, frac=0.35):
    """Deepest interior valley run in a projection, or None."""
    n, mx = len(sums), sums.max()
    best, i = None, int(n * 0.2)
    while i < int(n * 0.8):
        if sums[i] < mx * frac:
            j = i
            while j < int(n * 0.8) and sums[j] < mx * frac:
                j += 1
            if best is None or (j - i) > (best[1] - best[0]):
                best = (i, j)
            i = j
        else:
            i += 1
    return (best[0] + best[1]) // 2 if best else None


def find_boards(gray):
    """Recursive projection-valley splitting: physical bridges
    (binder clips) defeat erosion, but the dark channels between
    boards are always valleys in the glow projections."""
    H, W = gray.shape
    glow = cv2.blur(gray, (61, 61))
    m = (glow > 60).astype(np.uint8)
    boxes, queue = [], [(0, 0, W, H)]
    while queue:
        x, y, w, h = queue.pop()
        sub = m[y:y + h, x:x + w]
        if sub.sum() < H * W * 0.02:
            continue
        # Valleys on RAW gray sums — a narrow dark channel between
        # boards vanishes in the blurred binary glow mask
        graw = gray[y:y + h, x:x + w].astype(np.int64)
        vx = _valley(graw.sum(axis=0))
        if vx is not None:
            queue += [(x, y, vx, h), (x + vx, y, w - vx, h)]
            continue
        vy = _valley(graw.sum(axis=1))
        if vy is not None:
            queue += [(x, y, w, vy), (x, y + vy, w, h - vy)]
            continue
        cols = np.where(sub.sum(axis=0) > 0.15 * h)[0]
        rows = np.where(sub.sum(axis=1) > 0.15 * w)[0]
        if len(cols) > 50 and len(rows) > 50:
            boxes.append((x + int(cols[0]), y + int(rows[0]),
                          int(cols[-1] - cols[0]),
                          int(rows[-1] - rows[0])))
    return sorted(boxes, key=lambda b: (0 if b[1] < H // 2 else 1, b[0]))


def dark_sweep(g):
    """Rising ceiling: blob stats per T → most stable plateau."""
    series = []
    for T in range(20, 131, 5):
        n, lab, st, _ = cv2.connectedComponentsWithStats(
            (g < T).astype(np.uint8), connectivity=8)
        areas = st[1:, 4]
        areas = areas[(areas >= 15) & (areas <= 3000)]
        med = float(np.median(areas)) if len(areas) else 0
        good = areas[(areas > med * 0.3) & (areas < med * 3)] \
            if med else []
        series.append({"T": T, "n": len(good), "med": med})
    # Longest run of steady counts (±12%), then its center
    best_run, run = [], []
    for s in series:
        if s["n"] >= 30 and (not run or
                             abs(s["n"] - run[0]["n"])
                             <= max(5, run[0]["n"] * 0.12)):
            run.append(s)
        else:
            if len(run) > len(best_run):
                best_run = run
            run = [s] if s["n"] >= 30 else []
    if len(run) > len(best_run):
        best_run = run
    Tp = best_run[len(best_run) // 2]["T"] if best_run else 75
    return Tp, series


def _gap_runs(g, T):
    """Column projection of the bright mask → runs of strong
    columns. Narrow runs = gap lines; wide runs = surface flood.
    (CC geometry fails here: lines fragment vertically.)"""
    h = g.shape[0]
    strong = (g > T).sum(axis=0) > 0.35 * h
    runs, s0 = [], None
    for i, v in enumerate(strong):
        if v and s0 is None:
            s0 = i
        elif not v and s0 is not None:
            runs.append((s0, i - 1))
            s0 = None
    if s0 is not None:
        runs.append((s0, len(strong) - 1))
    # Real gaps are ~25-45px wide; surface flood makes much wider
    narrow = [(a + b) // 2 for a, b in runs if b - a + 1 <= 45]
    wide = sum(1 for a, b in runs if b - a + 1 > 70)
    # Merge lines closer than 25px (one bright gap can fragment)
    merged = []
    for c in sorted(narrow):
        if merged and c - merged[-1][-1] < 25:
            merged[-1].append(c)
        else:
            merged.append([c])
    return [int(np.mean(g)) for g in merged], wide


def bright_sweep(g):
    """Falling floor: track gap-line runs per T; best T = most
    clean narrow lines before surfaces flood in."""
    series = []
    for T in range(200, 99, -5):
        narrow, wide = _gap_runs(g, T)
        series.append({"T": T, "lines": len(narrow), "messy": wide})
    best = max(series, key=lambda s: (s["lines"] - 3 * s["messy"],
                                      s["T"]))
    return best["T"], series


def label_board(g):
    """Run both sweeps, label plugs + gap lines + strips."""
    h, w = g.shape
    Tp, dser = dark_sweep(g)
    Tg, bser = bright_sweep(g)

    n, lab, st, cen = cv2.connectedComponentsWithStats(
        (g < Tp).astype(np.uint8), connectivity=8)
    areas = st[1:, 4]
    med = np.median(areas[(areas >= 15) & (areas <= 3000)])
    plugs = [(int(cen[i][0]), int(cen[i][1]), int(st[i, 4]))
             for i in range(1, n)
             if med * 0.3 < st[i, 4] < med * 3]

    cand, _ = _gap_runs(g, Tg)
    cand = [c for c in cand if 0.03 * w < c < 0.97 * w]

    # TWO-ENDED FUSION: the dark side's plug columns dictate the
    # truth — exactly ONE gap line lives between consecutive
    # columns (bright lanes inside a strip are impostors).
    from scipy.signal import find_peaks
    hist, _ = np.histogram([p[0] for p in plugs],
                           bins=np.arange(0, w + 4, 4))
    hs = np.convolve(hist.astype(float), np.ones(3) / 3, "same")
    hs = np.concatenate([[0, 0, 0], hs, [0, 0, 0]])  # edge peaks
    pk, _ = find_peaks(hs, distance=10, height=hs.max() * 0.2)
    col_x = [int((p - 3) * 4 + 2) for p in pk]

    gaps = []
    for a, b in zip(col_x[:-1], col_x[1:]):
        mid = (a + b) // 2
        inside = [c for c in cand if a + 5 < c < b - 5]
        gaps.append(min(inside, key=lambda c: abs(c - mid))
                    if inside else mid)   # predicted if unseen

    # Strips: one per plug column, spanning gap line to gap line
    edges = [max(0, 2 * col_x[0] - gaps[0])] + gaps + \
            [min(w, 2 * col_x[-1] - gaps[-1])]
    strips = [(int(a) + 2, int(b) - 2)
              for a, b in zip(edges[:-1], edges[1:])]
    return Tp, Tg, plugs, gaps, strips, dser, bser


# ── Main ────────────────────────────────────────────────
out = make_output_dir("51_auto_calibrate")
path = sys.argv[1] if len(sys.argv) > 1 else FRAME
frame = cv2.imread(path)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
boards = find_boards(gray)
print(f"  {len(boards)} boards found")

vis = frame.copy()
report = []
for bi, (bx, by, bw_, bh_) in enumerate(boards):
    g = gray[by:by + bh_, bx:bx + bw_]
    Tp, Tg, plugs, gaps, strips, dser, bser = label_board(g)
    print(f"  Board {bi}: T_plug={Tp} T_gap={Tg}  "
          f"{len(plugs)} plugs, {len(gaps)} gap lines, "
          f"{len(strips)} strips")
    report.append({"board": bi, "box": [bx, by, bw_, bh_],
                   "T_plug": Tp, "T_gap": Tg,
                   "n_plugs": len(plugs), "gaps": gaps,
                   "strips": strips,
                   "dark_series": dser, "bright_series": bser})
    for (px, py, a) in plugs:
        cv2.circle(vis, (bx + px, by + py), 4, (0, 255, 0), 1)
    for gx in gaps:
        cv2.line(vis, (bx + gx, by), (bx + gx, by + bh_),
                 (255, 255, 0), 1)
    for (sa, sb) in strips:
        cv2.rectangle(vis, (bx + sa, by + 5),
                      (bx + sb, by + bh_ - 5), (255, 0, 0), 1)
    cv2.putText(vis, f"B{bi} Tp={Tp} Tg={Tg}", (bx + 5, by - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

save(out, "01_auto_overlay.jpg", vis)
with open(f"{out}/calibration.json", "w") as f:
    json.dump(report, f, default=int)

tot = sum(r["n_plugs"] for r in report)
print(f"\n  TOTAL: {tot} plugs across {len(boards)} boards")
print(f"\nDone! Check {out}/")
