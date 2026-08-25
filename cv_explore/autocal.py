"""autocal.py — auto-calibrating two-ended structure core.

Shared by 51 (single frame) and 52 (16-board batch).
DARK sweep rising: compact-blob population plateau → T_plug.
BRIGHT sweep falling: vertical column-runs → T_gap.
Two-ended fusion: plug-column histogram peaks dictate exactly
one gap line between consecutive columns → strips.
"""

import cv2
import numpy as np
from scipy.signal import find_peaks

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
    boards are always valleys in the raw-gray projections."""
    H, W = gray.shape
    glow = cv2.blur(gray, (61, 61))
    m = (glow > 60).astype(np.uint8)
    boxes, queue = [], [(0, 0, W, H)]
    while queue:
        x, y, w, h = queue.pop()
        sub = m[y:y + h, x:x + w]
        if sub.sum() < H * W * 0.02:
            continue
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
    columns. Real gaps are ~25-45px wide; surface flood is much
    wider. Fragmented bright gaps get merged."""
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
    narrow = [(a + b) // 2 for a, b in runs if b - a + 1 <= 45]
    wide = sum(1 for a, b in runs if b - a + 1 > 70)
    merged = []
    for c in sorted(narrow):
        if merged and c - merged[-1][-1] < 25:
            merged[-1].append(c)
        else:
            merged.append([c])
    return [int(np.mean(gr)) for gr in merged], wide


def bright_sweep(g):
    """Falling floor: most clean vertical gap lines before the
    strip surfaces flood in."""
    series = []
    for T in range(200, 99, -5):
        narrow, wide = _gap_runs(g, T)
        series.append({"T": T, "lines": len(narrow), "messy": wide})
    best = max(series, key=lambda s: (s["lines"] - 3 * s["messy"],
                                      s["T"]))
    return best["T"], series


def label_board(g):
    """Run both sweeps on one board crop; return thresholds,
    plugs, fused gap lines, strips, and both sweep series."""
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

    # Plug-column histogram peaks (zero-padded so edge columns fire)
    hist, _ = np.histogram([p[0] for p in plugs],
                           bins=np.arange(0, w + 4, 4))
    hs = np.convolve(hist.astype(float), np.ones(3) / 3, "same")
    hs = np.concatenate([[0, 0, 0], hs, [0, 0, 0]])
    pk, _ = find_peaks(hs, distance=10, height=hs.max() * 0.2)
    col_x = [int((p - 3) * 4 + 2) for p in pk]

    gaps = []
    for a, b in zip(col_x[:-1], col_x[1:]):
        mid = (a + b) // 2
        inside = [c for c in cand if a + 5 < c < b - 5]
        gaps.append(min(inside, key=lambda c: abs(c - mid))
                    if inside else mid)

    edges = ([max(0, 2 * col_x[0] - gaps[0])] + gaps +
             [min(w, 2 * col_x[-1] - gaps[-1])]) if gaps else []
    strips = [(int(a) + 2, int(b) - 2)
              for a, b in zip(edges[:-1], edges[1:])]

    # PER-STRIP REFINEMENT: one global T under-detects in the
    # vignette (a dim edge strip can lose 2/3 of its plugs).
    # Re-read each fenced strip with its own local percentile.
    refined = []
    for (sa, sb) in strips:
        sub = g[:, max(0, sa):sb]
        # Local mini-sweep, PLATEAU objective (max-count invites
        # serration shadows; a dim vignetted strip just needs its
        # own stable threshold): longest run of steady counts.
        detects = []
        for Ts in range(40, 141, 10):
            n2, _, st2, cen2 = cv2.connectedComponentsWithStats(
                (sub < Ts).astype(np.uint8), connectivity=8)
            got = [(int(cen2[i][0]) + max(0, sa),
                    int(cen2[i][1]), int(st2[i, 4]))
                   for i in range(1, n2)
                   if med * 0.3 < st2[i, 4] < med * 3]
            detects.append(got)
        best_run, run = [], []
        for k, got in enumerate(detects):
            n_ = len(got)
            if n_ >= 8 and (not run or
                            abs(n_ - len(detects[run[0]]))
                            <= max(2, len(detects[run[0]]) * 0.12)):
                run.append(k)
            else:
                if len(run) > len(best_run):
                    best_run = run
                run = [k] if n_ >= 8 else []
        if len(run) > len(best_run):
            best_run = run
        if best_run:
            refined.extend(detects[best_run[len(best_run) // 2]])
    if len(refined) >= len(plugs) * 0.8:
        plugs = refined
    return Tp, Tg, plugs, gaps, strips, dser, bser
