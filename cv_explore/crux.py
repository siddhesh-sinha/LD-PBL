"""crux.py — shared core for the 5 positioning SYSTEMS (scripts 40-44).

Integer-pixel contract: positions are ints, masks are binary.
Sub-pixel is fiction at this resolution — on/off per pixel is
the floor of knowledge (Nyquist: structure <2px is unknowable).

Provides: board acquisition, histogram-derived threshold,
lattice lock (proven dual-projection method from scripts 30/31),
integer shift tracking, cell reading, crispness health metric.
"""

from common import crop_board

import cv2
import numpy as np
from scipy.signal import find_peaks

FRAME_PATH = ("/root/.claude/uploads/"
              "608e2092-c86e-5429-b039-27e3dec388ed/f95b95fa-image.jpg")


def remove_blue_lines(img):
    """Remove blue overlay lines via HSV mask + inpaint."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, (100, 80, 80), (130, 255, 255))
    blue = cv2.dilate(blue, np.ones((3, 3), np.uint8), iterations=1)
    return cv2.inpaint(img, blue, 5, cv2.INPAINT_TELEA)


def acquire_board(frame):
    """Frame → clean board (blue removed, 5% trim). Returns bgr, gray."""
    board = remove_blue_lines(crop_board(frame, board_idx=0))
    h, w = board.shape[:2]
    tx, ty = int(w * 0.05), int(h * 0.05)
    board = board[ty:h - ty, tx:w - tx]
    return board, cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)


def pick_threshold(gray):
    """Darkness threshold from histogram percentile, clamped to the
    known-stable plateau band (script 28: T=65-110 all give ~178)."""
    return int(np.clip(np.percentile(gray, 5), 55, 110))


def dark_mask(gray, T):
    """Binary: pixel is plug-dark or it isn't."""
    return (gray < T).astype(np.uint8)


def mid_mask(gray, T):
    """Binary: plastic-edge brightness band (script 29: edges sit
    ~10-45 gray levels above the plug cutoff)."""
    return ((gray >= T + 10) & (gray < T + 45)).astype(np.uint8)


def _smooth(v, k):
    return np.convolve(v.astype(float), np.ones(k) / k, mode="same")


def lock_lattice(gray, expected_strips=9):
    """Full lattice lock: strips, row spacing, per-strip rows, tube
    edges. All integer. Returns lattice dict or None on failure."""
    T = pick_threshold(gray)
    d = dark_mask(gray, T)
    h, w = gray.shape[:2]

    # Strips: column projection peaks
    colp = _smooth(d.sum(axis=0), 7)
    min_dist = w // (expected_strips + 3)
    for frac in (0.15, 0.08, 0.04):
        peaks, _ = find_peaks(colp, distance=min_dist,
                              height=colp.max() * frac)
        if len(peaks) >= expected_strips:
            break
    if len(peaks) < expected_strips - 1:
        return None
    peaks = peaks[:expected_strips]

    # Global row spacing: autocorrelation of total row projection
    rowp = _smooth(d.sum(axis=1), 5)
    rz = rowp - rowp.mean()
    ac = np.correlate(rz, rz, mode="full")[len(rz) - 1:]
    dy = int(np.argmax(ac[15:45]) + 15)

    # Per-strip rows + tube edge (comb-completed: fill gaps the
    # peak finder missed using the known spacing dy)
    mid = mid_mask(gray, T)
    strips = []
    for i, pk in enumerate(peaks):
        x0 = 0 if i == 0 else (peaks[i - 1] + pk) // 2
        x1 = w if i == len(peaks) - 1 else (pk + peaks[i + 1]) // 2
        sp = _smooth(d[:, x0:x1].sum(axis=1), 5)
        # min distance 0.45*dy: tube racking is irregular and
        # neighboring plugs can sit closer than the median pitch
        rows, _ = find_peaks(sp, distance=max(10, int(dy * 0.45)),
                             height=sp.max() * 0.08)
        rows = [int(r) for r in rows]
        if len(rows) >= 3:
            rows = _comb_fill(rows, dy)
        ep = mid[:, x0:x1].sum(axis=0)
        edge_x = int(x0 + np.argmax(ep))

        # Per-plug x: each plug sits at its own depth inside its
        # tube (insertion varies ±40px), so search each row band
        # for the densest dark window instead of assuming the
        # strip center. x - edge_x = plug insertion depth.
        k = max(int(dy * 0.6) | 1, 7)
        hk = k // 2
        xs = []
        for ry in rows:
            band = d[max(0, ry - hk):min(h, ry + hk + 1),
                     x0:x1].sum(axis=0)
            prof = np.convolve(band, np.ones(k), mode="same")
            bx = int(np.argmax(prof))
            xs.append(int(x0 + bx) if prof[bx] >= 0.12 * k * k
                      else int(pk))

        # Vertical extent of the plastic strip (= the tube rack):
        # rows outside it (pin rail, bezel) are not plug cells
        mrow = mid[:, x0:x1].sum(axis=1)
        ys_on = np.where(mrow > mrow.max() * 0.15)[0]
        y_top = int(ys_on[0]) if len(ys_on) else 0
        y_bot = int(ys_on[-1]) if len(ys_on) else h

        strips.append({"x": int(pk), "x0": int(x0), "x1": int(x1),
                       "edge_x": edge_x, "rows": rows, "xs": xs,
                       "y_top": y_top, "y_bot": y_bot})

    return {"T": T, "dy": dy, "w": w, "h": h, "strips": strips,
            "ref_colproj": d.sum(axis=0).astype(np.int32),
            "ref_rowproj": d.sum(axis=1).astype(np.int32)}


def _comb_fill(rows, dy):
    """Real peaks stay at their true y (tube spacing is NOT
    perfectly regular — gaps of 0.9-1.5x dy occur); only fill
    genuinely large gaps with evenly spaced phantom teeth."""
    rows = sorted(rows)
    out = [rows[0]]
    for r in rows[1:]:
        gap = r - out[-1]
        n = int(round(gap / dy))
        if n >= 2:
            a = out[-1]
            step = gap / n
            for j in range(1, n):
                out.append(int(round(a + step * j)))
        out.append(r)
    return sorted(out)


def _corr_shift(ref, cur, search):
    """Best integer shift of cur vs ref by dot-product; returns
    (shift, peak_ratio) — ratio of best to best-outside-±3."""
    n = len(ref)
    scores = np.full(2 * search + 1, -1.0)
    for s in range(-search, search + 1):
        a0, a1 = max(0, s), min(n, n + s)
        b0, b1 = max(0, -s), min(n, n - s)
        seg = a1 - a0
        if seg < n // 2:
            continue
        scores[s + search] = np.dot(ref[a0:a1], cur[b0:b1]) / seg
    best = int(np.argmax(scores))
    rival = scores.copy()
    rival[max(0, best - 3):best + 4] = -1
    ratio = scores[best] / max(rival.max(), 1e-9)
    return best - search, float(ratio)


def track_shift(lat, gray, search=40, T=None):
    """Per-frame rigid shift vs locked reference projections.
    Returns (dx, dy, ratio_x, ratio_y) — all integer shifts.
    Pass T to adapt the threshold per frame (lighting changes)."""
    d = dark_mask(gray, T if T is not None else lat["T"])
    dx, rx = _corr_shift(lat["ref_colproj"],
                         d.sum(axis=0).astype(np.int32), search)
    dyy, ry = _corr_shift(lat["ref_rowproj"],
                          d.sum(axis=1).astype(np.int32), search)
    return -dx, -dyy, rx, ry


def read_cells(gray, lat, dx=0, dy=0, T=None):
    """Occupancy at every lattice cell, shifted by (dx, dy).
    Returns list of (strip, row_idx, x, y, frac) with binary
    occ = frac > 0.12; frac = dark-pixel fraction in the window.
    Pass T to adapt the threshold per frame (lighting changes)."""
    T = T if T is not None else lat["T"]
    k = max(int(lat["dy"] * 0.6) | 1, 7)
    hk = k // 2
    h, w = gray.shape[:2]
    out = []
    for si, s in enumerate(lat["strips"]):
        xs = s.get("xs")
        for ri, y0 in enumerate(s["rows"]):
            x = (xs[ri] if xs and ri < len(xs) else s["x"]) + dx
            y = y0 + dy
            if not (hk <= x < w - hk and hk <= y < h - hk):
                out.append((si, ri, x, y, -1.0))
                continue
            win = gray[y - hk:y + hk + 1, x - hk:x + hk + 1]
            frac = float(np.mean(win < T))
            out.append((si, ri, x, y, frac))
    return out


def crispness(cells):
    """Health: fraction of readable cells that are decisive
    (clearly full >0.3 or clearly empty <0.05). Script 27 metric."""
    fr = [c[4] for c in cells if c[4] >= 0]
    if not fr:
        return 0.0
    fr = np.array(fr)
    return float(np.mean((fr > 0.3) | (fr < 0.05)))


def occupancy_vector(cells):
    """Binary occupancy tuple for comparing frames."""
    return tuple(1 if c[4] > 0.12 else 0 for c in cells)
