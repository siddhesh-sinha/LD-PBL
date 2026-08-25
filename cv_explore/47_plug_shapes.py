"""
47 — Plug Shapes: radial darkness sweep → exact outline per plug
=================================================================
For every locked plug center:
  1. Center-of-darkness refine (the shrinking bubble)
  2. Sweep 48 rays outward; along each ray the darkness weight
     falls as cotton gives way to plastic — the steepest drop /
     half-max crossing is that ray's boundary radius
  3. Regularize r(theta) across angles (circular median) — rays
     can't disagree wildly with their neighbors
  4. Emit per-plug polygon outline + mask + shape properties
     (area, radius, aspect, edge sharpness = fuzzy vs crisp)
The union mask is the reference for observing what moves later.
"""

from common import make_output_dir, save, crop_board
from crux import (FRAME_PATH, remove_blue_lines, lock_lattice,
                  read_cells)

import cv2
import numpy as np
import json

N_ANG = 48
ANGLES = np.linspace(0, 2 * np.pi, N_ANG, endpoint=False)
COS, SIN = np.cos(ANGLES), np.sin(ANGLES)


def circ_median(r, k=5):
    """Circular median filter over the r(theta) profile."""
    n = len(r)
    out = np.empty(n)
    for i in range(n):
        idx = [(i + j - k // 2) % n for j in range(k)]
        out[i] = np.median(r[idx])
    return out


def plug_shape(gray, cx, cy, rmax):
    """Radial sweep around one plug. Returns dict or None."""
    h, w = gray.shape[:2]
    if not (rmax <= cx < w - rmax and rmax <= cy < h - rmax):
        return None
    patch = gray[cy - rmax:cy + rmax + 1,
                 cx - rmax:cx + rmax + 1].astype(np.int16)

    # Local background = median of the patch border ring (plastic)
    ring = np.concatenate([patch[0], patch[-1],
                           patch[:, 0], patch[:, -1]])
    bg = int(np.median(ring))
    wmap = np.clip(bg - patch, 0, None)          # darkness weight

    # Shrinking bubble: refine center to the center-of-darkness
    ys, xs = np.mgrid[0:patch.shape[0], 0:patch.shape[1]]
    tot = wmap.sum()
    if tot < 50:
        return None
    for _ in range(2):
        mx = int(round((xs * wmap).sum() / tot))
        my = int(round((ys * wmap).sum() / tot))
    core = float(wmap[max(0, my - 2):my + 3,
                      max(0, mx - 2):mx + 3].mean())
    if core < 8:
        return None

    # Radial profiles from the refined center
    r_edge = np.zeros(N_ANG)
    grads = np.zeros(N_ANG)
    for a in range(N_ANG):
        prof = []
        for r in range(rmax):
            px = mx + int(round(COS[a] * r))
            py = my + int(round(SIN[a] * r))
            if not (0 <= px < patch.shape[1]
                    and 0 <= py < patch.shape[0]):
                break
            prof.append(wmap[py, px])
        prof = np.array(prof, dtype=float)
        if len(prof) < 4:
            r_edge[a] = 2
            continue
        # Half-max crossing of the darkness profile
        half = core * 0.5
        below = np.where(prof < half)[0]
        r_edge[a] = below[0] if len(below) else len(prof) - 1
        # Edge sharpness: steepest single-step drop near the edge
        d = -np.diff(prof)
        grads[a] = d.max() if len(d) else 0

    r_s = np.clip(circ_median(r_edge), 2, rmax - 1)

    # Polygon in board coords (integer pixels)
    poly = np.array([[cx - rmax + mx + int(round(COS[a] * r_s[a])),
                      cy - rmax + my + int(round(SIN[a] * r_s[a]))]
                     for a in range(N_ANG)], dtype=np.int32)

    return {"cx": cx - rmax + mx, "cy": cy - rmax + my,
            "poly": poly, "r_mean": float(r_s.mean()),
            "aspect": float(r_s.max() / max(r_s.min(), 1)),
            "sharp": float(grads.mean()), "bg": bg,
            "core": core}


# ── Main ────────────────────────────────────────────────
out = make_output_dir("47_plug_shapes")
frame = cv2.imread(FRAME_PATH)
board_full = remove_blue_lines(crop_board(frame, board_idx=0))
h0, w0 = board_full.shape[:2]
tx, ty = int(w0 * 0.05), int(h0 * 0.05)
board = board_full[ty:h0 - ty, tx:w0 - tx]
gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
h, w = gray.shape[:2]

lat = lock_lattice(gray)
cells = read_cells(gray, lat)
occ = [(si, ri, x, y) for (si, ri, x, y, fr) in cells if fr > 0.12]
rmax = int(lat["dy"] * 0.75)
print(f"  {len(occ)} occupied cells, sweep radius {rmax}px")

shapes = []
for (si, ri, x, y) in occ:
    s = plug_shape(gray, x, y, rmax)
    if s is not None:
        s["strip"], s["row"] = si, ri
        shapes.append(s)
print(f"  {len(shapes)} plug outlines extracted")

# ── Stats ───────────────────────────────────────────────
areas = [cv2.contourArea(s["poly"]) for s in shapes]
print(f"\n  Shape properties across {len(shapes)} plugs:")
print(f"  area   px²: med={np.median(areas):.0f} "
      f"q1-q3={np.percentile(areas,25):.0f}"
      f"-{np.percentile(areas,75):.0f}")
print(f"  radius px : med={np.median([s['r_mean'] for s in shapes]):.1f}")
print(f"  aspect    : med={np.median([s['aspect'] for s in shapes]):.2f}"
      f"  (1.0 = round)")
sh = [s["sharp"] for s in shapes]
print(f"  edge sharp: med={np.median(sh):.1f}  "
      f"fuzziest={min(sh):.1f}  crispest={max(sh):.1f}")

# ── Overlays ────────────────────────────────────────────
vis = board.copy()
for s in shapes:
    cv2.polylines(vis, [s["poly"]], True, (0, 255, 0), 1)
    cv2.circle(vis, (s["cx"], s["cy"]), 1, (0, 0, 255), -1)
save(out, "01_outlines_full.jpg", vis)

# Union mask — the reference mask for later motion observation
mask = np.zeros((h, w), dtype=np.uint8)
for s in shapes:
    cv2.fillPoly(mask, [s["poly"]], 255)
save(out, "02_plug_union_mask.png", mask)

# Contact sheet: every plug patch at 3x with its outline
cols_n, cell_px = 14, (2 * rmax + 1) * 3
rows_n = int(np.ceil(len(shapes) / cols_n))
sheet = np.zeros((rows_n * cell_px, cols_n * cell_px, 3),
                 dtype=np.uint8)
for i, s in enumerate(shapes):
    cx, cy = s["cx"], s["cy"]
    x0, y0 = cx - rmax, cy - rmax
    patch = board[max(0, y0):y0 + 2 * rmax + 1,
                  max(0, x0):x0 + 2 * rmax + 1].copy()
    if patch.shape[0] != 2 * rmax + 1 or patch.shape[1] != 2 * rmax + 1:
        continue
    p3 = cv2.resize(patch, (cell_px, cell_px),
                    interpolation=cv2.INTER_NEAREST)
    poly3 = (s["poly"] - [x0, y0]) * 3
    cv2.polylines(p3, [poly3], True, (0, 255, 0), 1)
    gr, gc = divmod(i, cols_n)
    sheet[gr * cell_px:(gr + 1) * cell_px,
          gc * cell_px:(gc + 1) * cell_px] = p3
save(out, "03_contact_sheet.jpg", sheet)

with open(f"{out}/shapes.json", "w") as f:
    json.dump([{k: (v.tolist() if k == "poly" else v)
                for k, v in s.items()} for s in shapes], f)

print(f"\nDone! Check {out}/")
