"""
54 — Corrugation Texture Parser
================================
The user's insight from the heat maps: the NOT-strip region is
smooth (uniform yellow), the STRIP carries horizontal LINES (the
tube walls — green with line texture). So parse by TEXTURE, not
brightness:
  strip     = columns with strong periodic vertical gradients
  tube rows = the row positions of those horizontal lines
Illumination-independent: works on dim strips and EMPTY strips
(no plugs needed — the corrugation alone gives the tube grid).
Demo on the 3 hand-cut slices + one full board.
"""

from common import make_output_dir, save
from autocal import find_boards

import cv2
import numpy as np
from scipy.signal import find_peaks

UP = "/root/.claude/uploads/608e2092-c86e-5429-b039-27e3dec388ed/"
SLICES = [("strip9", "bb739927-image.png"),
          ("strip6", "79c0ae6e-image.png"),
          ("strip10_empty", "7e6df2a1-image.png")]

out = make_output_dir("54_corrugation_parse")


def texture_energy(gray):
    """Per-column corrugation score: mean |vertical gradient|
    NORMALIZED by local brightness — contrast ratio survives the
    vignette that raw gradient energy does not (an empty dim
    strip keeps its corrugation signature)."""
    sy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    bright = np.clip(gray.astype(np.float32).mean(axis=0), 20, None)
    return sy, sy.mean(axis=0) / bright


def tube_lines(sy, x0, x1, min_pitch=14):
    """Row positions of the horizontal tube-wall lines within a
    column span: row projection of vertical-gradient energy."""
    prof = sy[:, x0:x1].mean(axis=1)
    prof = np.convolve(prof, np.ones(3) / 3, mode="same")
    pk, _ = find_peaks(prof, distance=min_pitch,
                       height=np.median(prof) * 1.1)
    return [int(p) for p in pk], prof


# ── Part 1: the 3 hand-cut slices ───────────────────────
for name, fn in SLICES:
    img = cv2.imread(UP + fn)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    sy, col_energy = texture_energy(gray)

    # Strip columns = high texture energy (relative to slice max)
    thr = col_energy.max() * 0.55
    strip_cols = np.where(col_energy > thr)[0]
    sx0, sx1 = (int(strip_cols[0]), int(strip_cols[-1])) \
        if len(strip_cols) else (0, w)

    lines, prof = tube_lines(sy, sx0, sx1)
    print(f"  {name}: strip cols [{sx0},{sx1}] of {w}, "
          f"{len(lines)} tube lines, "
          f"pitch {np.median(np.diff(lines)):.1f}px"
          if len(lines) > 2 else f"  {name}: {len(lines)} lines")

    # Panel: original | texture energy map | parsed overlay
    en_vis = cv2.applyColorMap(
        np.clip(sy * 8, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    ov = img.copy()
    cv2.rectangle(ov, (sx0, 0), (sx1, h - 1), (255, 0, 0), 1)
    for y in lines:
        cv2.line(ov, (sx0, y), (sx1, y), (0, 220, 220), 1)
    sep = np.full((h, 4, 3), (0, 0, 255), dtype=np.uint8)
    panel = np.hstack([img, sep, en_vis, sep, ov])
    panel = cv2.resize(panel, (panel.shape[1] * 2, panel.shape[0] * 2),
                       interpolation=cv2.INTER_NEAREST)
    for i, lb in enumerate(["orig", "texture", "parsed"]):
        cv2.putText(panel, lb, (i * (w + 4) * 2 + 6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    save(out, f"{name}_texture.jpg", panel)

# ── Part 2: one full board, texture-only strip finding ──
frame = cv2.imread(UP + "6ef21155-image.png")
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
bx, by, bw_, bh_ = find_boards(gray)[1]          # board 1 (has empty strip 10)
g = gray[by:by + bh_, bx:bx + bw_]
sy, col_energy = texture_energy(g)
ce = np.convolve(col_energy, np.ones(7) / 7, mode="same")

# Strips = runs of high normalized texture energy; close small
# holes (normalization noise fragments runs), then keep runs >12
on = (ce > np.median(ce) * 1.8).astype(np.uint8)
on = cv2.morphologyEx(on[None, :], cv2.MORPH_CLOSE,
                      np.ones((1, 9), np.uint8))[0].astype(bool)
runs, s0 = [], None
for i, v in enumerate(on):
    if v and s0 is None:
        s0 = i
    elif not v and s0 is not None:
        if i - s0 > 12:
            runs.append((s0, i))
        s0 = None
if s0 is not None and len(on) - s0 > 12:
    runs.append((s0, len(on)))

print(f"\n  Board 1 by TEXTURE alone: {len(runs)} strips")
vis = frame[by:by + bh_, bx:bx + bw_].copy()
for si, (a, b) in enumerate(runs):
    cv2.rectangle(vis, (a, 3), (b, bh_ - 3), (255, 0, 0), 1)
    lines, _ = tube_lines(sy, a, b)
    for y in lines:
        cv2.line(vis, (a, y), (b, y), (0, 220, 220), 1)
    cv2.putText(vis, f"{len(lines)}", (a + 2, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    print(f"    strip {si}: x=[{a},{b}] {len(lines)} tube lines")
save(out, "board1_texture_strips.jpg", vis)

# Energy profile visual
pv = np.zeros((120, bw_, 3), dtype=np.uint8)
cn = ce / (ce.max() + 1e-9) * 110
for x in range(bw_):
    cv2.line(pv, (x, 119), (x, 119 - int(cn[x])), (0, 200, 0), 1)
save(out, "board1_energy_profile.jpg", pv)

print(f"\nDone! Check {out}/")
