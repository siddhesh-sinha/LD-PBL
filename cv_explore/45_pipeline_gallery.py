"""
45 — Pipeline Gallery: every intermediate image, stage by stage
================================================================
Dump each transformation the suite performs so the whole data
flow is visible: raw frame → board crop → blue removal → trim →
gray → dark mask → mid mask → projections → lattice → cell read,
plus the system-specific views (FFT spectrum, template, NCC map).
"""

from common import make_output_dir, save, crop_board
from crux import (FRAME_PATH, remove_blue_lines, pick_threshold,
                  dark_mask, mid_mask, lock_lattice, read_cells)

import cv2
import numpy as np

out = make_output_dir("45_pipeline_gallery")

# ── Crux stages (shared by every system) ────────────────
frame = cv2.imread(FRAME_PATH)
save(out, "s00_raw_frame.jpg", frame)

board0 = crop_board(frame, board_idx=0)
save(out, "s01_board_cropped.jpg", board0)

noblue = remove_blue_lines(board0)
save(out, "s02_blue_removed.jpg", noblue)

h0, w0 = noblue.shape[:2]
tx, ty = int(w0 * 0.05), int(h0 * 0.05)
board = noblue[ty:h0 - ty, tx:w0 - tx]
save(out, "s03_border_trimmed.jpg", board)

gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
save(out, "s04_grayscale.jpg", gray)

T = pick_threshold(gray)
d = dark_mask(gray, T)
dm = d * 255
cv2.putText(dm, f"gray < T={T}", (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1)
save(out, "s05_dark_mask.jpg", dm)

mm = mid_mask(gray, T) * 255
cv2.putText(mm, f"plastic band {T+10}-{T+45}", (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1)
save(out, "s06_mid_mask_plastic.jpg", mm)

# Column projection + strip peaks
lat = lock_lattice(gray)
h, w = gray.shape[:2]
colp = d.sum(axis=0).astype(float)
proj = np.zeros((160, w, 3), dtype=np.uint8)
cn = colp / (colp.max() + 1e-9) * 150
for x in range(w):
    cv2.line(proj, (x, 159), (x, 159 - int(cn[x])), (0, 200, 0), 1)
for s in lat["strips"]:
    cv2.line(proj, (s["x"], 0), (s["x"], 159), (0, 0, 255), 1)
    cv2.line(proj, (s["x0"], 0), (s["x0"], 159), (255, 200, 0), 1)
save(out, "s07_column_projection.jpg", proj)

# Per-strip row projections (rotated 90° so rows read downward)
rp = np.zeros((h, 9 * 62, 3), dtype=np.uint8)
for i, s in enumerate(lat["strips"]):
    sp = d[:, s["x0"]:s["x1"]].sum(axis=1).astype(float)
    sn = sp / (sp.max() + 1e-9) * 55
    x_base = i * 62
    for y in range(h):
        cv2.line(rp, (x_base, y), (x_base + int(sn[y]), y),
                 (0, 200, 0), 1)
    for ry in s["rows"]:
        cv2.line(rp, (x_base, ry), (x_base + 58, ry), (0, 0, 255), 1)
save(out, "s08_row_projections.jpg", rp)

# Lattice + cell read
cells = read_cells(gray, lat)
k = max(int(lat["dy"] * 0.6) | 1, 7)
vis = board.copy()
for s in lat["strips"]:
    cv2.line(vis, (s["edge_x"], 0), (s["edge_x"], h), (255, 150, 0), 1)
for (si, ri, x, y, fr) in cells:
    if fr < 0:
        continue
    occ = fr > 0.12
    col = (0, 220, 0) if occ else (0, 0, 255)
    cv2.rectangle(vis, (x - k // 2, y - k // 2),
                  (x + k // 2, y + k // 2), col, 1)
save(out, "s09_lattice_cells.jpg", vis)

n_occ = sum(1 for c in cells if c[4] > 0.12)
print(f"  Lattice: {len(cells)} cells, {n_occ} occupied, T={T}, "
      f"dy={lat['dy']}")

# ── System D view: FFT spectrum ─────────────────────────
hann = np.outer(np.hanning(h), np.hanning(w))
F = np.abs(np.fft.fftshift(np.fft.fft2(d * hann)))
Flog = np.log1p(F)
Fimg = (Flog / Flog.max() * 255).astype(np.uint8)
Fimg = cv2.applyColorMap(Fimg, cv2.COLORMAP_INFERNO)
r0 = h / lat["dy"]
cy0, cx0 = h // 2, w // 2
cv2.circle(Fimg, (cx0, cy0), int(r0), (0, 255, 0), 1)
cv2.putText(Fimg, f"row-freq ring r={r0:.1f} (h/dy)", (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
save(out, "s10_fft_spectrum_sysD.jpg", Fimg)

# ── System C view: template + NCC score map ─────────────
kt = int(lat["dy"] * 0.55) | 1
patches = [gray[y - kt // 2:y + kt // 2 + 1,
                x - kt // 2:x + kt // 2 + 1]
           for (si, ri, x, y, fr) in cells
           if fr > 0.12 and kt // 2 <= y < h - kt // 2
           and kt // 2 <= x < w - kt // 2]
tmpl = np.median(np.stack(patches), axis=0).astype(np.uint8)
save(out, "s11_plug_template_sysC.jpg",
     cv2.resize(tmpl, (kt * 10, kt * 10),
                interpolation=cv2.INTER_NEAREST))

score = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
sc = np.clip((score + 1) / 2 * 255, 0, 255).astype(np.uint8)
sc = cv2.applyColorMap(sc, cv2.COLORMAP_HOT)
save(out, "s12_ncc_scoremap_sysC.jpg", sc)

# ── System E view: frame diff during an event ───────────
from stress import stress_sequence
prev = None
for f, truth in stress_sequence(board):
    if truth["i"] == 39:
        prev = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    if truth["i"] == 40:  # the jerk frame
        cur = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        break
diff = cv2.absdiff(cur, prev)
diff = np.clip(diff.astype(int) * 3, 0, 255).astype(np.uint8)
cv2.putText(diff, "frame 39 vs 40 (the jerk) x3", (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1)
save(out, "s13_frame_diff_sysE.jpg", diff)

print(f"\nDone! 14 stage images in {out}/")
