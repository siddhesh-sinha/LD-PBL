"""
31 — Period + Phase Fit (autocorrelation comb)
===============================================
Treat the board as a periodic signal and generate the COMPLETE lattice:
  1. Strip x-centers: dark column projection -> autocorrelation period
     (~bw/9) -> comb phase scan -> 9 x-centers.
  2. Row period dy: summed row projection -> autocorrelation -> first
     strong peak (lag>=15) with parabolic sub-sample refinement.
  3. Per-strip row phase phi_s (independent tubes) -> candidate rows
     inside each strip's dark-pixel vertical support.
  4. Probe each lattice cell (9x9 mean gray) -> confidence -> keep
     conf>0.15 -> refine by darkness-weighted center of mass.
  5. Tube-end plastic edge per strip: mid-band (80-115) column profile,
     strongest vertical line.
"""

from common import make_output_dir, save, crop_board

import cv2
import numpy as np
import json
from scipy.signal import find_peaks


def remove_blue_lines(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, (100, 80, 80), (130, 255, 255))
    blue_mask = cv2.dilate(blue_mask, np.ones((3, 3), np.uint8), iterations=1)
    return cv2.inpaint(img, blue_mask, 5, cv2.INPAINT_TELEA)


def parabolic(v, i):
    """Sub-sample peak position around index i of array v."""
    if i <= 0 or i >= len(v) - 1:
        return float(i)
    d = v[i - 1] - 2 * v[i] + v[i + 1]
    return float(i) if d == 0 else i + 0.5 * (v[i - 1] - v[i + 1]) / d


def comb_phase(proj, period, n_teeth, step=0.25):
    """Best phase in [0, period): maximize sum of proj at comb teeth."""
    yy = np.arange(len(proj), dtype=float)
    best_phi, best_score = 0.0, -1.0
    for phi in np.arange(0, period, step):
        pos = phi + np.arange(n_teeth) * period
        pos = pos[pos <= len(proj) - 1]
        score = np.interp(pos, yy, proj).sum()
        if score > best_score:
            best_phi, best_score = phi, score
    return best_phi, best_score


# ── Board prep ──────────────────────────────────────────
frame = cv2.imread("/root/.claude/uploads/608e2092-c86e-5429-b039-27e3dec388ed/f95b95fa-image.jpg")
board_raw = crop_board(frame, board_idx=0)
board_noblue = remove_blue_lines(board_raw)
bh, bw = board_noblue.shape[:2]
tx, ty = int(bw * 0.05), int(bh * 0.05)
board = board_noblue[ty:bh - ty, tx:bw - tx]
gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
bh, bw = gray.shape
out = make_output_dir("31_period_phase")
print(f"  Board: {bw}x{bh}")

# ── STEP 1: strip period + phase from column projection ─
dark = (gray < 70)
col_proj = dark.sum(axis=0).astype(float)
cp = col_proj - col_proj.mean()
ac_x = np.correlate(cp, cp, mode="full")[len(cp) - 1:]
lo, hi = int(bw / 9 * 0.75), int(bw / 9 * 1.25)
lag = lo + int(np.argmax(ac_x[lo:hi]))
period = parabolic(ac_x, lag)
ambiguous = ac_x[lag] < 0.15 * ac_x[0]
if ambiguous:  # fall back: direct peak picking
    sm = np.convolve(col_proj, np.ones(9) / 9, mode="same")
    pks, _ = find_peaks(sm, distance=int(bw / 12))
    pks = np.sort(pks[np.argsort(sm[pks])[-9:]])
    period = float(np.median(np.diff(pks)))
    print("  WARNING: x-autocorr ambiguous -> find_peaks fallback")
phi_x, _ = comb_phase(col_proj, period, 9)
x_centers = phi_x + np.arange(9) * period
print(f"  Strip period: {period:.2f} px  phase: {phi_x:.2f}"
      f"  (autocorr peak ratio {ac_x[lag]/ac_x[0]:.2f})")

strips = []
for s, xc in enumerate(x_centers):
    xl = max(0.0, xc - period / 2)
    xr = min(float(bw), xc + period / 2)
    strips.append({"idx": s, "x_left": xl, "x_right": xr, "x_center": float(xc)})

# ── STEP 2: row period dy from summed row projections ───
row_proj = dark.sum(axis=1).astype(float)
rp = row_proj - row_proj.mean()
ac_y = np.correlate(rp, rp, mode="full")[len(rp) - 1:]
pks_y, _ = find_peaks(ac_y[15:60], height=0.1 * ac_y[0])
lag_y = 15 + (int(pks_y[0]) if len(pks_y) else int(np.argmax(ac_y[15:60])))
dy = parabolic(ac_y, lag_y)
print(f"  Row spacing dy: {dy:.3f} px (autocorr lag {lag_y})")

# ── STEP 3+4: per-strip phase -> lattice -> probe -> refine ──
n_gen = n_keep = 0
plugs = []
for st in strips:
    xl, xr = int(round(st["x_left"])), int(round(st["x_right"]))
    p_s = dark[:, xl:xr].sum(axis=1).astype(float)
    phi_s, _ = comb_phase(p_s, dy, int(bh / dy) + 1)
    sup = np.where(p_s > 3)[0]
    y0, y1 = (sup[0], sup[-1]) if len(sup) else (0, bh - 1)
    st["phase"], st["gen"], st["keep"] = phi_s, 0, 0
    for k in range(int(bh / dy) + 1):
        y = phi_s + k * dy
        if y < y0 - 3 or y > y1 + 3:
            continue
        st["gen"] += 1
        n_gen += 1
        xi, yi = int(round(st["x_center"])), int(round(y))
        win = gray[max(0, yi - 4):yi + 5, max(0, xi - 4):xi + 5]
        conf = float(np.clip((125 - win.mean()) / 60, 0, 1))
        if conf <= 0.15:
            continue
        # refine: darkness-weighted center of mass
        hy, hx = int(dy / 2 - 2), 10
        ya, xa = max(0, yi - hy), max(0, xi - hx)
        sub = gray[ya:yi + hy + 1, xa:xi + hx + 1].astype(float)
        w8 = np.clip(125.0 - sub, 0, None)
        if w8.sum() > 0:
            ys, xs = np.mgrid[ya:ya + sub.shape[0], xa:xa + sub.shape[1]]
            cx, cy = (w8 * xs).sum() / w8.sum(), (w8 * ys).sum() / w8.sum()
        else:
            cx, cy = float(xi), float(y)
        plugs.append({"strip": st["idx"], "row": st["keep"],
                      "cx": float(cx), "cy": float(cy), "conf": conf})
        st["keep"] += 1
        n_keep += 1

# ── STEP 5: tube edge per strip (mid-band vertical line) ─
mid = ((gray >= 80) & (gray <= 115)).astype(float)
for st in strips:
    xl, xr = int(round(st["x_left"])), int(round(st["x_right"]))
    prof = np.convolve(mid[:, xl:xr].sum(axis=0), np.ones(3) / 3, mode="same")
    st["edge_x"] = xl + parabolic(prof, int(np.argmax(prof)))

# ── Residual RMS: y = y0_s + row*dy_fit (shared dy) on REFINED pts ──
rows = np.array([p["row"] for p in plugs], float)
cys = np.array([p["cy"] for p in plugs], float)
sid = np.array([p["strip"] for p in plugs])
num = den = 0.0
for s in range(9):
    m = sid == s
    if m.sum() < 2:
        continue
    r, c = rows[m] - rows[m].mean(), cys[m] - cys[m].mean()
    num, den = num + (r * c).sum(), den + (r * r).sum()
dy_fit = num / den
res = np.concatenate([cys[sid == s] - cys[sid == s].mean()
                      - dy_fit * (rows[sid == s] - rows[sid == s].mean())
                      for s in range(9) if (sid == s).any()])
residual_rms = float(np.sqrt((res ** 2).mean()))

# ── Results table + overlay + JSON ──────────────────────
print(f"\n  {'Strip':>5} | {'x_ctr':>6} | {'edge_x':>6} | {'phase':>5} | "
      f"{'gen':>3} | {'keep':>4}")
print(f"  {'-' * 46}")
for st in strips:
    print(f"  S{st['idx']:<4} | {st['x_center']:>6.1f} | {st['edge_x']:>6.1f} | "
          f"{st['phase']:>5.1f} | {st['gen']:>3} | {st['keep']:>4}")
print(f"\n  Generated cells:   {n_gen}")
print(f"  Confirmed plugs:   {n_keep}")
print(f"  Rejected as empty: {n_gen - n_keep}")
print(f"  Row spacing dy:    {dy:.3f} px (LSQ refit {dy_fit:.3f})")
print(f"  Residual RMS:      {residual_rms:.3f} px")

vis = board.copy()
for st in strips:
    ex = int(round(st["edge_x"]))
    cv2.line(vis, (ex, 0), (ex, bh), (0, 255, 255), 1)
    cv2.putText(vis, f"S{st['idx']}", (int(st["x_center"]) - 10, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
for p in plugs:
    cv2.circle(vis, (int(round(p["cx"])), int(round(p["cy"]))), 3,
               (0, 0, 255), -1)
save(out, "05_final_overlay.jpg", vis)

result = {"approach": "31_period_phase", "board": {"w": int(bw), "h": int(bh)},
          "row_spacing": float(dy),
          "strips": [{"idx": s["idx"], "x_left": float(s["x_left"]),
                      "x_right": float(s["x_right"]),
                      "x_center": float(s["x_center"]),
                      "edge_x": float(s["edge_x"])} for s in strips],
          "plugs": plugs}
with open(f"{out}/positions.json", "w") as f:
    json.dump(result, f, indent=1)
print(f"  → positions.json\n\nDone! Check {out}/")
