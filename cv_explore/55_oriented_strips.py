"""
55 — Orientation-Free Strip Finder (rulers thrown on a table)
==============================================================
No assumed orientation: each strip is found as a coherent region
of anisotropic corrugation texture via the STRUCTURE TENSOR:
  energy    = how much gradient lives here (normalized by light)
  coherence = how aligned those gradients are (corrugation ~ 1)
  angle     = the strip's own axis, measured not assumed
Each strip gets a rotated rectangle + its tube lines sampled
along its measured axis. Acid test: the whole board rotated 25
degrees must parse identically.
"""

from common import make_output_dir, save
from autocal import find_boards

import cv2
import numpy as np
from scipy.signal import find_peaks

UP = "/root/.claude/uploads/608e2092-c86e-5429-b039-27e3dec388ed/"


def ridge_energy(gray):
    """Orientation-FREE corrugation detector: difference of
    Gaussians tuned to the ridge scale responds to ~ridge-thick
    structure in ANY direction; plugs respond too, so a strip
    stays one contiguous region. Normalized by local brightness."""
    g32 = gray.astype(np.float32)
    dog = cv2.GaussianBlur(g32, (0, 0), 2) \
        - cv2.GaussianBlur(g32, (0, 0), 6)
    en = cv2.boxFilter(np.abs(dog), -1, (15, 15))
    bright = np.clip(cv2.blur(g32, (31, 31)), 20, None)
    return en / bright


def _rect_axis(rect):
    """Angle of the LONG side of a minAreaRect, degrees."""
    box = cv2.boxPoints(rect)
    e1 = box[1] - box[0]
    e2 = box[2] - box[1]
    v = e1 if np.linalg.norm(e1) > np.linalg.norm(e2) else e2
    return float(np.degrees(np.arctan2(v[1], v[0]))) % 180


def find_oriented_strips(gray):
    """Strip regions at ANY angle: high ridge energy -> CC ->
    rotated rects; the axis is each region's own geometry."""
    en = ridge_energy(gray)
    lit = gray > 60                       # stats on the board only
    med = np.median(en[lit]) if lit.any() else np.median(en)
    mask = ((en > med * 1.5) & lit).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            np.ones((5, 5), np.uint8))

    def extract(m, depth=0):
        """CC -> strips; a merged blob (fails aspect, still big)
        is eroded and re-extracted — orientation-free splitting."""
        got = []
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
        for i in range(1, n):
            if st[i, 4] < 2500:
                continue
            sub = (lab == i).astype(np.uint8)
            ys, xs = np.where(sub)
            rect = cv2.minAreaRect(
                np.column_stack([xs, ys]).astype(np.float32))
            (cx, cy), (rw, rh), _ = rect
            L, W = max(rw, rh), min(rw, rh)
            if L >= 3.5 * W and L >= 100:
                got.append({"rect": rect, "cx": cx, "cy": cy,
                            "len": L, "wid": W,
                            "axis": _rect_axis(rect)})
            elif depth < 6:
                er = cv2.erode(sub, np.ones((3, 3), np.uint8))
                got.extend(extract(er, depth + 1))
        return got

    strips = extract(mask)
    strips.sort(key=lambda s: s["cx"])
    return strips, mask, en


def tube_lines_along_axis(gray, s):
    """Rotate the strip's ROI so its measured axis is vertical,
    then find the corrugation line peaks."""
    cx, cy, L, W = s["cx"], s["cy"], s["len"], s["wid"]
    rot = cv2.getRotationMatrix2D((cx, cy),
                                  s["axis"] - 90 if s["axis"] < 135
                                  else s["axis"] - 270, 1.0)
    h, w = gray.shape
    gr = cv2.warpAffine(gray, rot, (w, h), flags=cv2.INTER_NEAREST,
                        borderValue=int(np.median(gray)))
    x0 = int(max(0, cx - W / 2 - 2))
    x1 = int(min(w, cx + W / 2 + 2))
    y0 = int(max(0, cy - L / 2))
    y1 = int(min(h, cy + L / 2))
    sy = np.abs(cv2.Sobel(gr[y0:y1, x0:x1], cv2.CV_32F, 0, 1, 3))
    prof = np.convolve(sy.mean(axis=1), np.ones(3) / 3, "same")
    pk, _ = find_peaks(prof, distance=14,
                       height=np.median(prof) * 1.1)
    return len(pk)


def run(gray, tag, frame_bgr, pitch=20):
    strips, mask, en = find_oriented_strips(gray)
    # Validate by periodicity: a real strip's tube count matches
    # its length / pitch; serration-band artifacts don't
    valid = []
    for s in strips:
        nl = tube_lines_along_axis(gray, s)
        exp = s["len"] / pitch
        if 0.6 * exp <= nl <= 1.6 * exp:
            s["tubes"] = nl
            valid.append(s)
    # Dedupe erosion fragments: keep longest, drop overlapping
    valid.sort(key=lambda s: -s["len"])
    kept = []
    for s in valid:
        if all(np.hypot(s["cx"] - k["cx"], s["cy"] - k["cy"])
               > 0.5 * min(s["wid"], k["wid"]) + 8 for k in kept):
            kept.append(s)
    kept.sort(key=lambda s: s["cx"])

    vis = frame_bgr.copy()
    angles = []
    for si, s in enumerate(kept):
        box = cv2.boxPoints(s["rect"]).astype(int)
        cv2.drawContours(vis, [box], 0, (255, 0, 0), 2)
        angles.append(s["axis"])
        cv2.putText(vis, f"{s['axis']:.1f} {s['tubes']}t",
                    (int(s['cx']) - 28, int(s['cy'])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        print(f"    strip {si}: axis={s['axis']:+6.1f}deg  "
              f"len={s['len']:.0f} wid={s['wid']:.0f}  "
              f"tubes={s['tubes']}")
    save(out, f"{tag}_strips.jpg", vis)
    save(out, f"{tag}_mask.jpg", mask * 255)
    return angles


# ── Main ────────────────────────────────────────────────
out = make_output_dir("55_oriented_strips")
frame = cv2.imread(UP + "6ef21155-image.png")
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
bx, by, bw_, bh_ = find_boards(gray)[1]
g = gray[by:by + bh_, bx:bx + bw_]
fb = frame[by:by + bh_, bx:bx + bw_]

print("  Board 1, native orientation:")
a0 = run(g, "native", fb)

# ── Acid test: rotate the whole board 25 degrees ────────
H, W = g.shape
M = cv2.getRotationMatrix2D((W / 2, H / 2), 25, 1.0)
big = int(np.ceil(max(H, W) * 1.5))
M[0, 2] += (big - W) / 2
M[1, 2] += (big - H) / 2
gr = cv2.warpAffine(g, M, (big, big),
                    borderValue=int(np.median(g[:20])))
fr = cv2.warpAffine(fb, M, (big, big))
print("\n  Board 1 rotated 25deg (rulers thrown on a table):")
a1 = run(gr, "rot25", fr)

if a0 and a1:
    print(f"\n  Native median axis:  {np.median(a0):+.1f} deg")
    print(f"  Rotated median axis: {np.median(a1):+.1f} deg "
          f"(expected ~{np.median(a0) - 25:+.1f})")
print(f"\nDone! Check {out}/")
