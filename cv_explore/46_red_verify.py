"""
46 — Red-Number Ground Truth Verification
==========================================
The red numbers were added BY A HUMAN, one per plug. Use them
STRICTLY as verification — never for detection: extract the red
text positions, match each to our nearest lattice cell, and audit:
  MATCHED   human number ↔ cell we read occupied
  MISSED    human number sits on a cell we read EMPTY
  NO-CELL   human number with no lattice cell near it at all
  EXTRA     cell we read occupied with no human number near it
"""

from common import make_output_dir, save, crop_board
from crux import (FRAME_PATH, remove_blue_lines, pick_threshold,
                  lock_lattice, read_cells)

import cv2
import numpy as np
import json

out = make_output_dir("46_red_verify")

frame = cv2.imread(FRAME_PATH)
board0 = crop_board(frame, board_idx=0)
board = remove_blue_lines(board0)          # FULL board — no trim
gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
h, w = gray.shape[:2]
tx, ty = int(w * 0.05), int(h * 0.05)      # trim used for LOCK only

# ── Extract red number labels ───────────────────────────
hsv = cv2.cvtColor(board, cv2.COLOR_BGR2HSV)
red = (cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
       | cv2.inRange(hsv, (160, 80, 80), (180, 255, 255)))
save(out, "01_red_mask.jpg", red)

# Merge digit strokes of one number into one blob
merged = cv2.dilate(red, np.ones((3, 7), np.uint8), iterations=1)
n_cc, labels, stats, cents = cv2.connectedComponentsWithStats(
    merged, connectivity=8)

groups = []
for i in range(1, n_cc):
    if stats[i, cv2.CC_STAT_AREA] < 10:
        continue
    groups.append([float(cents[i][0]), float(cents[i][1]),
                   int(stats[i, cv2.CC_STAT_AREA])])

# Second merge pass: digit groups of the same number
groups.sort(key=lambda g: (g[1], g[0]))
nums = []
for g in groups:
    for m in nums:
        if abs(g[0] - m[0]) < 16 and abs(g[1] - m[1]) < 9:
            tot = m[2] + g[2]
            m[0] = (m[0] * m[2] + g[0] * g[2]) / tot
            m[1] = (m[1] * m[2] + g[1] * g[2]) / tot
            m[2] = tot
            break
    else:
        nums.append(list(g))
# Keep only label-sized text: drop big tape blocks and bezel
# bleed hugging the board edges
nums = [(int(round(x)), int(round(y))) for x, y, a in nums
        if a <= 700                       # 3-digit labels reach ~560
        and 12 <= x <= w - 12 and 12 <= y <= h - 12]
print(f"  Human red-number labels found: {len(nums)}")

# ── Our detections ──────────────────────────────────────
# Lock on the trimmed CORE (border junk breaks the lock: 0% trim
# gives 45 phantom rows), then map to full coords and comb-EXTEND
# each strip into the margins only where a plug-like cell responds.
lat = lock_lattice(gray[ty:h - ty, tx:w - tx])
lat["w"], lat["h"] = w, h
for s in lat["strips"]:
    s["x"] += tx
    s["x0"] += tx
    s["x1"] += tx
    s["edge_x"] += tx
    s["rows"] = [r + ty for r in s["rows"]]
    s["xs"] = [x + tx for x in s["xs"]]
    s["y_top"] += ty
    s["y_bot"] += ty

k = max(int(lat["dy"] * 0.6) | 1, 7)
hk, T, dy = k // 2, lat["T"], lat["dy"]


def pluglike(x, y):
    """Plug-like response: some dark, but not border-junk solid."""
    if not (hk <= x < w - hk and hk <= y < h - hk):
        return False
    win = gray[y - hk:y + hk + 1, x - hk:x + hk + 1]
    return 0.12 < float(np.mean(win < T)) < 0.85


n_ext = 0
for s in lat["strips"]:
    for up in (True, False):               # extend up, then down
        for _ in range(2):                 # at most 2 teeth each way
            edge = min(s["rows"]) if up else max(s["rows"])
            # 2D mini-search: plugs off-comb vertically too, so
            # scan y in [0.7dy, 1.6dy] beyond the edge row (but
            # stay inside the plastic strip — pin rail excluded)
            lo = edge - int(1.6 * dy) if up else edge + int(0.7 * dy)
            hi = edge - int(0.7 * dy) if up else edge + int(1.6 * dy)
            lo = max(lo, s["y_top"] - 5, hk)
            hi = min(hi, s["y_bot"] + 5, h - hk - 1)
            best = None
            for cy in range(lo, hi + 1, 2):
                band = (gray[cy - hk:cy + hk + 1,
                             s["x0"]:s["x1"]] < T).sum(axis=0)
                prof = np.convolve(band, np.ones(k), mode="same")
                bx = int(np.argmax(prof))
                if best is None or prof[bx] > best[0]:
                    best = (prof[bx], s["x0"] + bx, cy)
            if best is None or best[0] < 0.12 * k * k:
                break
            if up:
                s["rows"].insert(0, best[2])
                s["xs"].insert(0, best[1])
            else:
                s["rows"].append(best[2])
                s["xs"].append(best[1])
            n_ext += 1
print(f"  Comb-extended {n_ext} edge cells into the margins")

cells = read_cells(gray, lat)
occ_cells = [(si, ri, x, y) for (si, ri, x, y, fr) in cells
             if fr > 0.12]
emp_cells = [(si, ri, x, y) for (si, ri, x, y, fr) in cells
             if 0 <= fr <= 0.12]
print(f"  Lattice: {len(cells)} cells, {len(occ_cells)} read "
      f"occupied, {len(emp_cells)} read empty")


def eligible(px, py, c):
    """Labeling convention: the number sits to the RIGHT of its
    plug (typically +25..45px), level or slightly below it."""
    dx, dyd = px - c[2], abs(py - c[3])
    # NB: some strips' labels hug the plug (dx~5), others sit
    # ~30px right; red pen marks ON plugs also exist. Global 1:1
    # assignment resolves the ambiguity — surplus blobs go
    # unmatched without stealing a real label's cell.
    if -10 <= dx <= 58 and dyd <= 14:
        return dyd * 2 + abs(dx - 28)
    return None


# GLOBAL one-to-one assignment (greedy-by-score): a rogue red
# blob near a plug must not steal the real label's cell and
# cascade mismatches down the strip.
pairs = []
for li, (nx, ny) in enumerate(nums):
    for c in occ_cells:
        sc = eligible(nx, ny, c)
        if sc is not None:
            pairs.append((sc, li, (c[0], c[1]), c))
pairs.sort(key=lambda p: p[0])
lab_done, used = set(), set()
matched, missed, nocell = [], [], []
for sc, li, ck, c in pairs:
    if li in lab_done or ck in used:
        continue
    lab_done.add(li)
    used.add(ck)
    matched.append((nums[li][0], nums[li][1], c))
for li, (nx, ny) in enumerate(nums):
    if li in lab_done:
        continue
    nx, ny = nums[li]
    ec = min(emp_cells, default=None,
             key=lambda c: (eligible(nx, ny, c) is None,
                            eligible(nx, ny, c) or 1e9))
    if ec is not None and eligible(nx, ny, ec) is not None:
        missed.append((nx, ny, ec))
    else:
        nocell.append((nx, ny))

extra = [c for c in occ_cells if (c[0], c[1]) not in used]

# ── Report ──────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  RED-NUMBER AUDIT (human labels = ground truth)")
print(f"{'='*60}")
print(f"  Human labels:        {len(nums)}")
print(f"  MATCHED (correct):   {len(matched)}  "
      f"({len(matched)/max(len(nums),1)*100:.1f}%)")
print(f"  MISSED (read empty): {len(missed)}")
print(f"  NO-CELL (lattice hole): {len(nocell)}")
print(f"  EXTRA (no label):    {len(extra)}")

print(f"\n  PER-STRIP: verified vs occupied "
      f"(surplus red = marks/pins/markers):")
for si in range(len(lat["strips"])):
    n_occ_s = sum(1 for c in occ_cells if c[0] == si)
    n_ver = sum(1 for (nx, ny, c) in matched if c[0] == si)
    n_unv = n_occ_s - n_ver
    flag = f"  ← {n_unv} unverified" if n_unv > 1 else ""
    print(f"    S{si}: {n_ver}/{n_occ_s} occupied cells "
          f"label-verified{flag}")

per_strip = {}
for (nx, ny, c) in missed:
    per_strip.setdefault(c[0], []).append((c[1], c[2], c[3]))
if missed:
    print(f"\n  MISSED plugs by strip:")
    for s in sorted(per_strip):
        rows = [f"row{r}@({x},{y})" for r, x, y in per_strip[s]]
        print(f"    S{s}: {', '.join(rows)}")
if nocell:
    print(f"\n  SURPLUS red blobs (unassigned): {nocell}")

# ── Overlay ─────────────────────────────────────────────
vis = board.copy()
for (nx, ny, c) in matched:
    cv2.circle(vis, (c[2], c[3]), 4, (0, 220, 0), -1)
for (nx, ny, c) in missed:
    cv2.circle(vis, (c[2], c[3]), 9, (0, 0, 255), 2)
    cv2.line(vis, (nx, ny), (c[2], c[3]), (0, 0, 255), 1)
for (nx, ny) in nocell:
    cv2.drawMarker(vis, (nx, ny), (255, 0, 255),
                   cv2.MARKER_TILTED_CROSS, 12, 2)
for c in extra:
    cv2.circle(vis, (c[2], c[3]), 6, (0, 255, 255), 1)
cv2.putText(vis,
            f"green=match({len(matched)}) red=missed({len(missed)}) "
            f"magenta=no-cell({len(nocell)}) yellow=extra({len(extra)})",
            (5, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (255, 255, 255), 1)
save(out, "02_audit_overlay.jpg", vis)

with open(f"{out}/audit.json", "w") as f:
    json.dump({"labels": len(nums), "matched": len(matched),
               "missed": [(int(nx), int(ny), c[0], c[1])
                          for nx, ny, c in missed],
               "nocell": nocell,
               "extra": [(c[0], c[1]) for c in extra]}, f, indent=2)

print(f"\nDone! Check {out}/")
