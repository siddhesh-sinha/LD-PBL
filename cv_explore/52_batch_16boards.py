"""
52 — 16-Board Batch: robustness study across all 4 frames
==========================================================
Run the auto-calibration pipeline on every board of every frame
(4 x 4 = 16), collect per-board metrics, and hunt for flaws:
  - plug count / column count consistency
  - stray plugs (not assigned to any column) = artifact rate
  - plugs-per-column spread (a ragged column = detection wobble)
  - threshold spread (how much lighting varies board to board)
Saves per-board strip slices for board 0 of frame 1 as a visual
check against the manually-cut reference slices.
"""

from common import make_output_dir, save
from autocal import find_boards, label_board

import cv2
import numpy as np
import json

UP = "/root/.claude/uploads/608e2092-c86e-5429-b039-27e3dec388ed/"
FRAMES = ["6ef21155-image.png", "a9b3df51-image.png",
          "d1ce22df-image.png", "68c98889-image.png"]

out = make_output_dir("52_batch_16boards")
rows = []
for fi, name in enumerate(FRAMES):
    frame = cv2.imread(UP + name)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    boards = find_boards(gray)
    vis = frame.copy()
    for bi, (bx, by, bw_, bh_) in enumerate(boards):
        g = gray[by:by + bh_, bx:bx + bw_]
        Tp, Tg, plugs, gaps, strips, _, _ = label_board(g)

        # Column assignment stats
        col_cnt = []
        stray = len(plugs)
        for (sa, sb) in strips:
            c = sum(1 for p in plugs if sa <= p[0] <= sb)
            col_cnt.append(c)
            stray -= c
        med_col = int(np.median(col_cnt)) if col_cnt else 0

        rows.append({
            "frame": fi, "board": bi, "w": bw_, "h": bh_,
            "Tp": Tp, "Tg": Tg, "n_plugs": len(plugs),
            "n_strips": len(strips), "col_cnt": col_cnt,
            "med_col": med_col, "stray": stray,
            "col_min": min(col_cnt) if col_cnt else 0,
            "col_max": max(col_cnt) if col_cnt else 0,
        })

        for (px, py, a) in plugs:
            cv2.circle(vis, (bx + px, by + py), 4, (0, 255, 0), 1)
        for (sa, sb) in strips:
            cv2.rectangle(vis, (bx + sa, by + 5),
                          (bx + sb, by + bh_ - 5), (255, 0, 0), 1)
        cv2.putText(vis, f"F{fi}B{bi} {len(plugs)}p", (bx + 5, by - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        # Visual check: strip slices of frame 0, board 1 (like the
        # manually-cut reference slices)
        if fi == 0 and bi == 1:
            for si, (sa, sb) in enumerate(strips):
                sl = frame[by:by + bh_,
                           bx + max(0, sa - 8):bx + sb + 8]
                cv2.imwrite(f"{out}/slice_b1_s{si}.jpg", sl)

    save(out, f"frame{fi}_overlay.jpg", vis)

# ── Table ───────────────────────────────────────────────
print(f"\n{'='*74}")
print(f" {'F.B':>4} | {'WxH':>9} | {'Tp':>3} | {'Tg':>3} | "
      f"{'plugs':>5} | {'strips':>6} | {'col med/min/max':>15} | "
      f"{'stray':>5}")
print(f" {'-'*74}")
for r in rows:
    print(f" {r['frame']}.{r['board']:<2} | "
          f"{r['w']:>4}x{r['h']:<4} | {r['Tp']:>3} | {r['Tg']:>3} | "
          f"{r['n_plugs']:>5} | {r['n_strips']:>6} | "
          f"{r['med_col']:>5} /{r['col_min']:>3} /{r['col_max']:>3} | "
          f"{r['stray']:>5}")

# ── Cross-board robustness analysis ─────────────────────
print(f"\n{'='*74}")
print(f"  ROBUSTNESS ANALYSIS ({len(rows)} boards):")
np_, ns = [r["n_plugs"] for r in rows], [r["n_strips"] for r in rows]
print(f"  plugs/board:  med={np.median(np_):.0f}  "
      f"range {min(np_)}-{max(np_)}")
print(f"  strips/board: {sorted(set(ns))} "
      f"(consistent)" if len(set(ns)) == 1 else
      f"  strips/board: values {sorted(set(ns))} ← INCONSISTENT")
print(f"  Tp spread: {min(r['Tp'] for r in rows)}-"
      f"{max(r['Tp'] for r in rows)}   "
      f"Tg spread: {min(r['Tg'] for r in rows)}-"
      f"{max(r['Tg'] for r in rows)}")
tot_stray = sum(r["stray"] for r in rows)
tot = sum(np_)
print(f"  stray plugs: {tot_stray}/{tot} "
      f"({tot_stray/tot*100:.1f}% artifact rate)")

print(f"\n  FLAGS:")
for r in rows:
    flags = []
    if r["n_strips"] != 9:
        flags.append(f"strips={r['n_strips']}")
    if not (180 <= r["n_plugs"] <= 195):
        flags.append(f"plugs={r['n_plugs']}")
    if r["col_min"] < r["med_col"] - 3:
        flags.append(f"ragged col (min {r['col_min']})")
    if r["stray"] > 4:
        flags.append(f"stray={r['stray']}")
    if flags:
        print(f"    F{r['frame']}B{r['board']}: {', '.join(flags)}")

with open(f"{out}/batch.json", "w") as f:
    json.dump(rows, f, default=int)
print(f"\nDone! Check {out}/")
