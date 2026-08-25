"""
51 — Auto-Calibrating Two-Ended Structure Finder (single frame)
================================================================
Core lives in autocal.py; this runs one frame and saves overlay
+ calibration.json. Usage: python3 51_auto_calibrate.py [image]
"""

from common import make_output_dir, save
from autocal import FRAME, find_boards, label_board

import cv2
import numpy as np
import json
import sys

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
