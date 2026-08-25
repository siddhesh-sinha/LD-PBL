"""
50 — Two-Sided Cumulative Sweep (backlit frames)
=================================================
Attack the image from both ends of the brightness scale:
  DARK side:   gray < T, T rising  → what emerges from black
               (bet: plugs first)
  BRIGHT side: gray > T, T falling → what emerges from white
               (bet: the gaps between strips)
Cumulative masks, not bands — watch structure surface in order.
"""

from common import make_output_dir, save

import cv2
import numpy as np
import sys

FRAME = ("/root/.claude/uploads/608e2092-c86e-5429-b039-27e3dec388ed/"
         "6ef21155-image.png")

out = make_output_dir("50_two_sided_sweep")
frame = cv2.imread(sys.argv[1] if len(sys.argv) > 1 else FRAME)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
print(f"  gray range {gray.min()}-{gray.max()}")

# Dark side: rising ceiling — plugs should surface first.
# NB: the room background is also dark; restrict to the lit
# boards via a rough "board glow" mask (bright neighbourhood).
glow = cv2.blur(gray, (61, 61))
on_board = glow > 60

for T in (30, 45, 60, 75, 90, 105, 120):
    m = ((gray < T) & on_board).astype(np.uint8) * 255
    pct = np.mean(m > 0) * 100
    cv2.putText(m, f"DARK side: gray<{T} on-board ({pct:.1f}%)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 255, 2)
    save(out, f"dark_{T:03d}.jpg", m)

# Bright side: falling floor — what glows hardest surfaces first
for T in (195, 185, 175, 165, 155, 145, 135, 125):
    m = (gray > T).astype(np.uint8) * 255
    pct = np.mean(m > 0) * 100
    cv2.putText(m, f"BRIGHT side: gray>{T} ({pct:.1f}%)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 255, 2)
    save(out, f"bright_{T:03d}.jpg", m)

print(f"\nDone! Check {out}/")
