"""
49 — Decile Brightness Bands (new backlit frames)
==================================================
New frames: backlit boards, no number overlays. Convert to B&W,
split the 0-100% brightness scale into 10 even bands, save one
image per band (white = pixel lives in that band) for eyeball
feedback on where plugs / strips / structure live.
"""

from common import make_output_dir, save

import cv2
import numpy as np
import sys

FRAME = ("/root/.claude/uploads/608e2092-c86e-5429-b039-27e3dec388ed/"
         "6ef21155-image.png")

out = make_output_dir("49_decile_bands")
frame = cv2.imread(sys.argv[1] if len(sys.argv) > 1 else FRAME)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
h, w = gray.shape[:2]
print(f"  Frame: {w}x{h}  gray range {gray.min()}-{gray.max()} "
      f"mean {gray.mean():.0f}")

save(out, "00_grayscale.jpg", gray)

for i in range(10):
    lo = int(i * 25.6)
    hi = int((i + 1) * 25.6) if i < 9 else 256
    band = ((gray >= lo) & (gray < hi)).astype(np.uint8) * 255
    pct = np.mean(band > 0) * 100
    n_px = int(np.sum(band > 0))
    cv2.putText(band, f"band {i}: {lo}-{hi-1}  ({pct:.1f}% of image)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 255, 2)
    save(out, f"band_{i}_{lo:03d}-{hi-1:03d}.jpg", band)
    print(f"  band {i} [{lo:3d}-{hi-1:3d}]: {pct:5.1f}%  "
          f"({n_px} px)")

print(f"\nDone! Check {out}/")
