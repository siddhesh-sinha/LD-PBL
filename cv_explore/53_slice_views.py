"""
53 — Slice Views: heat maps + BW masks of the manually-cut strips
==================================================================
The user hand-cut 3 strip slices (strips 9, 6, 10 — the last one
empty). For each: original | grayscale | heat map | plug mask |
bright mask, side by side, so we calibrate on the same visuals.
"""

from common import make_output_dir, save

import cv2
import numpy as np

UP = "/root/.claude/uploads/608e2092-c86e-5429-b039-27e3dec388ed/"
SLICES = [("strip9", "bb739927-image.png"),
          ("strip6", "79c0ae6e-image.png"),
          ("strip10_empty", "7e6df2a1-image.png")]

out = make_output_dir("53_slice_views")

for name, fn in SLICES:
    img = cv2.imread(UP + fn)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    print(f"  {name}: {w}x{h}  gray {gray.min()}-{gray.max()} "
          f"med {int(np.median(gray))}")

    # Heat map: brightness → TURBO (blue=dark ... red=bright)
    heat = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)

    # Plug mask: ratio of the strip surface brightness
    surf = np.percentile(gray, 70)
    Tp = int(np.clip(0.62 * surf, 30, 150))
    dark = cv2.cvtColor(((gray < Tp) * 255).astype(np.uint8),
                        cv2.COLOR_GRAY2BGR)

    # Bright mask: top brightness (gap/naked-light side)
    Tb = int(np.percentile(gray, 92))
    bright = cv2.cvtColor(((gray > Tb) * 255).astype(np.uint8),
                          cv2.COLOR_GRAY2BGR)

    gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    sep = np.full((h, 4, 3), (0, 0, 255), dtype=np.uint8)
    panel = np.hstack([img, sep, gray3, sep, heat, sep,
                       dark, sep, bright])

    # Upscale ×2 and label the columns
    panel = cv2.resize(panel, (panel.shape[1] * 2, panel.shape[0] * 2),
                       interpolation=cv2.INTER_NEAREST)
    labels = ["orig", "gray", "heat", f"plug<{Tp}", f"brt>{Tb}"]
    step = (w + 4) * 2
    for i, lb in enumerate(labels):
        cv2.putText(panel, lb, (i * step + 6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    save(out, f"{name}_panel.jpg", panel)

print(f"\nDone! Check {out}/")
