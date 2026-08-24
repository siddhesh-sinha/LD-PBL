# Chamber Auto-Detection R&D

OpenCV prototyping to automatically detect fly-chamber grid structure from webcam snapshots,
replacing the manual ROI drawing step in Utpal's `calibrate_checker.py`.

## Status: Proof-of-concept successful

- **555 / 580 cells detected** (96% accuracy on 3 of 4 boards)
- Board detection: 100% reliable (Otsu + morphology)
- Column detection: autocorrelation on vertical edge projection — works on boards with consistent lighting
- Row detection: fly blob centroids → y-histogram peaks → regularized grid
- Known failure: Board 3 (bottom-left) — different lighting/angle broke both column and row detection

## Files

- `chamber_detect_v5.py` — Final working prototype (runs standalone, needs OpenCV + NumPy + SciPy)
- Earlier iterations (v1-v4) explored: raw Hough lines, morphological extraction, 
  autocorrelation-only, fly-based detection. v5 combines the best of each.

## Pipeline

```
HSV mask → inpaint (remove Utpal's red labels)
  → Otsu + morphology → find 4 boards
    → CLAHE + Canny → vertical projection → autocorrelation → columns
      → LAB blob detection → fly y-histogram → peaks → regularized rows
        → grid overlay
```

## v2 Recommendation

Semi-automatic calibration: user clicks 4 corners per board (16 clicks total),
then OpenCV perspective-warps each board and runs the auto-detection pipeline.
Interactive sliders let the user adjust col/row count + offset if detection fails.
Estimated calibration time: ~30 seconds (vs ~20 minutes with Utpal's manual tool).

## Full Report

See the interactive artifact: [DrosoLab Auto-Detection R&D](https://claude.ai/code/artifact/fe3e7362-4416-418e-b369-d76620d29b4e)
