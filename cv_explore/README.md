# cv_explore — OpenCV Feature Exploration for DrosoLab

Each file is one self-contained technique applied to a fly-chamber image.
Run any file standalone: `python3 cv_explore/<file>.py path/to/image.jpg`

## Files

### Core utilities
- `common.py` — Shared loader, output saver, image display helpers (~80 lines)

### Color & Contrast
- `01_color_spaces.py` — Split into RGB, HSV, LAB, grayscale channels
- `02_thresholding.py` — Otsu, adaptive (mean & gaussian), binary + inv
- `03_clahe.py` — CLAHE with different clip limits and tile sizes

### Edge Detection
- `04_edges.py` — Canny, Sobel (x/y/combined), Laplacian, Scharr

### Line Detection
- `05_hough_lines.py` — Standard + probabilistic Hough transform

### Morphology
- `06_morphology.py` — Erode, dilate, open, close, gradient, tophat, blackhat

### Blob Detection
- `07_blobs.py` — SimpleBlobDetector, contour-based blob finder

### Frequency Domain
- `08_frequency.py` — FFT, DCT, power spectrum (finds periodic patterns)

### Feature Detection
- `09_features.py` — ORB, SIFT keypoints + descriptors

### Projection Profiles
- `10_projections.py` — Horizontal/vertical intensity and edge projections

All outputs go to `cv_explore/output/<technique_name>/`
