"""
08 — Frequency Domain Analysis
=================================
What it does: Transforms the image from spatial domain (pixels) to
frequency domain (periodic patterns). The FFT reveals:
  - Repeating structures → bright spots at specific frequencies
  - The column spacing → a peak in the horizontal frequency
  - The row spacing → a peak in the vertical frequency

Why it matters:
  - A board with 9 evenly-spaced columns is literally a periodic signal
  - FFT can find that period even when individual walls are hard to see
  - This is what autocorrelation does internally, but visualized
"""

from common import load_image, make_output_dir, save, crop_board, remove_red_overlay

import cv2
import numpy as np

img, gray = load_image()
out = make_output_dir("08_frequency")

board = crop_board(remove_red_overlay(img), board_idx=0)
board_gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
bh, bw = board_gray.shape

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
enhanced = clahe.apply(board_gray).astype(np.float32)

# --- 2D FFT ---
# Pad to optimal size
rows = cv2.getOptimalDFTSize(bh)
cols = cv2.getOptimalDFTSize(bw)
padded = np.zeros((rows, cols), dtype=np.float32)
padded[:bh, :bw] = enhanced

# DFT
dft = cv2.dft(padded, flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft, axes=(0, 1))

# Magnitude spectrum (log scale for visibility)
mag = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
mag = np.log1p(mag)
mag = (mag / mag.max() * 255).astype(np.uint8)
save(out, "fft_magnitude.jpg", mag)

# --- 1D projections of the FFT (horizontal and vertical frequencies) ---
# Average magnitude along rows → shows vertical frequency content
h_freq = np.mean(mag, axis=0)
# Average along columns → shows horizontal frequency content
v_freq = np.mean(mag, axis=1)

# Draw as bar graphs
def draw_profile(profile, width, height, color=(0, 255, 0)):
    prof_img = np.zeros((height, width, 3), dtype=np.uint8)
    norm = profile / (profile.max() + 1e-6)
    for x in range(min(len(norm), width)):
        bar = int(norm[x] * height)
        cv2.line(prof_img, (x, height), (x, height - bar), color, 1)
    return prof_img

save(out, "h_frequency_profile.jpg",
     draw_profile(h_freq, len(h_freq), 200, (0, 200, 0)))
save(out, "v_frequency_profile.jpg",
     draw_profile(v_freq, len(v_freq), 200, (0, 100, 255)))

# --- Directional filtering in frequency domain ---
# Vertical pass filter: keep only horizontal frequencies
# (reveals vertical structures = column walls)
cy, cx = rows // 2, cols // 2
mask_vert = np.zeros((rows, cols, 2), dtype=np.float32)
band = 30  # how many pixels wide the pass band is
mask_vert[:, cx - band : cx + band, :] = 1.0
filtered_v = dft_shift * mask_vert
filtered_v = np.fft.ifftshift(filtered_v, axes=(0, 1))
result_v = cv2.idft(filtered_v)
result_v = cv2.magnitude(result_v[:, :, 0], result_v[:, :, 1])
result_v = result_v[:bh, :bw]
result_v = (result_v / result_v.max() * 255).astype(np.uint8)
save(out, "vert_structures_only.jpg", result_v)

# Horizontal pass filter: keep only vertical frequencies
# (reveals horizontal structures = row walls)
mask_horiz = np.zeros((rows, cols, 2), dtype=np.float32)
mask_horiz[cy - band : cy + band, :, :] = 1.0
filtered_h = dft_shift * mask_horiz
filtered_h = np.fft.ifftshift(filtered_h, axes=(0, 1))
result_h = cv2.idft(filtered_h)
result_h = cv2.magnitude(result_h[:, :, 0], result_h[:, :, 1])
result_h = result_h[:bh, :bw]
result_h = (result_h / result_h.max() * 255).astype(np.uint8)
save(out, "horiz_structures_only.jpg", result_h)

print(f"\nDone! Check {out}/")
