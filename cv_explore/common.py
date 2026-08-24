"""Shared utilities for cv_explore scripts."""

import cv2
import numpy as np
import os
import sys


def load_image(argv=None):
    """Load image from CLI arg or default test path. Returns (bgr, gray)."""
    if argv is None:
        argv = sys.argv
    if len(argv) > 1:
        path = argv[1]
    else:
        # Default: look for the uploaded snapshot
        candidates = [
            "/root/.claude/uploads/608e2092-c86e-5429-b039-27e3dec388ed/857fc50f-image.jpg",
            "test_image.jpg",
        ]
        path = next((p for p in candidates if os.path.exists(p)), None)
        if path is None:
            print("Usage: python3 <script>.py <image_path>")
            sys.exit(1)

    img = cv2.imread(path)
    if img is None:
        print(f"Error: cannot load {path}")
        sys.exit(1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"Loaded: {path} ({img.shape[1]}x{img.shape[0]})")
    return img, gray


def make_output_dir(name):
    """Create and return output directory for this technique."""
    out = os.path.join(os.path.dirname(__file__), "output", name)
    os.makedirs(out, exist_ok=True)
    return out


def save(out_dir, filename, image):
    """Save image to output dir. Returns full path."""
    path = os.path.join(out_dir, filename)
    cv2.imwrite(path, image)
    print(f"  → {filename}")
    return path


def crop_board(img, board_idx=0):
    """Quick crop of one board from the 2x2 grid. Returns cropped BGR."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    closed = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, np.ones((25, 25), np.uint8))
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = img.shape[:2]
    rects = sorted(
        [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > h * w * 0.03],
        key=lambda b: (0 if b[1] < h // 2 else 1, b[0]),
    )
    if board_idx >= len(rects):
        return img
    bx, by, bw, bh = rects[board_idx]
    return img[by : by + bh, bx : bx + bw]


def remove_red_overlay(img):
    """Remove Utpal's red number labels and blue lines via inpainting."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = (
        cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
        | cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
        | cv2.inRange(hsv, (100, 100, 100), (130, 255, 255))
    )
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
    return cv2.inpaint(img, mask, 7, cv2.INPAINT_TELEA)
