"""Find the round board once per setup, then crop every frame to it.

The camera and board are taped down for the demo, so the board circle is calibrated
once (median over several frames) and saved, not re-detected per frame -- per-frame
colour detection breaks under glare. Circles are stored as fractions of the frame
width/height so they survive a resolution change, but not a change of aspect ratio:
calibrate the live camera from live frames.

    python scripts/calibrate_board.py --camera 1       # live rig
    python scripts/calibrate_board.py --photos <glob>   # a set of rig photos
"""
import json
import os

import cv2
import numpy as np

BOARD_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "board.json")
EMPTY_FILE = os.path.join(os.path.dirname(BOARD_FILE), "board_empty.jpg")  # cropped, no dough
CROP_MARGIN = 0.15  # of the board radius, each side
DETECT_SIDE = 640
MIN_WOOD = 0.5  # fraction of the circle that must be wood-coloured to count as the board


def detect(img):
    """(cx, cy, r) as fractions of (width, height, width) for one frame, or None.

    Hough circle candidates, scored by wood-coloured pixels inside minus everything else,
    so the circle that hugs the board wins over ones spilling onto the towel or table.
    """
    small = img.convert("RGB").copy()
    small.thumbnail((DETECT_SIDE, DETECT_SIDE))
    a = np.asarray(small)
    hsv = cv2.cvtColor(a, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    wood = (h >= 9) & (h <= 25) & (s >= 50) & (v >= 40)  # towel is h~4, board h~14-17
    g = cv2.medianBlur(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), 7)
    mn = min(g.shape)
    cs = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.5, minDist=mn // 10, param1=80, param2=30,
                          minRadius=int(mn * 0.15), maxRadius=int(mn * 0.5))
    if cs is None:
        return None
    yy, xx = np.mgrid[:g.shape[0], :g.shape[1]]
    best = None
    for x, y, r in cs[0][:25]:
        inside = (xx - x) ** 2 + (yy - y) ** 2 <= r * r
        w = wood[inside].sum()
        score = w - (inside.sum() - w)
        if best is None or score > best[0]:
            best = (score, x, y, r)
    _, x, y, r = best
    inside = (xx - x) ** 2 + (yy - y) ** 2 <= r * r
    if wood[inside].mean() < MIN_WOOD:  # a round thing that isn't the board (a face, a plate)
        return None
    return x / a.shape[1], y / a.shape[0], r / a.shape[1]


QUARTER_MM = 24.26  # US quarter diameter: the scale reference kept beside the board


def find_quarter(img, circle):
    """Radius in pixels (of `img`) of the quarter beside the board, or None.

    Small Hough circles outside the board but within 1.7 board radii, keeping the one whose
    inside is most silver (least saturated). Its size gives the board real-world scale.
    """
    small = img.convert("RGB").copy()
    small.thumbnail((1280, 1280))
    k = img.width / small.width
    a = np.asarray(small)
    w, h = small.size
    bx, by, br = circle[0] * w, circle[1] * h, circle[2] * w
    g = cv2.medianBlur(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), 5)
    cs = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.2, minDist=br * 0.1, param1=100, param2=20,
                          minRadius=int(br * 0.05), maxRadius=int(br * 0.15))
    if cs is None:
        return None
    sat = cv2.cvtColor(a, cv2.COLOR_RGB2HSV)[..., 1]
    yy, xx = np.mgrid[:h, :w]
    best = None
    for x, y, r in cs[0]:
        d = np.hypot(x - bx, y - by)
        if not (br + r < d < 1.7 * br):
            continue
        s = np.median(sat[(xx - x) ** 2 + (yy - y) ** 2 <= (0.8 * r) ** 2])
        if s < 60 and (best is None or s < best[0]):
            best = (s, x, y, r)
    if best is None:
        return None
    # Hough radii wobble ~20%; refine from the silver blob's area (pink towel is saturated).
    _, x, y, r = best
    silver = ((sat < 70) & ((xx - x) ** 2 + (yy - y) ** 2 <= (1.6 * r) ** 2)).astype(np.uint8)
    silver = cv2.morphologyEx(silver, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    _, labels = cv2.connectedComponents(silver)
    blob = labels == labels[int(y), int(x)]
    if labels[int(y), int(x)] and blob.sum() > 0.5 * np.pi * r * r:
        r = np.sqrt(blob.sum() / np.pi)
    return x * k, y * k, r * k


def calibrate(frames):
    """Median board circle over several frames (fractions), or None if never found."""
    found = [c for c in (detect(f) for f in frames) if c is not None]
    if not found:
        return None
    return tuple(float(v) for v in np.median(np.array(found), axis=0))


def board_cm(frames, circle):
    """Board diameter in cm from the quarter, median over frames, or None."""
    sizes = []
    for f in frames:
        q = find_quarter(f, circle)
        if q is not None:
            sizes.append(2 * circle[2] * f.width * QUARTER_MM / (2 * q[2]) / 10)
    return float(np.median(sizes)) if sizes else None


def save(circle, cm=None, path=BOARD_FILE):
    with open(path, "w") as f:
        json.dump({"cx": circle[0], "cy": circle[1], "r": circle[2], "board_cm": cm}, f, indent=1)


def load_cm(path=BOARD_FILE):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f).get("board_cm")


def load(path=BOARD_FILE):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    return d["cx"], d["cy"], d["r"]


def load_empty(max_side, path=EMPTY_FILE):
    """The saved empty-board reference, shrunk like live frames, or None."""
    if not os.path.exists(path):
        return None
    from PIL import Image
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side))
    return img


def crop(img, circle):
    """Square crop around the board; the board is centred with CROP_MARGIN around it."""
    w, h = img.size
    cx, cy, r = circle[0] * w, circle[1] * h, circle[2] * w
    half = r * (1 + CROP_MARGIN)
    return img.crop((round(cx - half), round(cy - half), round(cx + half), round(cy + half)))


def overlay(img, circle):
    """The frame with the calibrated circle drawn on it, for a human to check."""
    a = np.asarray(img.convert("RGB")).copy()
    w, h = img.size
    cv2.circle(a, (round(circle[0] * w), round(circle[1] * h)), round(circle[2] * w), (0, 255, 0), max(2, w // 300))
    from PIL import Image
    return Image.fromarray(a)
