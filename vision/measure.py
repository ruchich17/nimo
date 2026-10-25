"""Measure the dough with plain OpenCV, so Gemma reasons over numbers instead of guessing.

Expects a frame already cropped by vision.board.crop: the board is the centred circle.
With an empty-board reference (same crop), the dough is whatever differs from it: glare
is in the reference too, so it cancels. Without one, the dough is the largest pale blob
clear of the rim (glare patches on a shiny board are pale too, but run out to the edge).
"""
import cv2
import numpy as np

from vision.board import CROP_MARGIN

ANGLES = np.arange(0, 180, 5)
DIFF_LAB = 15  # Lab colour distance from the empty board that counts as dough
RIM = 0.85  # blobs reaching beyond this fraction of the board radius are glare, not dough


def measure(img, empty=None):
    """Board-cropped PIL image -> dict, or None when no dough is found.

    empty: board-cropped PIL image of the same board with no dough, or None.

    coverage       dough diameter / board diameter (area-equivalent), 0..1
    aspect         widest extent / narrowest extent of the dough, >= 1
    long_axis_deg  direction the dough is widest
    narrow_deg     direction the dough is narrowest -- rolling along it spreads the dough there
    roundness      4*pi*area / perimeter^2, 1 = perfect circle
    Angles are in image terms, counter-clockwise: 0 = left-right, 90 = up-down.
    """
    hsv = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    cy, cx = h.shape[0] / 2, h.shape[1] / 2
    r = min(cx, cy) / (1 + CROP_MARGIN)
    inside = np.zeros(h.shape, np.uint8)
    cv2.circle(inside, (round(cx), round(cy)), round(r * 0.97), 1, -1)

    if empty is not None:
        pale = _differs(img, empty) & inside
    else:
        pale = ((s < 90) & (v > 120)).astype(np.uint8) & inside
        pale = cv2.morphologyEx(pale, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8), iterations=2)
    # Drop blobs reaching the rim (glare, or the board edge if it shifted); the dough sits
    # well inside it.
    rim = np.zeros(h.shape, np.uint8)
    cv2.circle(rim, (round(cx), round(cy)), round(r * RIM), 1, -1)
    n, labels = cv2.connectedComponents(pale)
    for i in np.unique(labels[(pale > 0) & (rim == 0)]):
        pale[labels == i] = 0
    dough = _largest(pale, 0.002)
    if dough is None or len(dough) < 5:
        return None

    pts = dough.reshape(-1, 2).astype(float)
    th = np.radians(ANGLES)
    # Image y points down, so "up" is -y: direction a is (cos a, -sin a) in pixels.
    proj = pts @ np.stack([np.cos(th), -np.sin(th)])
    extent = proj.max(0) - proj.min(0)
    area = cv2.contourArea(dough)
    return {
        "coverage": float(np.sqrt(area / (np.pi * r * r))),
        "aspect": float(extent.max() / max(extent.min(), 1e-6)),
        "long_axis_deg": float(ANGLES[extent.argmax()]),
        "narrow_deg": float(ANGLES[extent.argmin()]),
        "roundness": float(4 * np.pi * area / cv2.arcLength(dough, True) ** 2),
        "contour": dough,  # for drawing / debugging
    }


def _differs(img, empty):
    """1 where the frame's colour differs from the empty board by more than DIFF_LAB."""
    a = np.asarray(img.convert("RGB"))
    e = np.asarray(empty.convert("RGB").resize(img.size))
    la, le = (cv2.cvtColor(cv2.GaussianBlur(x, (9, 9), 0), cv2.COLOR_RGB2LAB).astype(np.float32) for x in (a, e))
    mask = (np.sqrt(((la - le) ** 2).sum(-1)) > DIFF_LAB).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=2)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)


def _largest(mask, min_frac):
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    return c if cv2.contourArea(c) > min_frac * mask.size else None
