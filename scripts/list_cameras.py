"""Grab one frame from each camera index and save a labelled contact sheet, so you can
see which index is the overhead webcam.  python scripts/list_cameras.py  ->  data/cameras.jpg
"""
import os
import sys

import cv2
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cameras.jpg")

tiles = []
for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        continue
    for _ in range(10):  # let exposure settle
        ok, frame = cap.read()
    cap.release()
    if not ok:
        continue
    h, w = frame.shape[:2]
    print("camera %d: %dx%d" % (i, w, h))
    t = cv2.resize(frame, (480, int(480 * h / w)))
    cv2.putText(t, "--camera %d" % i, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    tiles.append(t)
if not tiles:
    sys.exit("no cameras opened - check camera permission for this terminal app")
hmax = max(t.shape[0] for t in tiles)
tiles = [np.vstack([t, np.zeros((hmax - t.shape[0], 480, 3), np.uint8)]) for t in tiles]
cv2.imwrite(OUT, np.hstack(tiles))
print("saved", OUT)
