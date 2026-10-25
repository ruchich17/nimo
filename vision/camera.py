"""Frame sources, all returning a PIL image with the long side <= max_side.

Camera: overhead USB webcam (index 1 is usually the USB cam; 0 is the FaceTime cam).
Photos: cycles through a folder of dough photos, for offline runs.
Synthetic: a drawn lopsided blob, for testing with no camera at all.
"""
import glob
import json
import os
import unicodedata

from PIL import Image, ImageDraw, ImageOps

from vision import board

MAX_SIDE = 1024  # brain.gemma shrinks to 512 for the decision; the torn check zooms in
# The board circle comes from scripts/calibrate_board.py (data/board.json); every frame
# is cropped to it. Photos use the calibration of their setup in data/dough_photos/setups.json.


def shrink(img, max_side=MAX_SIDE):
    img = img.convert("RGB")
    img.thumbnail((max_side, max_side))
    return img


class Camera:
    def __init__(self, index=1, max_side=MAX_SIDE):
        import cv2
        self.cv2 = cv2
        self.max_side = max_side
        self.cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        if not self.cap.isOpened():
            raise RuntimeError("camera %d did not open - check the index and macOS camera "
                               "permission for this terminal (then restart the terminal)" % index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        for _ in range(10):  # let auto-exposure settle
            self.cap.read()
        self.board = board.load()
        if self.board is None:
            print("WARNING: no data/board.json - run scripts/calibrate_board.py; using the full frame")
        self.empty = board.load_empty(max_side)
        self.board_cm = board.load_cm()

    def read(self, flush=4):
        for _ in range(flush):  # drop buffered frames so we see the dough as it is now
            self.cap.grab()
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("camera read failed")
        img = Image.fromarray(self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB))
        return shrink(board.crop(img, self.board) if self.board else img, self.max_side)


class Photos:
    def __init__(self, path, max_side=MAX_SIDE):
        exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
        self.files = [path] if os.path.isfile(path) else sorted(
            f for e in exts for f in glob.glob(os.path.join(path, e)))
        if not self.files:
            raise RuntimeError("no photos in " + path)
        self.i = 0
        self.max_side = max_side
        self.empty = None  # empty-board reference for the photo last read, if its setup has one
        self.board_cm = None

    def read(self):
        f = self.files[self.i % len(self.files)]
        self.i += 1
        setup = photo_setup(f)
        if setup is None:
            self.empty = self.board_cm = None
            return shrink(ImageOps.exif_transpose(Image.open(f)), self.max_side)
        self.empty = None
        self.board_cm = setup.get("board_cm")
        if setup.get("empty"):
            e = ImageOps.exif_transpose(Image.open(_find_nfkc(os.path.dirname(f), setup["empty"])))
            self.empty = shrink(board.crop(e, setup["circle"]), self.max_side)
        img = ImageOps.exif_transpose(Image.open(f))
        return shrink(board.crop(img, setup["circle"]), self.max_side)


def setup_circle(path):
    s = photo_setup(path)
    return s and s["circle"]


def photo_setup(path):
    """A photo's setup from setups.json next to it:
    {name: {"match": substring, "circle": [cx, cy, r], "empty": optional empty-board photo}}.

    First entry whose substring is in the filename wins; "" matches everything.
    """
    f = os.path.join(os.path.dirname(os.path.abspath(path)), "setups.json")
    if not os.path.exists(f):
        return None
    with open(f) as fh:
        setups = json.load(fh)
    name = unicodedata.normalize("NFKC", os.path.basename(path))
    for s in setups.values():
        if s["match"] in name:
            return s
    return None


def _find_nfkc(folder, name):
    """Path of `name` in `folder`, matching macOS's U+202F-before-PM filenames too."""
    want = unicodedata.normalize("NFKC", name)
    for f in os.listdir(folder):
        if unicodedata.normalize("NFKC", f) == want:
            return os.path.join(folder, f)
    return os.path.join(folder, name)


class Synthetic:
    def read(self):
        img = Image.new("RGB", (512, 512), (120, 85, 50))  # wooden board
        d = ImageDraw.Draw(img)
        d.ellipse((90, 170, 400, 360), fill=(235, 215, 170))  # lopsided oval
        d.ellipse((60, 200, 220, 330), fill=(225, 200, 150))  # thick lump on the left
        return img


def make_source(camera=None, photos=None):
    if photos:
        return Photos(photos)
    if camera is not None:
        return Camera(camera)
    return Synthetic()
