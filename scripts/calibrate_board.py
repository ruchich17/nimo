"""Find the board once and save it. Run after taping down the camera and board,
with the board EMPTY (no dough) and the quarter beside it.

  python scripts/calibrate_board.py --camera 1
        # live rig -> data/board.json (circle + board size from the quarter)
        #             data/board_empty.jpg (empty-board reference for dough detection)
  python scripts/calibrate_board.py --photos "data/dough_photos/lopsided*.jpg" --name rig2 --match "" \
        --empty "Photo on 10-24-26 at 5.03 PM.jpg"
        # a photo set -> data/dough_photos/setups.json (for test_on_photos.py)

Writes an overlay image: CHECK IT. The green circle must sit on the board's edge.
If it doesn't, move clutter off the board and run it again.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageOps

from vision import board


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int)
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--photos", help="glob of rig photos from ONE setup")
    ap.add_argument("--name", help="setup name in setups.json (photos only)")
    ap.add_argument("--match", default="", help="filename substring that selects this setup")
    ap.add_argument("--empty", help="empty-board photo for this setup, in the same folder (photos only)")
    a = ap.parse_args()

    if a.camera is not None:
        import cv2
        cap = cv2.VideoCapture(a.camera, cv2.CAP_AVFOUNDATION)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        for _ in range(10):
            cap.read()
        frames = []
        for _ in range(a.frames):
            ok, f = cap.read()
            if ok:
                frames.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
        out_dir = os.path.dirname(board.BOARD_FILE)
    else:
        paths = sorted(glob.glob(a.photos))
        frames = [ImageOps.exif_transpose(Image.open(p)).convert("RGB") for p in paths]
        out_dir = os.path.dirname(paths[0]) if paths else "."
    if not frames:
        sys.exit("no frames")

    circle = board.calibrate(frames)
    if circle is None:
        sys.exit("board not found in any frame")
    cm = board.board_cm(frames, circle)
    print("board circle (fractions): cx %.3f cy %.3f r %.3f, from %d frames" % (circle + (len(frames),)))
    print("board diameter: %s" % ("%.1f cm (from the quarter)" % cm if cm else "unknown - quarter not found"))

    if a.camera is not None:
        board.save(circle, cm)
        board.crop(frames[-1], circle).save(board.EMPTY_FILE)
        print("saved", board.BOARD_FILE, "and", board.EMPTY_FILE, "(assumes the board was EMPTY)")
        check = os.path.join(out_dir, "board_check.jpg")
    else:
        f = os.path.join(out_dir, "setups.json")
        setups = json.load(open(f)) if os.path.exists(f) else {}
        setups[a.name or "default"] = {"match": a.match, "circle": list(circle), "board_cm": cm, "empty": a.empty}
        # Specific matches first, the catch-all "" last.
        setups = dict(sorted(setups.items(), key=lambda kv: kv[1]["match"] == ""))
        json.dump(setups, open(f, "w"), indent=1)
        print("saved", f)
        check = os.path.join(out_dir, "board_check_%s.jpg" % (a.name or "default"))
    board.overlay(frames[-1], circle).save(check)
    print("CHECK:", check)


if __name__ == "__main__":
    main()
