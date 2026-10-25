"""Run Gemma on every labelled dough photo, print its judgment + tool call, and score it.

Labels live in data/dough_photos/labels.csv: file, stage, expected, notes.
`expected` is a tool name, optionally with an angle ("roll 90"); a trailing "?" means
unsure and the photo is shown but not scored. Unlabelled photos are skipped.

Usage (venv active): python scripts/test_on_photos.py [labels.csv]
"""
import csv
import os
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain import gemma
from vision.camera import Photos, setup_circle

ANGLE_TOL = 30  # degrees; stroke axes, so 0 and 180 are the same


def norm(name):
    return unicodedata.normalize("NFKC", name)  # macOS screenshots use U+202F before "PM"


def angle_ok(got, want):
    d = abs(float(got) - want) % 180
    return min(d, 180 - d) <= ANGLE_TOL


def score(expected, name, args):
    """True/False, or None when the label is unsure."""
    if expected.endswith("?"):
        return None
    want = expected.split()
    if want == ["roll"] and name == "rotate_dough":  # turning the dough for a reachable stroke is progress too
        return True
    if name != want[0]:
        return False
    if len(want) > 1 and name == "roll":
        return angle_ok(args.get("direction_deg", -999), float(want[1]))
    return True


def main():
    labels_path = sys.argv[1] if len(sys.argv) > 1 else "data/dough_photos/labels.csv"
    folder = os.path.dirname(labels_path)
    with open(labels_path, newline="") as f:
        rows = list(csv.DictReader(f))
    on_disk = {norm(os.path.relpath(p, folder)): p
               for p in Photos(folder).files + Photos(os.path.join(folder, "jpg")).files}

    print("model: %s, %d labelled photos\n" % (gemma.MODEL, len(rows)))
    hits = scored = 0
    for row in rows:
        path = on_disk.get(norm(row["file"].strip()))
        if path is None:
            print("MISSING %s" % row["file"])
            continue
        if setup_circle(path) is None:
            print("skip    %-28s (no board calibration - not a rig shot)" % row["file"][-28:])
            continue
        src = Photos(path)
        speech, name, args, dt = gemma.decide(src.read(), empty=src.empty, board_cm=src.board_cm)
        ok = score(row["expected"].strip(), name, args)
        if ok is not None:
            scored += 1
            hits += ok
        mark = {True: "ok ", False: "XX ", None: " ? "}[ok]
        print("%s %-28s %4.1f s  want %-8s got %-6s %s" % (
            mark, row["file"][-28:], dt, row["expected"].strip(), name,
            "%.0f deg p%.1f" % (args.get("direction_deg", 0), args.get("pressure", 0)) if name == "roll" else ""))
        print("%40s\"%s\"" % ("", speech))
    print("\nSCORE %d / %d" % (hits, scored))


if __name__ == "__main__":
    main()
