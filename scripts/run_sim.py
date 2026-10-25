"""Run the roti loop: camera -> Gemma -> safety -> executor.

Usage (venv active):
  python scripts/run_sim.py                       # synthetic blob, stub executor, no camera
  python scripts/run_sim.py --camera 1            # overhead USB webcam
  python scripts/run_sim.py --photos data/dough_photos
  mjpython scripts/run_sim.py --camera 1 --executor sim   # MuJoCo reBot (Plan A)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain.loop import run
from brain.voice import Voice
from robot.executor import make_executor
from robot.safety import SafetyWrapper
from vision.camera import make_source


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, help="webcam index (USB cam is usually 1)")
    ap.add_argument("--photos", help="folder or file of dough photos instead of a camera")
    ap.add_argument("--executor", default="stub", choices=["stub", "sim", "real"])
    ap.add_argument("--steps", type=int, default=25, help="max decisions")
    ap.add_argument("--quiet", action="store_true", help="no spoken narration")
    a = ap.parse_args()

    source = make_source(a.camera, a.photos)
    robot = SafetyWrapper(make_executor(a.executor))
    run(source, robot, Voice(not a.quiet), a.steps)


if __name__ == "__main__":
    main()
