"""Human-as-the-arm mode: live camera -> Gemma -> instruction -> a person does it -> repeat.

Same brain, same safety wrapper and arm limits as the robot; the "executor" is a person.
A window shows the live board with the dough outline and, while it waits for you, the
instruction and an arrow. The instruction is also spoken.

    python scripts/run_human.py                 # overhead webcam (index 0 at the venue)
    python scripts/run_human.py --camera 1
    python scripts/run_human.py --photos <dir>  # offline: each SPACE moves to the next photo
    python scripts/run_human.py --executor taught   # the real arm replays the taught stroke

Keys (click the window first):  SPACE / Enter = I've done it    S = stop
                                N = start a new roti            Q = quit
Calibrate first (board empty, quarter beside it): python scripts/calibrate_board.py --camera 0
"""
import argparse
import math
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from brain import gemma
from brain.voice import Voice
from robot.executor import make_executor
from robot.safety import SafetyWrapper
from vision.camera import Camera, Photos
from vision.measure import measure

VIEW = 640           # board view size in the window
PANEL = 230          # text panel height under it
SETTLE_S = 0.8       # after SPACE, let hands leave the frame before looking
WIN = "Roti coach"


class Session:
    """The decision loop on a worker thread; the window (main thread) only reads `state`."""

    def __init__(self, source, live, voice, max_steps, executor="human"):
        self.source, self.live, self.voice, self.max_steps = source, live, voice, max_steps
        self.human = make_executor(executor)
        self.robot = SafetyWrapper(self.human)
        self.human.on_instruction = self._instruct
        self.latest = None        # newest frame (PIL), board-cropped
        self.lock = threading.Lock()
        self.state = {"phase": "starting", "assessment": "", "instruction": "", "step": 0,
                      "latency": 0.0, "arrow": None, "turn": None, "report": ""}
        self.thread = None

    def start(self):
        self.robot.reset()
        self.robot.motions = 0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _set(self, **kw):
        with self.lock:
            self.state.update(kw)

    def _frame(self):
        if self.live:
            time.sleep(SETTLE_S)
            with self.lock:
                return self.latest.copy() if self.latest is not None else None
        f = self.source.read()
        with self.lock:
            self.latest = f
        return f

    def _instruct(self, text):
        a = self.state["assessment"].split(", so")[0].split(";")[0].rstrip(".")
        self.voice.say("%s. %s" % (a, text) if a else text)
        self._set(phase="your turn", instruction=text)

    def _run(self):
        history = []
        for step in range(1, self.max_steps + 1):
            if self.robot.stopped.is_set():
                return
            self._set(phase="looking", step=step, instruction="", arrow=None, turn=None)
            frame = self._frame()
            if frame is None:
                continue
            speech, name, args, dt = gemma.decide(frame, history, empty=self.source.empty,
                                                  board_cm=self.source.board_cm)
            if self.robot.stopped.is_set():
                return
            self._set(assessment=speech, latency=dt,
                      arrow=args.get("direction_deg") if name == "roll" else None,
                      turn=args.get("deg") if name == "rotate_dough" else None)
            print("[step %2d | gemma %.1f s] %s  -> %s %s" % (step, dt, speech, name, args))
            if name == "done":
                self.voice.say(speech + " Press N for a new roti.")
                self._set(phase="done", instruction="Finished! Press N for a new roti.")
                return
            if name == "stop":
                self.voice.say(speech)
                self.robot.stop(args.get("reason", "model stop"))
                self._set(phase="stopped", instruction="Stopped: %s. Press N to start over." % args.get("reason", ""))
                return
            if name == "inspect":
                history.append("inspect")
                continue
            status = self.robot.execute(name, args)  # blocks until SPACE for a valid instruction
            print("    safety: %s" % status)
            history.append("%s %s (%s)" % (name, args, status))
        self._set(phase="stopped", instruction="Step limit reached. Press N to start over.")


def draw(session, frame_rgb):
    """Board view with dough outline, arrow / turn hint, and the text panel. Returns BGR."""
    img = cv2.cvtColor(np.asarray(frame_rgb), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (VIEW, VIEW))
    with session.lock:
        st = dict(session.state)
    small = frame_rgb.copy()
    small.thumbnail((512, 512))
    m = measure(small, session.source.empty)
    cx, cy = VIEW // 2, VIEW // 2
    if m is not None:
        c = (m["contour"].astype(np.float32) * VIEW / small.width).astype(np.int32)
        cv2.drawContours(img, [c], -1, (80, 220, 80), 2)
        mo = cv2.moments(c)
        if mo["m00"]:
            cx, cy = int(mo["m10"] / mo["m00"]), int(mo["m01"] / mo["m00"])
    if st["phase"] == "your turn" and st["arrow"] is not None:
        t = math.radians(st["arrow"])
        dx, dy = int(140 * math.cos(t)), int(-140 * math.sin(t))
        cv2.arrowedLine(img, (cx - dx, cy - dy), (cx + dx, cy + dy), (0, 140, 255), 6, tipLength=0.15)
        cv2.arrowedLine(img, (cx + dx, cy + dy), (cx - dx, cy - dy), (0, 140, 255), 6, tipLength=0.15)
    if st["phase"] == "your turn" and st["turn"] is not None:
        deg = st["turn"]
        # Counter-clockwise on screen = decreasing OpenCV angle (y points down).
        a0, a1 = (0, -abs(deg)) if deg >= 0 else (0, abs(deg))
        cv2.ellipse(img, (cx, cy), (130, 130), 0, a0, a1, (255, 120, 0), 6)
        end = math.radians(a1)
        tip = (int(cx + 130 * math.cos(end)), int(cy + 130 * math.sin(end)))
        cv2.circle(img, tip, 12, (255, 120, 0), -1)

    panel = np.full((PANEL, VIEW, 3), 250, np.uint8)
    colour = {"your turn": (0, 120, 230), "done": (40, 160, 40), "stopped": (0, 0, 200)}.get(st["phase"], (90, 90, 90))
    put = lambda y, text, scale=0.6, col=(30, 30, 30), w=1: cv2.putText(
        panel, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, w, cv2.LINE_AA)
    put(30, "Step %d  |  %s  |  Gemma %.1f s" % (st["step"], st["phase"].upper(), st["latency"]), 0.7, colour, 2)
    y = 62
    for line in _wrap("Gemma: " + st["assessment"], 70)[:2]:
        put(y, line); y += 24
    for line in _wrap(st["instruction"], 52)[:3]:
        put(y + 6, line, 0.75, colour, 2); y += 30
    put(PANEL - 12, "SPACE = done   S = stop   N = new roti   Q = quit", 0.55, (110, 110, 110))
    return np.vstack([img, panel])


def _wrap(text, width):
    out, line = [], ""
    for w in text.split():
        if len(line) + len(w) + 1 > width:
            out.append(line)
            line = w
        else:
            line = (line + " " + w).strip()
    return out + ([line] if line else [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0, help="overhead webcam index (0 at the venue)")
    ap.add_argument("--photos", help="folder of rig photos instead of the camera")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--executor", default="human", choices=["human", "taught"],
                    help="taught = the real arm replays data/strokes/stroke.npz for every roll")
    a = ap.parse_args()

    live = not a.photos
    source = Camera(a.camera) if live else Photos(a.photos)
    if live and source.board is None:
        sys.exit("No board calibration. Run: python scripts/calibrate_board.py --camera %d" % a.camera)
    s = Session(source, live, Voice(not a.quiet), a.steps, a.executor)
    s.start()
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    while True:
        if live:
            f = source.read(flush=0)
            with s.lock:
                s.latest = f
        with s.lock:
            f = s.latest
        if f is not None:
            cv2.imshow(WIN, draw(s, f))
        k = cv2.waitKey(30) & 0xFF
        if k in (ord(" "), 13):
            s.human.done()
        elif k in (ord("s"), ord("S")):
            s.robot.stop("stop key")
            s._set(phase="stopped", instruction="Stopped. Press N to start over.")
        elif k in (ord("n"), ord("N")) and (s.thread is None or not s.thread.is_alive()):
            s._set(assessment="", instruction="", step=0, latency=0.0)
            s.start()
        elif k in (ord("q"), ord("Q"), 27):
            s.robot.stop("quit")
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
