"""Teach mode: record a stroke by moving the arm by hand, then replay it.

    python scripts/teach.py record stroke     # arm limp, gripper holds the roller
    python scripts/teach.py replay stroke     # slow replay (--speed 0.5 = half speed)
    python scripts/teach.py release           # make the arm limp (SUPPORT IT FIRST)

Recordings go to data/strokes/<name>.npz. Ctrl-C during replay FREEZES the arm in place;
it only goes limp when you say someone is holding it.
"""
import argparse
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from robot.rebot_sdk import make_arm

STROKES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "strokes")
REC_HZ = 50
ARM_VLIM = 0.6       # rad/s per arm joint during replay (config default is 5.0)
GRIP_VLIM = 1.0
APPROACH_RAD_S = 0.3  # speed of the move from wherever the arm is to the stroke's start


class Arm:
    """The SDK arm with a target that a 500 Hz loop keeps sending once enabled."""

    def __init__(self):
        self.r = make_arm()
        self.r.connect()
        self.n = self.r.arm.num_joints
        self.target = None
        self.lock = threading.Lock()
        self.enabled = False

    def q(self):
        return self.r.get_positions()

    def _loop(self, r, dt):
        with self.lock:
            t = self.target.copy()
        r.arm.send_pos_vel(t[:self.n])
        if r.has_gripper:
            r.gripper.send_pos_vel(t[self.n:])

    def hold_here(self, groups="all"):
        """Enable and hold the current pose ("gripper" = only the gripper, arm stays limp)."""
        with self.lock:
            self.target = self.q().copy()
        if groups == "gripper":
            self.r.gripper.mode_pos_vel(vlim=np.full(self.r.gripper.num_joints, GRIP_VLIM))
            self.r.gripper.enable()
            self.r.start_control_loop(lambda r, dt: r.gripper.send_pos_vel(self.target[self.n:]))
            return
        self.r.arm.mode_pos_vel(vlim=np.full(self.n, ARM_VLIM))
        if self.r.has_gripper:
            self.r.gripper.mode_pos_vel(vlim=np.full(self.r.gripper.num_joints, GRIP_VLIM))
        self.r.enable_all()
        self.enabled = True
        if self.r.control_loop_active:  # e.g. the gripper-only loop from recording
            self.r.stop_control_loop()
        self.r.start_control_loop(self._loop)

    def go(self, q_to, seconds):
        """Linear move of the target to q_to over `seconds` (the motors' vlim caps speed too)."""
        with self.lock:
            q0 = self.target.copy()
        steps = max(1, int(seconds * 100))
        for i in range(1, steps + 1):
            with self.lock:
                self.target = q0 + (q_to - q0) * i / steps
            time.sleep(seconds / steps)

    def freeze(self):
        with self.lock:
            self.target = self.q().copy()

    def release(self):
        if self.r.control_loop_active:
            self.r.stop_control_loop()
        self.r.disable_all()


def record(name):
    arm = Arm()
    input("\n1) Put the roller handle in the gripper fingers and close the fingers on it BY HAND.\n"
          "   Keep holding them closed, then press Enter (the gripper will lock).")
    arm.hold_here("gripper")
    print("   Gripper locked at %.1f deg." % np.degrees(arm.target[arm.n]))
    input("\n2) The ARM IS LIMP: support it. Move it to the start of the stroke (roller just above\n"
          "   the near edge of the dough). Press Enter to START recording.")
    rec, t0, stop = [], time.time(), threading.Event()

    def sampler():
        while not stop.is_set():
            rec.append((time.time() - t0, arm.q()))
            time.sleep(1.0 / REC_HZ)
    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    input("   RECORDING... lower the roller onto the dough, roll forward and back, lift it.\n"
          "   Press Enter to STOP.")
    stop.set()
    th.join()
    t = np.array([r[0] for r in rec])
    q = np.array([r[1] for r in rec])
    os.makedirs(STROKES, exist_ok=True)
    path = os.path.join(STROKES, name + ".npz")
    np.savez(path, t=t, q=q)
    span = np.degrees(q.max(0) - q.min(0))
    print("\nSaved %s: %d samples, %.1f s. Joint ranges (deg): %s" % (
        path, len(t), t[-1], " ".join("%.0f" % s for s in span)))
    input("\n3) Keep supporting the arm and press Enter to LOCK it in place.")
    arm.hold_here()
    input("   Arm locked. Rest it safely if you like, then press Enter: the arm will go LIMP.\n"
          "   SUPPORT IT FIRST.")
    arm.release()
    print("Released.")


def replay(name, speed, arm=None, confirm=True):
    """Move to the recorded start, play the stroke, end back at its start. Returns the Arm."""
    d = np.load(os.path.join(STROKES, name + ".npz"))
    t, q = d["t"] / speed, d["q"]
    arm = arm or Arm()
    if confirm:
        input("\nThe arm will MOVE (slowly). Clear the area, hand on the stop. Press Enter to start.")
    if not arm.enabled:
        arm.hold_here()
    dist = np.abs(q[0][:arm.n] - arm.target[:arm.n]).max()
    print("Moving to the stroke start (%.0f deg away)..." % np.degrees(dist))
    arm.go(q[0], max(1.5, dist / APPROACH_RAD_S))
    print("Playing the stroke (%.1f s)..." % t[-1])
    start = time.time()
    for i in range(1, len(t)):
        dt = t[i] - (time.time() - start)
        if dt > 0:
            time.sleep(dt)
        with arm.lock:
            arm.target = q[i].copy()
    arm.go(q[0], 2.0)
    print("Done; holding at the stroke start.")
    return arm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["record", "replay", "release"])
    ap.add_argument("name", nargs="?", default="stroke")
    ap.add_argument("--speed", type=float, default=0.5, help="replay speed vs the recording (0.5 = half)")
    a = ap.parse_args()
    arm = None
    try:
        if a.cmd == "record":
            record(a.name)
        elif a.cmd == "replay":
            arm = replay(a.name, a.speed)
            input("Press Enter to make the arm LIMP (SUPPORT IT FIRST), or Ctrl-C to keep it locked and exit.")
            arm.release()
        else:
            arm = Arm()
            input("The arm will go LIMP. Support it, then press Enter.")
            arm.release()
    except KeyboardInterrupt:
        if arm is not None and arm.enabled:
            arm.freeze()
            print("\nFROZEN in place.")
            try:
                input("Support the arm, then press Enter to make it limp (Ctrl-C again leaves it locked).")
                arm.release()
            except KeyboardInterrupt:
                print("\nLeft locked. Run `python scripts/teach.py release` when it's supported.")


if __name__ == "__main__":
    main()
