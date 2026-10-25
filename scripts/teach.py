"""Teach mode: record a stroke by moving the arm by hand, then replay it.

    python scripts/teach.py gripcal           # once: open/close the fingers by hand to calibrate
    python scripts/teach.py grip              # test: open, grip the roller firmly, open
    python scripts/teach.py record stroke     # arm limp, gripper holds the roller
    python scripts/teach.py replay stroke     # slow replay (--speed 0.5 = half speed)
    python scripts/teach.py release           # make the arm limp (SUPPORT IT FIRST)

Recordings go to data/strokes/<name>.npz. Ctrl-C during replay FREEZES the arm in place;
it only goes limp when you say someone is holding it.
"""
import argparse
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from motorbridge import Mode

from robot.rebot_sdk import make_arm

STROKES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "strokes")
REC_HZ = 50
ARM_VLIM = 0.6       # rad/s per arm joint during replay (config default is 5.0)
GRIP_VLIM = 1.0
APPROACH_RAD_S = 0.3  # speed of the move from wherever the arm is to the stroke's start
GRIP_CAL = os.path.join(os.path.dirname(STROKES), "gripper.json")
GRIP_KP, GRIP_KD = 8.0, 1.0  # MIT stiffness (SDK default for the gripper); force = KP x squeeze
GRIP_SWEEP = 0.8      # rad/s while opening/closing
GRIP_STALL = 0.15     # target ahead of the fingers by this much (rad) = they've hit the handle
GRIP_SQUEEZE = 0.3    # aim this far past contact: ~2.4 N*m of squeeze


class Arm:
    """The SDK arm driven by one 200 Hz loop of ours.

    Arm joints: position-velocity mode, only once enabled (hold_here). Gripper: its own
    motor, switched on alone, in MIT ("spring") mode so it can squeeze the roller handle
    with a set force while the arm stays limp for teaching.
    """

    def __init__(self):
        self.r = make_arm()
        self.r.connect()
        self.n = self.r.arm.num_joints
        self.gm = self.r._motor_map["gripper"]  # the one gripper motor
        self.target = self.q().copy()
        self.grip = None        # gripper MIT target (rad) while gripping
        self.arm_on = False
        self.lock = threading.Lock()
        self._alive = True
        threading.Thread(target=self._loop, daemon=True).start()

    @property
    def enabled(self):
        return self.arm_on

    def q(self):
        return self.r.get_positions()

    def _loop(self):
        while self._alive:
            with self.lock:
                t, g, on = self.target.copy(), self.grip, self.arm_on
            try:
                if on:
                    self.r.arm.send_pos_vel(t[:self.n])
                if g is not None:
                    self.gm.send_mit(float(g), 0.0, GRIP_KP, GRIP_KD, 0.0)
            except Exception as e:
                print("[loop]", e)
            time.sleep(0.005)

    # ---- gripper ----
    def grip_pos(self):
        return float(self.q()[self.n])

    def _grip_on(self):
        if self.grip is None:
            with self.lock:
                self.grip = self.grip_pos()
            self.gm.ensure_mode(Mode.MIT, 1000)
            self.gm.enable()
            time.sleep(0.1)

    def _sweep(self, to, stop_on_contact):
        """Move the grip target toward `to`; with stop_on_contact, stop where the fingers stall."""
        self._grip_on()
        sign = 1.0 if to > self.grip else -1.0
        while (to - self.grip) * sign > 0:
            with self.lock:
                self.grip += sign * GRIP_SWEEP * 0.02
            time.sleep(0.02)
            if stop_on_contact and (self.grip - self.grip_pos()) * sign > GRIP_STALL:
                contact = self.grip_pos()
                with self.lock:
                    self.grip = contact + sign * GRIP_SQUEEZE
                return contact
        return None

    def open_gripper(self):
        self._sweep(gripper_cal()["open"], stop_on_contact=False)

    def grip_firmly(self):
        """Close until the fingers stop on the handle, then squeeze GRIP_SQUEEZE past contact."""
        cal = gripper_cal()
        contact = self._sweep(cal["closed"], stop_on_contact=True)
        if contact is None:
            print("   WARNING: the fingers closed fully - nothing between them?")
        else:
            print("   Gripping: contact at %.0f deg, squeezing %.0f deg past it." % (
                np.degrees(contact), np.degrees(GRIP_SQUEEZE)))
        return contact

    # ---- arm ----
    def hold_here(self):
        """Enable the arm joints and hold the current pose."""
        with self.lock:
            self.target = self.q().copy()
        self.r.arm.mode_pos_vel(vlim=np.full(self.n, ARM_VLIM))
        for name in self.r.arm.joint_names:
            self.r._motor_map[name].enable()
        with self.lock:
            self.arm_on = True

    def go(self, q_to, seconds):
        """Linear move of the arm target to q_to over `seconds` (the motors' vlim caps speed too)."""
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

    def release(self, gripper=False):
        """Arm joints limp (and the gripper too if gripper=True)."""
        with self.lock:
            self.arm_on = False
        for name in self.r.arm.joint_names:
            self.r._motor_map[name].disable()
        if gripper:
            with self.lock:
                self.grip = None
            self.gm.disable()


def gripper_cal():
    if not os.path.exists(GRIP_CAL):
        sys.exit("No gripper calibration - run: python scripts/teach.py gripcal")
    return json.load(open(GRIP_CAL))


def gripcal():
    arm = Arm()
    input("\nGripper motor is OFF. Open the fingers FULLY by hand, then press Enter.")
    o = arm.grip_pos()
    input("Now close the fingers FULLY by hand (nothing between them), then press Enter.")
    c = arm.grip_pos()
    if abs(o - c) < 0.2:
        sys.exit("Open and closed read almost the same (%.0f vs %.0f deg) - try again." % (np.degrees(o), np.degrees(c)))
    json.dump({"open": o, "closed": c}, open(GRIP_CAL, "w"))
    print("Saved %s: open %.0f deg, closed %.0f deg." % (GRIP_CAL, np.degrees(o), np.degrees(c)))


def grip_test():
    arm = Arm()
    print("\nOpening the gripper...")
    arm.open_gripper()
    input("Hold the roller with its orange handle between the fingers, then press Enter to GRIP.")
    arm.grip_firmly()
    input("Gripping. Tug the roller gently - it should not slip. Press Enter to OPEN (hold the roller!).")
    arm.open_gripper()
    time.sleep(0.3)
    arm.release(gripper=True)
    print("Opened and released.")


def record(name):
    arm = Arm()
    print("\n1) Opening the gripper...")
    arm.open_gripper()
    input("   Hold the roller with its orange handle between the fingers, then press Enter to GRIP.")
    arm.grip_firmly()
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
    print("Arm released; the gripper still holds the roller.")
    input("Press Enter to OPEN the gripper (hold the roller!) and finish.")
    arm.open_gripper()
    arm.release(gripper=True)


def replay(name, speed, arm=None, confirm=True):
    """Move to the recorded start, play the stroke, end back at its start. Returns the Arm."""
    d = np.load(os.path.join(STROKES, name + ".npz"))
    t, q = d["t"] / speed, d["q"]
    arm = arm or Arm()
    if confirm:
        input("\nThe arm will MOVE (slowly). Clear the area, hand on the stop. Press Enter to start.")
    if arm.grip is None:
        print("Gripping the roller (it must already be between the fingers)...")
        arm.grip_firmly()
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
    ap.add_argument("cmd", choices=["gripcal", "grip", "record", "replay", "release"])
    ap.add_argument("name", nargs="?", default="stroke")
    ap.add_argument("--speed", type=float, default=0.5, help="replay speed vs the recording (0.5 = half)")
    a = ap.parse_args()
    arm = None
    try:
        if a.cmd == "gripcal":
            gripcal()
        elif a.cmd == "grip":
            grip_test()
        elif a.cmd == "record":
            record(a.name)
        elif a.cmd == "replay":
            arm = replay(a.name, a.speed)
            input("Press Enter to make the arm LIMP (SUPPORT IT FIRST), or Ctrl-C to keep it locked and exit.")
            arm.release()
            input("Arm limp; the gripper still holds the roller. Press Enter to OPEN it (hold the roller!).")
            arm.open_gripper()
            arm.release(gripper=True)
        else:
            arm = Arm()
            input("The arm AND gripper will go LIMP. Support the arm and hold the roller, then press Enter.")
            arm.release(gripper=True)
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
