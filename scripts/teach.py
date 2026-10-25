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
MAX_STEP_DEG = 6      # a hand can't move a joint this far in 1/50 s; more = a bad reading
ARM_VLIM = 0.6       # rad/s per arm joint during replay (config default is 5.0)
GRIP_VLIM = 1.0
APPROACH_RAD_S = 0.3  # speed of the move from wherever the arm is to the stroke's start
GRIP_CAL = os.path.join(os.path.dirname(STROKES), "gripper.json")
GRIP_KP, GRIP_KD = 8.0, 1.0  # MIT stiffness and damping (SDK defaults); force = KP x squeeze
GRIP_SWEEP = 0.6      # rad/s while opening/closing (full travel ~7.2 rad = ~12 s)
GRIP_LEAD = 0.3      # contact needs the target at least this far ahead of the fingers (rad)...
GRIP_STILL = 0.05     # ...and the fingers moved less than this (rad)...
GRIP_STILL_S = 0.5    # ...over this long
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
        self.motors = [self.r._motor_map[n] for n in self.r.arm.joint_names] + [self.gm]
        self.ctrls = list(self.r._ctrl_map.values())
        self.grip = None        # gripper MIT target (rad) while gripping
        self.arm_on = False
        self.lock = threading.Lock()
        self.latest = None      # newest positions, written only by the bus thread
        self._alive = True
        self.target = None
        threading.Thread(target=self._loop, daemon=True).start()
        self.target = self.q().copy()  # trusted read (see q)

    @property
    def enabled(self):
        return self.arm_on

    def pos_now(self):
        """Latest positions from the bus thread (may be mid-motion), or None before any arrive."""
        with self.lock:
            return None if self.latest is None else self.latest.copy()

    def q(self):
        """Settled positions we can trust: wait until no motor reads an exact zero (nothing
        received yet - moving toward that makes a joint lunge) and two reads 60 ms apart agree."""
        last = None
        for _ in range(100):
            p = self.pos_now()
            if p is not None and not np.any(p == 0.0):
                if last is not None and np.allclose(p, last, atol=0.02):
                    return p
                last = p
            time.sleep(0.06)
        raise RuntimeError("no reliable joint feedback from the arm - check power and the USB cable")

    def _loop(self):
        """The only thread on the bus. Every 5 ms: send commands (each one gets a reply), ask
        idle motors for their position, then drain ALL replies. Reading on another thread
        while this one floods the bus with gripper commands left arm readings seconds stale."""
        while self._alive:
            with self.lock:
                t = None if self.target is None else self.target.copy()
                g, on = self.grip, self.arm_on
            try:
                if on and t is not None:
                    self.r.arm.send_pos_vel(t[:self.n])
                else:
                    for m in self.motors[:self.n]:
                        m.request_feedback()
                if g is not None:
                    self.gm.send_mit(float(g), 0.0, GRIP_KP, GRIP_KD, 0.0)
                else:
                    self.gm.request_feedback()
                for _ in range(4):
                    time.sleep(0.001)
                    for c in self.ctrls:
                        c.poll_feedback_once()
                states = [m.get_state() for m in self.motors]
                if all(st is not None for st in states):
                    with self.lock:
                        self.latest = np.array([st.pos for st in states])
            except Exception as e:
                print("[loop]", e)
            time.sleep(0.001)

    # ---- gripper ----
    def grip_pos(self):
        return float(self.pos_now()[self.n])

    def _grip_on(self):
        if self.grip is None:
            g, cal = self.grip_pos(), gripper_cal()
            lo, hi = sorted((cal["open"], cal["closed"]))
            if not lo - 0.3 <= g <= hi + 0.3:
                raise RuntimeError("gripper reads %.2f rad, outside its calibrated %.2f..%.2f - not enabling"
                                   % (g, lo, hi))
            with self.lock:
                self.grip = g
            self.gm.ensure_mode(Mode.MIT, 1000)
            self.gm.enable()
            time.sleep(0.1)

    def _sweep(self, to, stop_on_contact):
        """Move the grip target toward `to`; with stop_on_contact, stop where the fingers stall.

        Contact = the fingers have stopped moving (< GRIP_STILL rad in GRIP_STILL_S) while the
        target is well ahead of them. Not "target ahead by X": the fingers always trail a
        moving target, and that lag alone tripped the old check almost immediately.
        """
        self._grip_on()
        sign = 1.0 if to > self.grip else -1.0
        hist = []  # (time, actual position)
        while (to - self.grip) * sign > 0:
            with self.lock:
                self.grip += sign * GRIP_SWEEP * 0.02
            time.sleep(0.02)
            if not stop_on_contact:
                continue
            now, p = time.time(), self.grip_pos()
            hist.append((now, p))
            old = [h for h in hist if now - h[0] >= GRIP_STILL_S]
            if old and (self.grip - p) * sign > GRIP_LEAD and abs(p - old[-1][1]) < GRIP_STILL:
                with self.lock:
                    self.grip = p + sign * GRIP_SQUEEZE
                return p
        if stop_on_contact:  # let the fingers catch up with the end of the sweep
            time.sleep(1.0)
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
        p = self.pos_now()
        with self.lock:
            self.target = p

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
    input("\n2) The ARM IS LIMP: support it. Move it to a PARK pose OUT OF THE CAMERA'S VIEW,\n"
          "   where it can wait while the camera looks at the dough. Press Enter to save it.")
    park = arm.q()
    input("\n3) Move it to the start of the stroke (roller just above the near edge of the dough).\n"
          "   Press Enter to START recording.")
    rec, t0, stop = [], time.time(), threading.Event()

    def sampler():  # latest reads: the arm is moving, so settled reads (Arm.q) never come
        while not stop.is_set():
            rec.append((time.time() - t0, arm.pos_now()))
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
    np.savez(path, t=t, q=q, park=park)
    step = np.degrees(np.abs(np.diff(q[:, :arm.n], axis=0)).max())
    if step > MAX_STEP_DEG:
        print("\nWARNING: a joint jumped %.0f deg between two samples - stale readings. "
              "Record again before replaying." % step)
    span = np.degrees(q.max(0) - q.min(0))
    print("\nSaved %s: %d samples, %.1f s. Joint ranges (deg): %s" % (
        path, len(t), t[-1], " ".join("%.0f" % s for s in span)))
    input("\n4) Keep supporting the arm and press Enter to LOCK it in place.")
    arm.hold_here()
    input("   Arm locked. Rest it safely if you like, then press Enter: the arm will go LIMP.\n"
          "   SUPPORT IT FIRST.")
    arm.release()
    print("Arm released; the gripper still holds the roller.")
    input("Press Enter to OPEN the gripper (hold the roller!) and finish.")
    arm.open_gripper()
    arm.release(gripper=True)


def replay(name, speed, arm=None, confirm=True):
    """Move to the recorded start, play the stroke, then go to the park pose (out of the
    camera's view) if one was recorded, else back to the start. Returns the Arm."""
    d = np.load(os.path.join(STROKES, name + ".npz"))
    t, q = d["t"] / speed, d["q"]
    park = d["park"] if "park" in d.files else q[0]
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
    dist = np.abs(park[:arm.n] - q[-1][:arm.n]).max()
    print("Stroke done; parking...")
    arm.go(park, max(1.5, dist / APPROACH_RAD_S))
    print("Parked.")
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
