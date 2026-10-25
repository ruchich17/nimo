"""Taught backend: the real arm replays a stroke recorded with scripts/teach.py.

A quick-demo stand-in for full IK: every `roll` replays the one taught front-and-back
stroke (direction, pressure and length are ignored -- the arm only rolls front-and-back
anyway). Turning the dough and dusting flour stay with a person, as in human mode.
"""
import os
import sys

from robot.human_backend import HumanExecutor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import teach  # noqa: E402

STROKE = os.environ.get("ROTI_STROKE", "stroke")
SPEED = float(os.environ.get("ROTI_STROKE_SPEED", "0.5"))


class TaughtExecutor(HumanExecutor):
    def __init__(self):
        super().__init__()
        self.arm = teach.Arm()
        print("Gripping the roller (it must already be between the fingers)...")
        self.arm.grip_firmly()
        self.arm.hold_here()  # enabled from the start, holding wherever it is

    def roll(self, direction_deg, pressure, length_cm):
        self.on_instruction("Rolling. Keep your hands clear.")
        teach.replay(STROKE, SPEED, arm=self.arm, confirm=False)

    def stop(self):
        super().stop()
        self.arm.freeze()
