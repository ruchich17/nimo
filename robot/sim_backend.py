"""Sim backend (Plan A): Gemma's tool calls drive the reBot arm in MuJoCo.

roll() -> primitives.roll.roll_path -> robot.ik.solve_stroke_path -> joint targets
streamed to the position actuators, one waypoint every STEP_S. A physics thread keeps
stepping and syncing the viewer between commands.

The viewer needs mjpython on macOS:  mjpython scripts/run_sim.py --executor sim
SIM_HEADLESS=1 skips the viewer (plain python is then fine, e.g. for tests).
"""
import os
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np

from primitives.roll import roll_path
from robot.arm_scene import BOARD_CENTER, BOARD_HALF, arm_dofs, build
from robot.executor import Executor
from robot.ik import solve_stroke_path

TOOL_SITE = "roller_contact"
STEP_S = 0.04  # seconds per waypoint; a 72-waypoint stroke takes ~3 s
IK_WARN_M = 0.01
HUMAN_TURN_S = 6  # time given to a person to turn the dough when asked

# Gemma gives angles in the camera image (0 = left-right, 90 = up-down, counter-
# clockwise). The sim measures on the board (0 = +x, straight out from the arm base).
# The arm faces the camera rig from the opposite side, so its base is at the TOP of the
# overhead image and +x points image-down. Strokes are axes (0 == 180), so base-at-top and
# base-at-bottom give the same mapping; a mirrored camera image would flip its sign.
# Check against the real camera before Plan B -- one of the measure-on-hardware facts.
IMAGE_TO_BOARD_DEG = -90.0


def board_deg(image_deg):
    return (image_deg + IMAGE_TO_BOARD_DEG) % 180.0


def pressure_level(p):
    """Map Gemma's 0.1-0.8 pressure onto roll_path's three press depths."""
    return "light" if p < 0.35 else "medium" if p < 0.6 else "firm"


class SimExecutor(Executor):
    def __init__(self, headless=None):
        self.model, self.data = build()
        self.plan_data = mujoco.MjData(self.model)  # IK scratch: never touch the live state
        self.dofs = arm_dofs(self.model)
        self.board_top = BOARD_CENTER[2] + BOARD_HALF[2]
        self.data.ctrl[:] = self.data.qpos[: self.model.nu]
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._alive = True

        headless = os.environ.get("SIM_HEADLESS") == "1" if headless is None else headless
        self.viewer = None
        if not headless:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        threading.Thread(target=self._physics, daemon=True).start()

    def _physics(self):
        dt = self.model.opt.timestep
        while self._alive and (self.viewer is None or self.viewer.is_running()):
            t0 = time.time()
            with self._lock:
                mujoco.mj_step(self.model, self.data)
            if self.viewer is not None:
                self.viewer.sync()
            time.sleep(max(0.0, dt - (time.time() - t0)))

    def plan(self, direction_deg, pressure, length_cm):
        """Joint targets for one stroke. Returns (targets, board heading, worst pos err m)."""
        heading = board_deg(direction_deg)
        path = roll_path(BOARD_CENTER, self.board_top, heading, length_cm, pressure_level(pressure))
        with self._lock:
            self.plan_data.qpos[:] = self.data.qpos
        targets, pos_err, _ = solve_stroke_path(
            self.model, self.plan_data, TOOL_SITE, path, heading, dofs=self.dofs)
        return targets, heading, pos_err

    def roll(self, direction_deg, pressure, length_cm):
        self._stop.clear()
        targets, heading, pos_err = self.plan(direction_deg, pressure, length_cm)
        print("    [sim] roll image %5.1f deg -> board %5.1f deg, %s, %.1f cm, IK err %.1f mm"
              % (direction_deg, heading, pressure_level(pressure), length_cm, pos_err * 1000))
        if pos_err > IK_WARN_M:
            print("    [sim] WARNING: IK missed by > 1 cm, stroke may be outside the workspace")
        for q in targets:
            if self._stop.is_set():
                print("    [sim] interrupted")
                return
            with self._lock:
                self.data.ctrl[:] = q[: self.model.nu]
            time.sleep(STEP_S)

    def rotate_dough(self, deg):
        """A person turns the dough (the voice has asked them); give them time to."""
        print("    [sim] waiting %d s for a person to turn the dough %+.0f deg" % (HUMAN_TURN_S, deg))
        self._stop.clear()
        self._stop.wait(HUMAN_TURN_S)

    # No dough physics in the sim: this just shows up in the log.

    def dust_flour(self):
        print("    [sim] dust flour (not simulated)")

    def stop(self):
        self._stop.set()
