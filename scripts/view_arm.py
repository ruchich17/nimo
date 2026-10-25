"""Open the MuJoCo viewer on the roti scene, optionally playing a rolling stroke.

    python scripts/view_arm.py                    # just look at the rig
    python scripts/view_arm.py --stroke 45        # roll at 45 degrees, on a loop
    python scripts/view_arm.py --stroke 0 --pressure firm --length-cm 16

Press the usual MuJoCo viewer keys; Ctrl-C in the terminal to quit.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from primitives.roll import roll_path  # noqa: E402
from robot.arm_scene import BOARD_CENTER, BOARD_HALF, arm_dofs, build  # noqa: E402
from robot.ik import solve_stroke_path  # noqa: E402

TOOL_SITE = "roller_contact"
STEP_SECONDS = 0.04


def plan_stroke(model, data, args) -> tuple[np.ndarray, float, float]:
    """IK the stroke waypoints into joint targets.

    Returns (targets, worst position error, worst orientation error).
    """
    board_top_z = BOARD_CENTER[2] + BOARD_HALF[2]
    path = roll_path(
        board_center=BOARD_CENTER,
        board_top_z=board_top_z,
        direction_deg=args.stroke,
        length_cm=args.length_cm,
        pressure=args.pressure,
        offset_cm=args.offset_cm,
    )
    return solve_stroke_path(model, data, TOOL_SITE, path, args.stroke, dofs=arm_dofs(model))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stroke", type=float, default=None, help="stroke direction in degrees (0 = +x)")
    parser.add_argument("--length-cm", type=float, default=14.0)
    parser.add_argument("--pressure", choices=["light", "medium", "firm"], default="medium")
    parser.add_argument("--offset-cm", type=float, default=0.0)
    args = parser.parse_args()

    model, data = build()
    mujoco.mj_forward(model, data)
    print(f"scene: nq={model.nq} nu={model.nu} | board top z={BOARD_CENTER[2] + BOARD_HALF[2]:.3f}")

    targets = None
    if args.stroke is not None:
        targets, worst_pos, worst_rot = plan_stroke(model, data, args)
        print(f"stroke {args.stroke:.0f} deg, {args.length_cm:.0f} cm, {args.pressure}")
        print(f"  {len(targets)} waypoints, worst IK error {worst_pos * 1000:.1f} mm / {np.degrees(worst_rot):.1f} deg")
        if worst_pos > 0.01:
            print("  WARNING: IK missed by more than 1 cm — stroke may be outside the workspace")
        if worst_rot > np.radians(10):
            print("  WARNING: roller is more than 10 deg off square to the stroke")
        mujoco.mj_resetData(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        next_advance = time.time()
        while viewer.is_running():
            if targets is not None and time.time() >= next_advance:
                data.ctrl[:] = targets[step % len(targets)]
                step += 1
                next_advance = time.time() + STEP_SECONDS

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
