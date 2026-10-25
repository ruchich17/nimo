"""Builds the MuJoCo scene for Plan A: a reBot B601 arm holding a roller over a board.

The arm model is `seeed_rebot_devarm` from mujoco_menagerie (MIT), vendored under
third_party/. It is left untouched — everything we add goes on with MjSpec.

Unlike the YAM model this replaced, the menagerie model already ships position
actuators with real motor torque limits, coupled gripper fingers, convex-decomposition
collision geometry and `home`/`raised` keyframes. So we only add the rig:
  * a roller on the `gripper_end` body
  * a board, a dough marker, an overhead camera and a light

NOTE ON VARIANTS: the menagerie model is the B601-RS, built around RobStride RS-06/RS-00
actuators. Our hardware is the B601-DM (Damiao motors). The kinematics are the same arm
— joint limits agree with Seeed's published DM figures to within a few degrees on joint4
— but the torque ratings do not. That only matters once we care about dynamics; Plan A
drives the arm kinematically.

Tuning knobs are the module constants below — all lengths in metres, angles in radians.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
ARM_XML = REPO_ROOT / "third_party/menagerie/seeed_rebot_devarm/seeed_rebot_devarm.xml"

ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
FINGER_JOINTS = ["joint_left", "joint_right"]

# Board: where the dough sits, in front of the arm base (which stands at the origin).
#
# 0.30 m is not arbitrary. Holding the wrist vertical (see robot.ik.stroke_frame)
# costs reach, and the ceiling drops the further out you go — measured, heading 0:
#     x <= 0.34  ->  reaches z = 0.08
#     x  = 0.37  ->  reaches z = 0.06
#     x  = 0.40  ->  reaches z = 0.04
#     x  = 0.43  ->  nothing
# A 14 cm stroke centred at 0.30 spans 0.23..0.37, so the far lift-off must stay
# under z = 0.06 — hence APPROACH_HEIGHT in primitives/roll.py is 0.03, not 0.08.
# Heading 0 is the worst case; other headings swing less far out.
# Re-measure this against the real table before trusting it.
BOARD_CENTER = (0.30, 0.0, 0.010)
BOARD_HALF = (0.15, 0.15, 0.010)

# Dough marker. Visual only (contype/conaffinity 0) — in Plan A the real dough lives
# under the real camera and this disc just shows where the arm thinks it is.
DOUGH_RADIUS = 0.060
DOUGH_HALF_HEIGHT = 0.004

# The roller tool. Physically this is a T: a stub handle clamped vertically in the
# parallel jaws, with the roller on a crossbar below it. Build it that way and the
# sim matches. ROLLER_DROP is how far the roller sits below the jaws — enough that
# the gripper body clears the dough.
ROLLER_RADIUS = 0.025
ROLLER_HALF_LENGTH = 0.070
ROLLER_DROP = 0.060
HANDLE_RADIUS = 0.008

# Jaw opening to clamp the handle, per finger.
GRIP_CLOSED = 0.010

CAMERA_HEIGHT = 0.70

OFFSCREEN_SIZE = (1280, 960)


def build_spec() -> mujoco.MjSpec:
    """Load the vendored arm and add the roti rig around it."""
    spec = mujoco.MjSpec.from_file(str(ARM_XML))
    spec.modelname = "rebot_roti"

    # Default offscreen buffer is 640x480; we render sim frames for telemetry.
    spec.visual.global_.offwidth = OFFSCREEN_SIZE[0]
    spec.visual.global_.offheight = OFFSCREEN_SIZE[1]

    _add_world(spec)
    _add_roller(spec)
    return spec


def build() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Compile the scene and return a ready model/data pair, gripper closed."""
    model = build_spec().compile()
    data = mujoco.MjData(model)
    for name in FINGER_JOINTS:
        data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]] = GRIP_CLOSED
    mujoco.mj_forward(model, data)
    return model, data


def arm_dofs(model: mujoco.MjModel) -> np.ndarray:
    """Velocity indices of the six arm joints — what the IK is allowed to move.

    Without this the solver happily "reaches" by sliding the gripper fingers.
    """
    return np.array(
        [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in ARM_JOINTS]
    )


def arm_qpos_adr(model: mujoco.MjModel) -> np.ndarray:
    """qpos indices of the six arm joints, in order."""
    return np.array(
        [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in ARM_JOINTS]
    )


def _add_world(spec: mujoco.MjSpec) -> None:
    world = spec.worldbody

    world.add_light(pos=[0, 0, 1.5], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])

    world.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[2.0, 2.0, 0.05],
        rgba=[0.25, 0.25, 0.28, 1.0],
    )

    board = world.add_body(name="board", pos=list(BOARD_CENTER))
    board.add_geom(
        name="board_top",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=list(BOARD_HALF),
        rgba=[0.12, 0.10, 0.09, 1.0],  # dark: maximum contrast against pale dough
    )

    # Sits on top of the board, hence the +z of both half-heights.
    dough_z = BOARD_CENTER[2] + BOARD_HALF[2] + DOUGH_HALF_HEIGHT
    dough = world.add_body(name="dough", pos=[BOARD_CENTER[0], BOARD_CENTER[1], dough_z])
    dough.add_geom(
        name="dough_disc",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[DOUGH_RADIUS, DOUGH_HALF_HEIGHT, 0.0],
        rgba=[0.93, 0.87, 0.73, 1.0],
        contype=0,
        conaffinity=0,
    )

    # Mirrors the real overhead webcam, so sim and real framing match.
    world.add_camera(
        name="overhead",
        pos=[BOARD_CENTER[0], BOARD_CENTER[1], CAMERA_HEIGHT],
        quat=[0.0, 1.0, 0.0, 0.0],
    )


def _add_roller(spec: mujoco.MjSpec) -> None:
    """Hang the roller off the gripper in a frame we can aim the IK at.

    The mount's frame is what `robot.ik.stroke_frame` targets:
      +x = down (the joint6 roll axis), +y = roller spin axis (across the stroke),
      +z = stroke direction.

    The stub therefore runs along +x and the roller's axis along +y. See
    `stroke_frame` for why +x is the one that must point at the board.
    """
    mount = spec.body("gripper_end").add_body(name="roller_mount")

    # Stub handle: what the jaws actually clamp, running down from the gripper.
    mount.add_geom(
        name="roller_handle",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=[HANDLE_RADIUS, ROLLER_DROP / 2.0, 0.0],
        pos=[ROLLER_DROP / 2.0, 0.0, 0.0],
        zaxis=[1.0, 0.0, 0.0],
        rgba=[0.45, 0.45, 0.48, 1.0],
        contype=0,
        conaffinity=0,
    )

    mount.add_geom(
        name="roller",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[ROLLER_RADIUS, ROLLER_HALF_LENGTH, 0.0],
        pos=[ROLLER_DROP, 0.0, 0.0],
        zaxis=[0.0, 1.0, 0.0],  # spin axis along the mount's +y, i.e. across the stroke
        rgba=[0.80, 0.60, 0.35, 1.0],
        contype=0,
        conaffinity=0,
    )

    mount.add_site(name="grasp_site")
    # Directly under the roller's axis — the point that actually touches the dough.
    # The mount's +x points down, so the contact is along +x.
    mount.add_site(name="roller_contact", pos=[ROLLER_DROP + ROLLER_RADIUS, 0.0, 0.0])


if __name__ == "__main__":
    model, data = build()
    print(f"compiled  nq={model.nq} nv={model.nv} nu={model.nu} ncam={model.ncam}")
    print(f"arm dofs  {arm_dofs(model)}")
    print(f"grasp_site at {data.site('grasp_site').xpos}")
    print(f"roller_contact at {data.site('roller_contact').xpos}")
