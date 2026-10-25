"""Damped-least-squares inverse kinematics for a site on the YAM arm.

Deliberately dependency-free (mink is available via i2rt, but this keeps the sim
runnable even when i2rt isn't installed — notably on Windows, where its ruckig
dependency needs a build constraint to compile).

Solves position and orientation together. Orientation is not optional in practice:
with position-only IK the arm reaches the right point with the roller spun to an
arbitrary angle and the wrist driven through the dough.
"""

from __future__ import annotations

import mujoco
import numpy as np

DAMPING = 1e-3
MAX_ITERS = 600  # 300 plateaus ~15 deg short on the first waypoint of a stroke
POS_TOLERANCE = 1e-4  # metres
ROT_TOLERANCE = 1e-3  # radians
MAX_STEP = 0.15  # radians per iteration, keeps the solver from leaping
ROT_WEIGHT = 0.5  # relative pull of orientation vs position


def solve_site_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_name: str,
    target_pos: np.ndarray,
    target_mat: np.ndarray | None = None,
    q_init: np.ndarray | None = None,
    dofs: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float]:
    """Find joint angles putting `site_name` at the given pose.

    `target_mat` is a 3x3 rotation whose columns are the desired site axes; pass
    None for position-only. `dofs` restricts which velocity indices the solver may
    move — pass the arm joints so it cannot "reach" by opening the gripper.

    Returns (joint angles, position error in metres, orientation error in radians).
    The solver returns its best effort rather than raising, because a near-miss is
    still worth seeing in the viewer.
    """
    nq = model.nq
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise ValueError(f"no site named {site_name!r}")

    q = np.array(data.qpos[:nq] if q_init is None else q_init, dtype=float)
    lo, hi = model.jnt_range[:, 0], model.jnt_range[:, 1]

    free = np.arange(model.nv) if dofs is None else np.asarray(dofs, dtype=int)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    target = np.asarray(target_pos, dtype=float)

    want_rot = target_mat is not None
    if want_rot:
        quat_des = np.zeros(4)
        mujoco.mju_mat2Quat(quat_des, np.asarray(target_mat, dtype=float).flatten())
        quat_cur = np.zeros(4)
        quat_neg = np.zeros(4)
        quat_err = np.zeros(4)
        omega = np.zeros(3)

    pos_err = rot_err = np.inf
    for _ in range(MAX_ITERS):
        data.qpos[:nq] = q
        # Kinematics + COM only. mj_forward would also run collision detection, which
        # this arm's convex-decomposition geometry makes ruinously slow in an IK loop
        # (8-12 collision parts per link). mj_jacSite needs subtree_com from mj_comPos.
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)

        site = data.site(site_name)
        dpos = target - site.xpos
        pos_err = float(np.linalg.norm(dpos))

        if want_rot:
            mujoco.mju_mat2Quat(quat_cur, site.xmat.flatten())
            mujoco.mju_negQuat(quat_neg, quat_cur)
            mujoco.mju_mulQuat(quat_err, quat_des, quat_neg)
            mujoco.mju_quat2Vel(omega, quat_err, 1.0)
            rot_err = float(np.linalg.norm(omega))
            if pos_err < POS_TOLERANCE and rot_err < ROT_TOLERANCE:
                break
            err = np.concatenate([dpos, ROT_WEIGHT * omega])
        else:
            rot_err = 0.0
            if pos_err < POS_TOLERANCE:
                break
            err = dpos

        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        jac = np.vstack([jacp, ROT_WEIGHT * jacr]) if want_rot else jacp
        jac = jac[:, free]

        # DLS: dq = Jᵀ(JJᵀ + λ²I)⁻¹ dx — stays well behaved through singularities.
        jjt = jac @ jac.T + DAMPING * np.eye(jac.shape[0])
        dq_free = jac.T @ np.linalg.solve(jjt, err)

        norm = np.linalg.norm(dq_free)
        if norm > MAX_STEP:
            dq_free *= MAX_STEP / norm

        dq = np.zeros(model.nv)
        dq[free] = dq_free
        q = np.clip(q + dq, lo, hi)

    return q, pos_err, rot_err


def stroke_frame(direction_deg: float) -> np.ndarray:
    """Desired roller orientation for a stroke, as a 3x3 rotation.

    Columns are the mount's axes in world: +x straight down, +y the roller's spin
    axis (horizontal, across the stroke), +z the stroke direction.

    Aiming +x down is the whole trick. On the reBot, joint6 rolls about the
    gripper's local +x and is the only wrist joint with a full +-180 deg range.
    Pointing that axis at the board makes joint6 the stroke-heading knob, so every
    heading is reachable. Aim any other axis down and the heading has to come out
    of joints 1-5, which run out of travel — strokes at 0 and 90 deg then miss by
    25 cm and 85 mm respectively.
    """
    theta = np.radians(direction_deg)
    along = np.array([np.cos(theta), np.sin(theta), 0.0])
    across = np.array([-np.sin(theta), np.cos(theta), 0.0])
    down = np.array([0.0, 0.0, -1.0])
    return np.column_stack([down, across, along])


def solve_stroke_path(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_name: str,
    path: np.ndarray,
    direction_deg: float,
    q_init: np.ndarray | None = None,
    dofs: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float]:
    """IK a whole stroke into joint targets, picking the reachable roller heading.

    Rolling along θ and along θ+180° is the same physical stroke, and joint6's
    ±120° limit means only one of the two is usually reachable. Solve both and
    keep whichever tracks the roller orientation better.

    Returns (targets, worst position error, worst orientation error).
    """
    best: tuple[np.ndarray, float, float] | None = None
    for heading in (direction_deg, direction_deg + 180.0):
        target_mat = stroke_frame(heading)
        targets = np.zeros((len(path), model.nq))
        q = data.qpos[: model.nq].copy() if q_init is None else np.array(q_init, dtype=float)
        worst_pos = worst_rot = 0.0
        for i, point in enumerate(path):
            q, pos_err, rot_err = solve_site_pose(
                model, data, site_name, point, target_mat, q_init=q, dofs=dofs
            )
            targets[i] = q
            worst_pos = max(worst_pos, pos_err)
            worst_rot = max(worst_rot, rot_err)
        if best is None or worst_rot < best[2]:
            best = (targets, worst_pos, worst_rot)
    assert best is not None
    return best
