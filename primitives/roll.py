"""The `roll` primitive: turn Gemma's tool call into a Cartesian path for the roller.

This is the parameterised stroke CLAUDE.md describes. Taught trajectories from YAM
teleop will replace the waypoint maths here; the signature stays the same so the
decision loop doesn't care which is underneath.
"""

from __future__ import annotations

import numpy as np

# Above the board, where a stroke starts and ends. Kept low deliberately: holding
# the wrist vertical costs reach, so the far end of a stroke cannot lift much. See
# the measured envelope in robot/arm_scene.py. 0.08 puts the far lift-off out of
# range and every stroke then misses by ~17 deg.
APPROACH_HEIGHT = 0.030
PRESS_DEPTH = {  # how far into the dough, by pressure level
    "light": 0.004,
    "medium": 0.008,
    "firm": 0.013,
}
WAYPOINTS_PER_SEGMENT = 24


def roll_path(
    board_center: tuple[float, float, float],
    board_top_z: float,
    direction_deg: float,
    length_cm: float,
    pressure: str = "medium",
    offset_cm: float = 0.0,
) -> np.ndarray:
    """Cartesian waypoints for one stroke, as an (N, 3) array.

    `direction_deg` is measured in the board plane, 0° = +x. `offset_cm` shifts the
    stroke sideways off centre, which is how we thin one side of a lopsided roti.
    """
    if pressure not in PRESS_DEPTH:
        raise ValueError(f"pressure must be one of {sorted(PRESS_DEPTH)}, got {pressure!r}")

    theta = np.radians(direction_deg)
    along = np.array([np.cos(theta), np.sin(theta), 0.0])
    across = np.array([-np.sin(theta), np.cos(theta), 0.0])

    half = (length_cm / 100.0) / 2.0
    centre = np.array([board_center[0], board_center[1], board_top_z]) + across * (offset_cm / 100.0)

    contact_z = board_top_z - PRESS_DEPTH[pressure]
    start = centre - along * half
    end = centre + along * half

    above_start = start + [0, 0, APPROACH_HEIGHT]
    above_end = end + [0, 0, APPROACH_HEIGHT]
    press_start = np.array([start[0], start[1], contact_z])
    press_end = np.array([end[0], end[1], contact_z])

    return np.vstack(
        [
            _lerp(above_start, press_start),  # descend
            _lerp(press_start, press_end),  # the stroke itself
            _lerp(press_end, above_end),  # lift off
        ]
    )


def _lerp(a: np.ndarray, b: np.ndarray, n: int = WAYPOINTS_PER_SEGMENT) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n).reshape(-1, 1)
    return a + (b - a) * t
