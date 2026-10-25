"""Safety wrapper: the only path from a tool call to the executor.

Rejects unknown tools, clamps every parameter to its range, rate-limits commands, caps the
number of strokes, and latches a stop that blocks everything until reset().
"""
import math
import threading
import time

LIMITS = {
    "direction_deg": (0.0, 180.0),
    "pressure": (0.1, 0.8),
    "length_cm": (3.0, 15.0),
    "deg": (-90.0, 90.0),
}
# The arm faces the rig from the top of the camera image and holds a paint roller tilted,
# so it can only roll toward/away from itself: image directions 60-120 deg (up-down +-30).
# Other directions need the dough turned (rotate_dough). Rejected, not clamped, so the
# brain hears why and picks rotate_dough instead of a silently different stroke.
ARM_ROLL_RANGE = (60.0, 120.0)
MIN_INTERVAL_S = 1.0
MAX_MOTIONS = 40


def clamp(key, value):
    lo, hi = LIMITS[key]
    v = float(value)
    if not math.isfinite(v):
        raise ValueError("%s is not finite" % key)
    if key == "direction_deg":
        v %= 180.0  # a stroke at 200 deg is the same axis as 20 deg
    return min(max(v, lo), hi)


class SafetyWrapper:
    def __init__(self, executor):
        self.executor = executor
        self.stopped = threading.Event()
        self.motions = 0
        self.last_t = 0.0

    def stop(self, reason="stop"):
        if not self.stopped.is_set():
            print("\n!!! STOP: %s" % reason)
        self.stopped.set()
        self.executor.stop()

    def reset(self):
        self.stopped.clear()

    def execute(self, name, args):
        """Run one motion tool. Returns a short status string for logs/history."""
        if self.stopped.is_set():
            return "blocked (stopped)"
        if self.motions >= MAX_MOTIONS:
            self.stop("motion cap %d reached" % MAX_MOTIONS)
            return "blocked (motion cap)"
        try:  # clamp now, so bad args are rejected before anything moves
            if name == "roll":
                p = (clamp("direction_deg", args.get("direction_deg", 0)),
                     clamp("pressure", args.get("pressure", 0.3)),
                     clamp("length_cm", args.get("length_cm", 8)))
                if not ARM_ROLL_RANGE[0] <= p[0] <= ARM_ROLL_RANGE[1]:
                    return ("rejected (arm can only roll %d-%d deg; turn the dough with rotate_dough)"
                            % ARM_ROLL_RANGE)
                call = lambda: self.executor.roll(*p)
            elif name == "rotate_dough":
                deg = clamp("deg", args.get("deg", 0))
                call = lambda: self.executor.rotate_dough(deg)
            elif name == "dust_flour":
                call = self.executor.dust_flour
            else:
                return "rejected (not a motion tool: %s)" % name
        except (TypeError, ValueError) as e:
            return "rejected (bad args %s: %s)" % (args, e)

        wait = MIN_INTERVAL_S - (time.time() - self.last_t)
        if wait > 0 and self.stopped.wait(wait):
            return "blocked (stopped)"
        self.motions += 1
        call()
        self.last_t = time.time()
        return "ok"
