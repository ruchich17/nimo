"""Primitive executor interface. Backends: stub (prints), sim (MuJoCo reBot B601), real (Seeed B601 SDK, not written).

Sim <-> real is the --executor flag. Never call a backend directly from model output:
always go through robot.safety.SafetyWrapper.
"""
import threading


class Executor:
    def roll(self, direction_deg, pressure, length_cm): raise NotImplementedError
    def rotate_dough(self, deg): raise NotImplementedError
    def dust_flour(self): raise NotImplementedError
    def home(self): pass
    def stop(self): pass


class StubExecutor(Executor):
    """Prints the stroke and pretends to take `stroke_s` seconds; stop() interrupts it."""

    def __init__(self, stroke_s=1.5):
        self.stroke_s = stroke_s
        self._stop = threading.Event()

    def _move(self, what):
        print("    [stub] %s" % what)
        if self._stop.wait(self.stroke_s):
            print("    [stub] interrupted")

    def roll(self, direction_deg, pressure, length_cm):
        self._move("roll %5.1f deg  pressure %.2f  length %.1f cm" % (direction_deg, pressure, length_cm))

    def rotate_dough(self, deg):
        self._move("rotate dough %+.0f deg" % deg)

    def dust_flour(self):
        self._move("dust flour")

    def stop(self):
        self._stop.set()


def make_executor(name):
    if name == "stub":
        return StubExecutor()
    if name == "taught":
        from robot.taught_backend import TaughtExecutor  # real arm replays the taught stroke
        return TaughtExecutor()
    if name == "human":
        from robot.human_backend import HumanExecutor  # a person stands in for the arm
        return HumanExecutor()
    if name == "sim":
        from robot.sim_backend import SimExecutor  # Plan A
        return SimExecutor()
    if name == "real":
        from robot.real_backend import RealExecutor  # Plan B, Linux relay
        return RealExecutor()
    raise ValueError("unknown executor: " + name)
