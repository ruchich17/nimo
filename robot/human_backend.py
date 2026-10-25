"""Human backend: a person stands in for the arm.

Each motion is shown and spoken as an instruction; the call blocks until the person
says they've done it (done()), or until stop(). Same interface as the sim and real
backends, so it sits behind the same SafetyWrapper and the same arm limits apply --
practising with it is practising exactly what the arm will do.
"""
import threading

from robot.executor import Executor


def pressure_words(p):
    return "gently" if p < 0.35 else "with medium pressure" if p < 0.6 else "firmly"


class HumanExecutor(Executor):
    def __init__(self):
        self._done = threading.Event()
        self.instruction = None  # (text, tool name, args) while waiting for the person
        self.on_instruction = lambda text: None

    def _ask(self, text, name, args):
        self._done.clear()
        self.instruction = (text, name, args)
        self.on_instruction(text)
        self._done.wait()
        self.instruction = None

    def roll(self, direction_deg, pressure, length_cm):
        self._ask("Roll %s along the arrow, about %.0f cm, once each way." % (pressure_words(pressure), length_cm),
                  "roll", {"direction_deg": direction_deg})

    def rotate_dough(self, deg):
        self._ask("Turn the dough %d degrees %s." % (abs(round(deg)), "counter-clockwise" if deg >= 0 else "clockwise"),
                  "rotate_dough", {"deg": deg})

    def dust_flour(self):
        self._ask("Dust a little flour on the dough.", "dust_flour", {})

    def done(self):
        """The person pressed SPACE: their step is finished."""
        self._done.set()

    def stop(self):
        self._done.set()
