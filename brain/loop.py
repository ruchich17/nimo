"""Decision loop: frame -> Gemma -> speak plan -> safety wrapper -> executor, until done.

Keyboard (type + Enter, works while Gemma is thinking):
  s / stop / q   hard stop (Ctrl-C also stops)
  m              manual mode: Gemma pauses, only typed commands run
  a              back to auto (Gemma decides)
  roll 45 0.5 10 | rotate 30 | flour | inspect | done   typed command, runs next
"""
import queue
import sys
import threading
import time

from brain import gemma

STOP_WORDS = {"s", "stop", "q", "quit", "x"}


def parse_command(line):
    """Typed command -> (tool_name, args), or None if unrecognised."""
    p = line.split()
    if not p:
        return None
    try:
        if p[0] == "roll":
            nums = [float(x) for x in p[1:4]]  # pressure and length are optional
            nums += [0.4, 8.0][len(nums) - 1:]
            return "roll", {"direction_deg": nums[0], "pressure": nums[1], "length_cm": nums[2]}
        if p[0] in ("rotate", "rotate_dough"):
            return "rotate_dough", {"deg": float(p[1])}
        if p[0] in ("flour", "dust", "dust_flour"):
            return "dust_flour", {}
        if p[0] in ("inspect", "look"):
            return "inspect", {}
        if p[0] == "done":
            return "done", {"reason": "typed"}
    except (IndexError, ValueError):
        pass
    return None


class Console:
    """Reads stdin on a thread so stop works even mid-Gemma-call."""

    def __init__(self, robot):
        self.robot = robot
        self.commands = queue.Queue()
        self.manual = False
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in sys.stdin:
            line = line.strip().lower()
            if line in STOP_WORDS:
                self.robot.stop("stop key")
            elif line == "m":
                self.manual = True
                print("--- manual mode: type commands, 'a' for auto")
            elif line == "a":
                self.manual = False
                print("--- auto mode: Gemma decides")
            elif line:
                cmd = parse_command(line)
                if cmd:
                    self.commands.put(cmd)
                else:
                    print("??? unknown command: %s" % line)

    def next_command(self):
        """A typed command if one is queued; in manual mode, wait for one."""
        while True:
            try:
                return self.commands.get(timeout=0.2 if self.manual else 0)
            except queue.Empty:
                if not self.manual or self.robot.stopped.is_set():
                    return None


def run(source, robot, voice, max_steps=25):
    console = Console(robot)
    history = []
    print(__doc__)
    t_start = time.time()
    try:
        for step in range(1, max_steps + 1):
            if robot.stopped.is_set():
                break
            cmd = console.next_command()
            if robot.stopped.is_set():
                break
            frame = source.read()
            if cmd:
                name, args = cmd
                speech, latency, who = gemma.describe(name, args), 0.0, "typed"
            else:
                speech, name, args, latency = gemma.decide(frame, history, empty=getattr(source, "empty", None),
                                                               board_cm=getattr(source, "board_cm", None))
                who = "gemma"
            if robot.stopped.is_set():  # stop pressed while Gemma was thinking
                break
            if who == "gemma" and console.manual:  # 'm' pressed while Gemma was thinking
                print("--- discarded Gemma's %s (manual mode)" % name)
                continue

            print("\n[step %2d | %s %4.1f s] %s" % (step, who, latency, speech))
            print("    -> %s %s" % (name, args))
            voice.say(speech)

            if name == "done":
                print("=== DONE after %d steps, %.0f s: %s" % (step, time.time() - t_start, args.get("reason", "")))
                return
            if name == "stop":
                robot.stop("model: %s" % args.get("reason", ""))
                break
            if name == "inspect":
                history.append("inspect")
                continue
            status = robot.execute(name, args)
            print("    safety: %s" % status)
            history.append("%s %s (%s)" % (name, args, status))
        else:
            print("=== step limit %d reached" % max_steps)
    except KeyboardInterrupt:
        robot.stop("Ctrl-C")
    finally:
        if not robot.stopped.is_set():
            robot.executor.home()
