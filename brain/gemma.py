"""Gemma client: one overhead image in, one tool call out. Local Ollama only.

OLLAMA_HOST_URL / GEMMA_MODEL env vars override the defaults.
"""
import base64
import io
import json
import os
import time
import urllib.request

import cv2
import numpy as np

from brain.tools import TOOLS
from robot.safety import ARM_ROLL_RANGE
from vision.measure import measure

OLLAMA_URL = os.environ.get("OLLAMA_HOST_URL", "http://localhost:11434") + "/api/chat"
MODEL = os.environ.get("GEMMA_MODEL", "gemma4:e4b")
# Thinking costs ~180 tokens (~3-7 s) per decision for the same answer on test images.
# GEMMA_THINK=1 turns it back on if real photos need it.
THINK = os.environ.get("GEMMA_THINK") == "1"
# From the labelled rig photos: finished rotis measure 0.37-0.41 coverage, unfinished <= 0.34.
DONE_COVERAGE = 0.36
ROUND_ASPECT = 1.15  # rolled-round discs measure 1.13-1.14; lopsided 3 (an oval) 1.18
DECIDE_SIDE = 512  # what the main decision sees; sources deliver more for the torn close-up
THICK_COVERAGE = 0.30

SYSTEM = (
    "You control a robot arm that rolls Indian roti dough into a thin, round circle with a rolling pin. "
    "You see the dough on a board from directly overhead.\n"
    "Each turn you get the image plus a camera report on the dough's size and shape, "
    "computed from its outline. Trust the report for size, shape and angles; use the "
    "image to spot hands, tears or sticky dough.\n"
    "Choose:\n"
    "- done: the report says the size is finished AND the shape is round.\n"
    "- rotate_dough: when the report says to turn the dough (the arm only rolls up-down). "
    "A person turns it; say so.\n"
    "- roll: otherwise, at the angle the report suggests. Pressure 0.7 if the report "
    "says the dough is thick, else 0.5. Length as the report says, else 12.\n"
    "- stop: no dough visible, a hand is over the board, or the dough is TORN, CRACKED "
    "or in pieces. Torn dough cannot be fixed by rolling: stop and say why.\n"
    "Say ONE short sentence for the audience: what you see, then the action and its "
    "angle. Use your own words for this dough. Then call exactly one tool."
)


def camera_report(m, history=(), board_cm=None):
    """Turn outline measurements into plain facts, so Gemma never compares numbers."""
    if m is None:
        return " Camera report: dough not found - check the image yourself."
    if m["coverage"] >= DONE_COVERAGE:
        size = "big enough to be finished"
    elif m["coverage"] < THICK_COVERAGE:
        size = "small and thick, keep rolling"
    else:
        size = "still too small, keep rolling"
    if m["aspect"] >= ROUND_ASPECT:
        across = round(m["narrow_deg"])
        if ARM_ROLL_RANGE[0] <= across <= ARM_ROLL_RANGE[1]:
            shape = ("lopsided: %.1f times longer one way, narrowest at %d degrees, so roll "
                     "at %d degrees to spread it there" % (m["aspect"], across, across))
        else:
            turn = (90 - across + 90) % 180 - 90  # -90..90, counter-clockwise positive
            shape = ("lopsided: narrowest at %d degrees, which the arm cannot reach. Ask for the "
                     "dough to be turned: call rotate_dough with deg %d" % (across, turn))
    elif m["coverage"] >= DONE_COVERAGE:
        return " Camera report: size big enough to be finished. Shape round. The roti is DONE: call done."
    else:
        across = _fresh_angle(history)
        shape = "round; to spread it evenly, roll at %d degrees" % across
    report = " Camera report: size %s. Shape %s." % (size, shape)
    if board_cm and m["coverage"] < DONE_COVERAGE:
        across_cm = m["coverage"] * board_cm
        report += " The dough is about %.0f cm across; use length_cm %.0f." % (
            across_cm, min(15, max(6, across_cm + 4)))
    return report


def _fresh_angle(history):
    """A reachable stroke direction, as far as possible from the last two rolls in `history`."""
    used = []
    for h in history[-2:]:
        if h.startswith("roll"):
            try:
                used.append(float(h.split("direction_deg': ")[1].split(",")[0].rstrip("}")))
            except (IndexError, ValueError):
                pass
    for cand in (90, 60, 120):
        if all(min(abs(cand - u) % 180, 180 - abs(cand - u) % 180) >= 30 for u in used):
            return cand
    return 90


def encode(img):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def describe(name, args):
    """Fallback speech when Gemma gives no text."""
    if name == "roll":
        return "Rolling at %d degrees." % round(float(args.get("direction_deg", 0)))
    if name == "rotate_dough":
        deg = round(float(args.get("deg", 0)))
        return "Please turn the dough %d degrees %s." % (abs(deg), "counter-clockwise" if deg >= 0 else "clockwise")
    return {"rotate_dough": "Rotating the dough.", "dust_flour": "Dusting flour.",
            "inspect": "Taking another look.", "done": "Done. It's round.",
            "stop": "Stopping."}.get(name, name)


def _chat(body):
    req = urllib.request.Request(OLLAMA_URL, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r).get("message", {})


TORN_Q = ("Look closely at the dough on the wooden board. Is the dough TORN, CRACKED, BROKEN or in "
          "SEPARATE PIECES? A lumpy, wrinkled or lopsided dough that is one intact piece is NOT torn. "
          "Answer with exactly one word: yes or no.")


def is_torn(img, contour, model=MODEL):
    """Gemma's yes/no on a close-up of the dough. One focused question works far better than
    a torn rule buried in the main prompt (that missed both broken-dough photos)."""
    x, y, w, h = cv2.boundingRect(contour)
    p = int(0.35 * max(w, h))
    zoom = img.crop((max(0, x - p), max(0, y - p), min(img.width, x + w + p), min(img.height, y + h + p)))
    zoom.thumbnail((DECIDE_SIDE, DECIDE_SIDE))
    msg = _chat({"model": model, "stream": False, "think": False, "keep_alive": "60m",
                 "options": {"temperature": 0.0, "num_predict": 5},
                 "messages": [{"role": "user", "content": TORN_Q, "images": [encode(zoom)]}]})
    return (msg.get("content") or "").strip().lower().startswith("yes")


def decide(img, history=(), model=MODEL, empty=None, board_cm=None):
    """Ask Gemma for the next action. Returns (speech, tool_name, args, latency_s).

    `img` may be larger than DECIDE_SIDE: the torn check zooms into it at full detail.
    """
    t0 = time.time()
    small = img.copy()
    small.thumbnail((DECIDE_SIDE, DECIDE_SIDE))
    m = measure(small, empty)
    if m is not None:
        k = img.width / small.width
        if is_torn(img, (m["contour"] * k).astype(np.int32), model):
            return ("The dough is torn. Rolling can't fix that, so I'm stopping.", "stop",
                    {"reason": "dough torn or cracked"}, time.time() - t0)

    prompt = "Here is the dough now." + camera_report(m, history, board_cm)
    if history:
        prompt += " Your previous actions, oldest first: " + "; ".join(history[-6:]) + "."
    msg = _chat({
        "model": model,
        "stream": False,
        "think": THINK,
        "keep_alive": "60m",
        "options": {"num_predict": 600 if THINK else 200, "temperature": 0.0},
        "tools": TOOLS,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt, "images": [encode(small)]},
        ],
    })
    dt = time.time() - t0

    text = (msg.get("content") or "").strip().split("\n")[0][:200].strip("\"' ")
    calls = msg.get("tool_calls") or []
    if not calls:
        return text or "I'm not sure, looking again.", "inspect", {}, dt
    fn = calls[0]["function"]
    args = fn.get("arguments") or {}
    if isinstance(args, str):
        args = json.loads(args)
    if fn["name"] == "rotate_dough":  # a person acts on this: always say exactly how far and which way
        text = describe(fn["name"], args)
    if fn["name"] == "roll" and text:  # never say one angle and roll another
        try:
            if "%d degree" % round(float(args.get("direction_deg", 0))) not in text:
                text = describe(fn["name"], args)
        except (TypeError, ValueError):
            pass
    return text or describe(fn["name"], args), fn["name"], args, dt
