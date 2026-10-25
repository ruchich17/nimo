"""Smoke test: can Gemma see a dough image and return a parseable tool call, and how fast?

Usage: [OLLAMA_HOST_URL=http://<mac-ip>:11434] python scripts/smoke_tool_call.py [image_path] [model]
With no image (or "-"), draws a synthetic lopsided dough blob.
"""
import base64
import io
import json
import sys
import time
import os
import urllib.request

from PIL import Image, ImageDraw

OLLAMA_URL = os.environ.get("OLLAMA_HOST_URL", "http://localhost:11434") + "/api/chat"

TOOLS = [
    {"type": "function", "function": {
        "name": "roll",
        "description": "Roll the dough with one stroke of the roller.",
        "parameters": {"type": "object", "properties": {
            "direction_deg": {"type": "number", "description": "Stroke direction, 0 = left-to-right, 90 = bottom-to-top"},
            "pressure": {"type": "number", "description": "0.0 (light) to 1.0 (firm)"},
            "length_cm": {"type": "number", "description": "Stroke length in cm, 3 to 15"},
        }, "required": ["direction_deg", "pressure", "length_cm"]}}},
    {"type": "function", "function": {
        "name": "inspect",
        "description": "Re-capture the camera image and re-assess the dough.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "done",
        "description": "Stop: the dough is round and evenly thin.",
        "parameters": {"type": "object", "properties": {
            "reason": {"type": "string"}}, "required": ["reason"]}}},
]

SYSTEM = (
    "You control a robot arm that rolls Indian roti dough with a rolling pin. "
    "You see the dough from directly overhead. Briefly say what is wrong with its shape, "
    "then call exactly one tool. Roll toward the parts that need to spread out."
)


def synthetic_blob():
    img = Image.new("RGB", (512, 512), (120, 85, 50))  # wooden board
    d = ImageDraw.Draw(img)
    d.ellipse((90, 170, 400, 360), fill=(235, 215, 170))   # lopsided oval
    d.ellipse((60, 200, 220, 330), fill=(225, 200, 150))   # thick lump on the left
    return img


def encode(img):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    img = Image.open(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] not in ("", "-") else synthetic_blob()
    model = sys.argv[2] if len(sys.argv) > 2 else "gemma4:e4b"
    body = {
        "model": model,
        "stream": False,
        "tools": TOOLS,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Here is the dough now.", "images": [encode(img)]},
        ],
    }
    req = urllib.request.Request(OLLAMA_URL, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.load(r)
    dt = time.time() - t0

    msg = resp.get("message", {})
    print("model:      ", model)
    print("latency:    %.1f s" % dt)
    print("text:       ", (msg.get("content") or "").strip()[:500])
    calls = msg.get("tool_calls") or []
    for c in calls:
        print("tool call:  ", c["function"]["name"], json.dumps(c["function"].get("arguments")))
    print("RESULT:     ", "PASS - tool call parsed" if calls else "FAIL - no tool call")


if __name__ == "__main__":
    main()
