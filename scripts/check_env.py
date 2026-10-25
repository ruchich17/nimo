"""Check the dev box is ready: Ollama + Gemma models + Python deps + MuJoCo.

Usage (inside the venv): python scripts/check_env.py
Prints OK/FAIL per check; exits non-zero if anything failed.
"""
import importlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request

OLLAMA = os.environ.get("OLLAMA_HOST_URL", "http://localhost:11434")
MODELS = ["gemma4:e4b"]
failed = []


def check(name, fn):
    try:
        detail = fn()
        print("OK    %-28s %s" % (name, detail or ""))
    except Exception as e:
        failed.append(name)
        print("FAIL  %-28s %s" % (name, e))


def python_version():
    v = sys.version_info
    assert (v.major, v.minor) == (3, 11), "need 3.11, got %d.%d (venv not activated?)" % (v.major, v.minor)
    return sys.executable


def ollama_version():
    with urllib.request.urlopen(OLLAMA + "/api/version", timeout=5) as r:
        v = json.load(r)["version"]
    major, minor = (int(x) for x in v.split(".")[:2])
    assert (major, minor) >= (0, 22), "need >= 0.22.0, got " + v
    return v


def models():
    with urllib.request.urlopen(OLLAMA + "/api/tags", timeout=5) as r:
        have = {m["name"] for m in json.load(r)["models"]}
    missing = [m for m in MODELS if m not in have]
    assert not missing, "missing: " + ", ".join(missing)
    return ", ".join(MODELS)


def module(name):
    return lambda: getattr(importlib.import_module(name), "__version__", "imported")


def mujoco_sim():
    import mujoco
    m = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body><freejoint/><geom size='.1'/></body></worldbody></mujoco>")
    d = mujoco.MjData(m)
    for _ in range(100):
        mujoco.mj_step(m, d)
    assert d.qpos[2] < 0, "ball did not fall"
    return "100 physics steps ran"


def mjpython():
    if sys.platform != "darwin":
        return "not needed on " + sys.platform
    path = shutil.which("mjpython")
    assert path, "mjpython not on PATH (venv not activated?)"
    return path


def tool_call():
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smoke_tool_call.py")
    out = subprocess.run([sys.executable, script, "-", MODELS[0]],
                         capture_output=True, text=True, timeout=600).stdout
    assert "PASS" in out, out.strip()[-300:] or "no output"
    latency = [l for l in out.splitlines() if l.startswith("latency")]
    return latency[0].split(":", 1)[1].strip() + " (first call includes model load)"


check("python 3.11 venv", python_version)
check("ollama server >= 0.22", ollama_version)
check("gemma models pulled", models)
for mod in ["ollama", "cv2", "PIL", "numpy", "mujoco"]:
    check("import " + mod, module(mod))
check("mujoco physics", mujoco_sim)
check("mjpython (macOS viewer)", mjpython)
check("gemma vision + tool call", tool_call)

print()
print("ALL GOOD" if not failed else "FAILED: " + ", ".join(failed))
sys.exit(1 if failed else 0)
