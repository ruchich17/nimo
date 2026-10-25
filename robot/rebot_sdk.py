"""Load Seeed's reBot SDK (third_party/reBotArm_control_py) for the DM arm on this Mac.

Imports the actuator module by path: the SDK package __init__ also pulls in Pinocchio,
which we don't need for joint-level record/replay. Always uses the DM hardware config
(the SDK's default is the RS arm) with the serial port swapped for this Mac's.
"""
import importlib.util
import os
import sys
import tempfile

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SDK = os.path.join(ROOT, "third_party", "reBotArm_control_py")
DM_YAML = os.path.join(SDK, "config", "rebotarm_dm.yaml")


def find_port():
    """The arm's USB2CAN bridge on macOS, e.g. /dev/tty.usbmodem00000000050C1."""
    ports = sorted(f for f in os.listdir("/dev") if f.startswith("tty.usbmodem"))
    if not ports:
        raise RuntimeError("arm USB adapter not found (no /dev/tty.usbmodem*) - is it plugged in?")
    return "/dev/" + ports[0]


def make_arm(port=None):
    """A disconnected RebotArm for the DM arm on `port`. Call .connect(); nothing moves until enabled."""
    spec = importlib.util.spec_from_file_location(
        "rebotarm_sdk", os.path.join(SDK, "reBotArm_control_py", "actuator", "rebotarm.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rebotarm_sdk"] = mod  # dataclasses look the module up while it loads
    spec.loader.exec_module(mod)
    cfg = yaml.safe_load(open(DM_YAML))
    cfg["channel"] = port or find_port()
    tmp = os.path.join(tempfile.gettempdir(), "rebotarm_dm_mac.yaml")
    with open(tmp, "w") as f:
        yaml.safe_dump(cfg, f)
    return mod.RebotArm(hw_yaml=tmp)
