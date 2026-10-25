"""Read-only arm check: connect and print joint angles. Never enables a motor.

    python scripts/arm_read.py            # 10 readings, 0.5 s apart
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from robot.rebot_sdk import find_port, make_arm

port = find_port()
print("port:", port)
arm = make_arm(port)
arm.connect()
names = arm.arm.joint_names + (arm.gripper.joint_names if arm.has_gripper else [])
print("joints:", names)
try:
    for _ in range(int(sys.argv[1]) if len(sys.argv) > 1 else 10):
        q = arm.get_positions()
        print("deg:", " ".join("%+7.1f" % v for v in np.degrees(q)))
        time.sleep(0.5)
finally:
    arm.disconnect()
