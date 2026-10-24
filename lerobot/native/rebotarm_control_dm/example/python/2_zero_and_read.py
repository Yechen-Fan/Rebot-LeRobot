#!/usr/bin/env python3
"""Zero current motor positions, then print live joint state.

This example disables motors during zeroing. Make sure the arm is supported and
already placed at the desired zero pose before continuing.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

SOURCE_PYTHON = Path(__file__).resolve().parents[2] / "python"
if SOURCE_PYTHON.exists() and str(SOURCE_PYTHON) not in sys.path:
    sys.path.insert(0, str(SOURCE_PYTHON))

from rebotarm_control_dm.actuator import RobotArm
from _example_config import add_port_argument, config_with_port


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "-c", default=None, help="Path to arm YAML config.")
    add_port_argument(parser)
    parser.add_argument("--skip-zero", action="store_true", help="Only read state; do not call set_zero().")
    parser.add_argument("--interval", type=float, default=0.05, help="Print interval in seconds.")
    args = parser.parse_args()

    arm = RobotArm(config_with_port(args.config, args.port))
    try:
        arm.connect()
        print("--- connected ---")
        print(f"joints: {list(arm.joint_names)}")

        if not args.skip_zero:
            answer = input("Set current pose as zero? Type YES to continue: ").strip()
            if answer != "YES":
                print("aborted")
                return
            arm.set_zero()
            print("--- zero set ---")

        print("--- live state in deg, Ctrl+C to exit ---")
        while True:
            pos, vel, torque = arm.get_state(request=True)
            row = "  ".join(
                f"{name}:{deg:+7.2f}"
                for name, deg in zip(arm.joint_names, np.degrees(pos))
            )
            print(f"\r{row}  ", end="", flush=True)
            time.sleep(max(args.interval, 0.001))
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        print()
        arm.disconnect()


if __name__ == "__main__":
    main()
