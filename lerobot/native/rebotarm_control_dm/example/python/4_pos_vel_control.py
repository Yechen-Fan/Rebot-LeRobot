#!/usr/bin/env python3
"""Interactive RT-native POS_VEL position control.

Python only updates targets. The control loop runs on a Rust thread.
Input n joint angles in degrees. Optional trailing value overrides vlim.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SOURCE_PYTHON = Path(__file__).resolve().parents[2] / "python"
if SOURCE_PYTHON.exists() and str(SOURCE_PYTHON) not in sys.path:
    sys.path.insert(0, str(SOURCE_PYTHON))

from rebotarm_control_dm.actuator import RobotArm
from _example_config import add_port_argument, config_with_port


def print_state(arm) -> None:
    pos, vel, torque = arm.get_state(request=True)
    names = list(arm.joint_names)
    print("  pos(deg):", {name: round(float(deg), 2) for name, deg in zip(names, np.degrees(pos))})
    print("  vel(deg/s):", {name: round(float(deg), 2) for name, deg in zip(names, np.degrees(vel))})
    print("  torque:", {name: round(float(tau), 3) for name, tau in zip(names, torque)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "-c", default=None, help="Path to arm YAML config.")
    add_port_argument(parser)
    parser.add_argument("--rate", type=float, default=None, help="RT loop rate in Hz. Defaults to YAML rate.")
    parser.add_argument("--rt-priority", type=int, default=0, help="Best-effort SCHED_FIFO priority.")
    parser.add_argument("--cpu", type=int, default=None, help="Optional CPU affinity.")
    parser.add_argument("--request-feedback", action="store_true", help="Request feedback from RT loop.")
    args = parser.parse_args()

    arm = RobotArm(config_with_port(args.config, args.port))
    try:
        arm.connect()
        print("--- connected ---")
        arm.enable()
        print("--- enabled ---")
        arm.mode_pos_vel()
        print("--- POS_VEL mode ---")

        n = arm.num_joints
        names = list(arm.joint_names)
        vlim = np.array([j.vlim for j in arm._joints], dtype=np.float64)

        # Do not set a target before start_rt_loop: native loop will hold current pose.
        arm.start_rt_loop(
            rate=args.rate,
            rt_priority=args.rt_priority,
            cpu=args.cpu,
            request_feedback=args.request_feedback,
        )
        print(f"joints: {names}")
        print("RT loop started. First target is current pose hold.")
        print("Input: q1 ... qN [vlim], 'state', or 'q'. Angles are degrees.")

        while True:
            line = input("> ").strip()
            if not line:
                continue
            if line.lower() in {"q", "quit", "exit"}:
                break
            if line.lower() == "state":
                print_state(arm)
                print(f"  overruns send/read: {arm.rt_send_overruns}/{arm.rt_read_overruns}")
                continue

            values = line.split()
            if len(values) < n:
                print(f"need {n} joint values")
                continue
            target_deg = np.array([float(v) for v in values[:n]], dtype=np.float64)
            if len(values) >= n + 1:
                vlim[:] = float(values[n])

            arm.set_targets(pos=np.radians(target_deg).tolist(), vlim=vlim.tolist())
            print(f"  target(deg): {[round(float(x), 2) for x in target_deg]}  vlim={vlim[0]:.3f} rad/s")
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        arm.disconnect()


if __name__ == "__main__":
    main()
