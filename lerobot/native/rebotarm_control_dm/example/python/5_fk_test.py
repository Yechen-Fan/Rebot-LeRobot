#!/usr/bin/env python3
"""reBotArm RT forward-kinematics example.

Input six joint angles in degrees and print end-effector pose.
This example does not connect to hardware.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

SOURCE_PYTHON = Path(__file__).resolve().parents[2] / "python"
if SOURCE_PYTHON.exists() and str(SOURCE_PYTHON) not in sys.path:
    sys.path.insert(0, str(SOURCE_PYTHON))

from rebotarm_control_dm.kinematics import compute_fk, get_joint_names, load_robot_model


def matrix_to_rpy_xyz(rot: np.ndarray) -> np.ndarray:
    sy = math.hypot(float(rot[0, 0]), float(rot[1, 0]))
    singular = sy < 1e-9
    if not singular:
        roll = math.atan2(float(rot[2, 1]), float(rot[2, 2]))
        pitch = math.atan2(float(-rot[2, 0]), sy)
        yaw = math.atan2(float(rot[1, 0]), float(rot[0, 0]))
    else:
        roll = math.atan2(float(-rot[1, 2]), float(rot[1, 1]))
        pitch = math.atan2(float(-rot[2, 0]), sy)
        yaw = 0.0
    return np.array([roll, pitch, yaw], dtype=float)


def parse_joint_degrees(line: str, n: int) -> np.ndarray:
    values = line.split()
    if len(values) != n:
        raise ValueError(f"need {n} joint values, got {len(values)}")
    return np.array([float(v) for v in values], dtype=float)


def main() -> None:
    model = load_robot_model()
    joint_names = get_joint_names(model)

    print("=" * 56)
    print("  reBotArm RT FK test")
    print("=" * 56)
    print(f"  joints: {joint_names}")
    print(f"  nq={model.nq}, nv={model.nv}")
    print()
    print(f"Input {model.nq} joint angles in degrees.")
    print("Example: 0 0 0 0 0 0")
    print("Example: 30 -40 -50 10 20 0")
    print("-" * 56)

    try:
        q_deg = parse_joint_degrees(input("> ").strip(), model.nq)
    except (EOFError, ValueError) as exc:
        print(f"error: {exc}")
        return

    position, rotation, transform = compute_fk(model, np.radians(q_deg))
    rpy_deg = np.degrees(matrix_to_rpy_xyz(np.asarray(rotation)))

    print()
    print("Result")
    print("-" * 56)
    for name, deg in zip(joint_names, q_deg):
        print(f"  {name:8s}: {deg:+8.3f} deg")
    print()
    print(f"  position [m]: x={position[0]:+.6f}, y={position[1]:+.6f}, z={position[2]:+.6f}")
    print(f"  rpy [deg]:    roll={rpy_deg[0]:+.3f}, pitch={rpy_deg[1]:+.3f}, yaw={rpy_deg[2]:+.3f}")
    print("  transform:")
    for row in np.asarray(transform):
        print("   ", " ".join(f"{v:+.6f}" for v in row))


if __name__ == "__main__":
    main()
