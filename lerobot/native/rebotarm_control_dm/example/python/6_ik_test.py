#!/usr/bin/env python3
"""reBotArm RT inverse-kinematics example.

Input target x y z in meters, optionally followed by roll pitch yaw in degrees.
If only position is provided, the solver keeps the neutral FK orientation.
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

from rebotarm_control_dm.kinematics import IKParams, compute_fk, get_joint_names, load_robot_model, pos_rot_to_se3


def rpy_xyz_to_matrix(rpy_rad: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(v) for v in rpy_rad]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
    rot_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
    rot_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return rot_z @ rot_y @ rot_x


def parse_target(line: str) -> tuple[np.ndarray, np.ndarray | None]:
    values = line.split()
    if len(values) not in (3, 6):
        raise ValueError(f"need 3 values (x y z) or 6 values (x y z roll pitch yaw), got {len(values)}")
    nums = np.array([float(v) for v in values], dtype=float)
    pos = nums[:3]
    rot = rpy_xyz_to_matrix(np.radians(nums[3:6])) if len(nums) == 6 else None
    return pos, rot


def main() -> None:
    model = load_robot_model()
    joint_names = get_joint_names(model)

    print("=" * 56)
    print("  reBotArm RT IK test")
    print("=" * 56)
    print(f"  joints: {joint_names}")
    print()
    print("Input target pose:")
    print("  x y z                         (meters, position only)")
    print("  x y z roll pitch yaw          (meters + degrees)")
    print("Example: 0.2603 0.0 0.1917")
    print("Example: 0.2603 0.0 0.1917 0 0 0")
    print("-" * 56)

    try:
        target_pos, target_rot = parse_target(input("> ").strip())
    except (EOFError, ValueError) as exc:
        print(f"error: {exc}")
        return

    q_seed = model.neutral()
    if target_rot is None:
        _, seed_rot, _ = compute_fk(model, q_seed)
        target_rot = seed_rot
    target = pos_rot_to_se3(target_pos, target_rot)
    params = IKParams(max_iter=2000, tolerance=1e-4, step_size=0.5, damping=1e-6)
    result = model.solve_ik_with_retry(target, q_seed, model.end_effector_frame_id(), params, 10)

    print()
    print("Result")
    print("-" * 56)
    print(f"  success: {result.success}")
    print(f"  iterations: {result.iterations}")
    print(f"  error: {result.error:.6g}")
    print()
    for name, rad in zip(joint_names, result.q):
        print(f"  {name:8s}: {math.degrees(float(rad)):+8.3f} deg  ({float(rad):+.5f} rad)")


if __name__ == "__main__":
    main()
