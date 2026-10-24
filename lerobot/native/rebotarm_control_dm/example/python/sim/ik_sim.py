#!/usr/bin/env python3
"""Inverse-kinematics MeshCat simulation."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_PYTHON_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, EXAMPLE_PYTHON_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rebotarm_control_dm.kinematics import IKParams, compute_fk, pos_rot_to_se3
from sim.visualizer import Visualizer

def rpy_xyz_to_matrix(rpy_rad: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(v) for v in rpy_rad]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def main() -> None:
    print("loading visualizer...")
    viz = Visualizer()
    viz.neutral()
    model = viz.model
    frame_id = model.end_effector_frame_id()
    q_seed = model.neutral()
    params = IKParams(max_iter=2000, tolerance=1e-4, step_size=0.5, damping=1e-6)

    print("MeshCat is ready. Input target pose:")
    print("  x y z                    (meters, keep neutral orientation)")
    print("  x y z roll pitch yaw     (meters + radians)")
    print("Examples:")
    print("  0.26 0.00 0.19")
    print("  0.20 0.10 0.20")
    print("  0.20 0.10 0.20 0 0.4 0")
    print("q / quit / exit to stop\n")

    while True:
        try:
            line = input("target pose > ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\nexit.")
            break
        if line in ("q", "quit", "exit", ""):
            break
        try:
            vals = [float(x) for x in line.split()]
            if len(vals) not in (3, 6):
                print("need 3 values or 6 values\n")
                continue
        except ValueError:
            print("invalid input\n")
            continue

        target_pos = np.array(vals[:3], dtype=float)
        if len(vals) == 6:
            target_rot = rpy_xyz_to_matrix(np.array(vals[3:6], dtype=float))
        else:
            _, target_rot, _ = compute_fk(model, q_seed)

        target = pos_rot_to_se3(target_pos, target_rot)
        result = model.solve_ik_with_retry(target, q_seed, frame_id, params, 10)
        viz.update(result.q)
        q_seed = np.asarray(result.q, dtype=float)

        status = "converged" if result.success else "not converged"
        print(f"  [{status}] iterations={result.iterations} error={result.error:.2e}")
        print(f"  q(deg): {np.degrees(result.q)}\n")


if __name__ == "__main__":
    main()
