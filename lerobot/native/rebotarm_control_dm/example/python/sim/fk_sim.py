#!/usr/bin/env python3
"""Forward-kinematics MeshCat simulation."""
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

from rebotarm_control_dm.kinematics import compute_fk
from sim.visualizer import Visualizer

def matrix_to_rpy_xyz(rot: np.ndarray) -> np.ndarray:
    sy = math.hypot(float(rot[0, 0]), float(rot[1, 0]))
    if sy >= 1e-9:
        roll = math.atan2(float(rot[2, 1]), float(rot[2, 2]))
        pitch = math.atan2(float(-rot[2, 0]), sy)
        yaw = math.atan2(float(rot[1, 0]), float(rot[0, 0]))
    else:
        roll = math.atan2(float(-rot[1, 2]), float(rot[1, 1]))
        pitch = math.atan2(float(-rot[2, 0]), sy)
        yaw = 0.0
    return np.array([roll, pitch, yaw], dtype=float)


def main() -> None:
    print("loading visualizer...")
    viz = Visualizer()
    q = np.zeros(viz.nq)
    viz.update(q)

    print("MeshCat is ready. Input joint angles in degrees.")
    print("Input format:")
    print("  j1 j2 j3 j4 j5 j6          (degrees)")
    print("Examples:")
    print("  0 0 0 0 0 0")
    print("  45 -30 -15 -60 90 60")
    print("  30 -20 -30 -40 70 0")
    print("q / quit / exit to stop\n")

    while True:
        try:
            line = input("joint angles > ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\nexit.")
            break
        if line in ("q", "quit", "exit", ""):
            break
        try:
            q_deg = [float(x) for x in line.split()]
            if len(q_deg) != viz.nq:
                print(f"need {viz.nq} values\n")
                continue
        except ValueError:
            print("invalid input\n")
            continue

        q = np.radians(q_deg)
        viz.update(q)
        pos, rot, _ = compute_fk(viz.model, q)
        rpy = np.degrees(matrix_to_rpy_xyz(np.asarray(rot)))
        print(f"  ee position: [{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}] m")
        print(f"  ee rpy:      [{rpy[0]:+.2f}, {rpy[1]:+.2f}, {rpy[2]:+.2f}] deg\n")


if __name__ == "__main__":
    main()
