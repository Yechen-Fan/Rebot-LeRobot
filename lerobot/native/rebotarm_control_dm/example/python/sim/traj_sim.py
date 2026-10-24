#!/usr/bin/env python3
"""Cartesian target trajectory simulation with MeshCat playback."""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_PYTHON_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, EXAMPLE_PYTHON_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rebotarm_control_dm.kinematics import IKParams, compute_fk, pos_rot_to_se3
from rebotarm_control_dm.trajectory import (
    CLIKParams,
    TrajPlanParams,
    TrajProfile,
    compute_traj_stats,
    plan_joint_space_trajectory,
)
from sim.visualizer import Visualizer

LINEAR_SPEED = 0.1


def rpy_xyz_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


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


def run_trajectory(viz: Visualizer, model, frame_id: int, q_start: np.ndarray, q_end: np.ndarray, duration: float):
    dt = 1.0 / 50.0
    params = TrajPlanParams(dt=dt, profile=TrajProfile.MIN_JERK, accel_ratio=0.25)
    clik_params = CLIKParams(max_iter=200, tolerance=1e-4, damping=1e-6, step_size=0.8)
    _, _, t_start = compute_fk(model, q_start)
    _, _, t_end = compute_fk(model, q_end)

    t0 = time.time()
    joint_traj = plan_joint_space_trajectory(model, frame_id, q_start, q_end, duration, params, clik_params, 0.1)
    elapsed_ms = (time.time() - t0) * 1000.0
    stats = compute_traj_stats(model, frame_id, joint_traj, t_start, t_end, duration, params)

    ee_positions = []
    for pt in joint_traj:
        _, _, transform = compute_fk(model, pt.q)
        ee_positions.append(transform[:3, 3].tolist())

    print("=" * 60)
    print(f"trajectory: points={len(joint_traj)} duration={duration:.2f}s dt={dt:.3f}s compute={elapsed_ms:.1f}ms")
    print(f"IK success={stats.success_rate:.1%} max_error={stats.max_ik_error:.3e} avg_error={stats.avg_ik_error:.3e}")
    print(f"q: {np.degrees(q_start).round(1).tolist()} -> {np.degrees(q_end).round(1).tolist()}")
    print("=" * 60)

    viz.clear_paths()
    viz.clear_trajectory_line()
    if ee_positions:
        viz.draw_ref_path(ee_positions)

    visited = []
    times = np.array([pt.time for pt in joint_traj], dtype=float)
    print("playing animation in MeshCat...")
    try:
        for i, pt in enumerate(joint_traj):
            viz.update(pt.q)
            if i < len(ee_positions):
                visited.append(ee_positions[i])
                viz.draw_actual_path(visited)
            if i < len(times) - 1:
                time.sleep(max(0.002, times[i + 1] - times[i]))
    except KeyboardInterrupt:
        print("\ntrajectory interrupted.")
        return []

    if joint_traj:
        joint_arr = np.array([pt.q for pt in joint_traj])
        q_deg = np.degrees(joint_arr)
        print("joint angle ranges (deg):")
        for i in range(joint_arr.shape[1]):
            print(f"  j{i + 1}: [{q_deg[:, i].min():.1f}, {q_deg[:, i].max():.1f}]")

    return joint_traj


def main() -> None:
    print("loading MeshCat visualizer...")
    viz = Visualizer(open_browser=True)
    model = viz.model
    frame_id = model.end_effector_frame_id()
    q_last = model.neutral()
    viz.update(q_last)

    ik_params = IKParams(max_iter=2000, tolerance=1e-4, step_size=0.5, damping=1e-6)
    print("Input: x y z [roll pitch yaw] (meters / radians), q to exit")
    print("Examples:")
    print("  0.20 0.10 0.20")
    print("  0.24 -0.08 0.22")
    print("  0.20 0.10 0.20 0 0.4 0\n")

    while True:
        _, _, current = compute_fk(model, q_last)
        pos = current[:3, 3]
        rpy = matrix_to_rpy_xyz(current[:3, :3])
        print(f"pos[{pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}] rpy[{rpy[0]:.3f} {rpy[1]:.3f} {rpy[2]:.3f}]> ", end="", flush=True)
        try:
            line = input().strip()
        except (KeyboardInterrupt, EOFError):
            print("\nexit.")
            break
        if not line:
            continue
        if line in ("q", "quit", "exit"):
            break

        try:
            vals = [float(x) for x in line.split()]
            if len(vals) not in (3, 6):
                print("format: x y z [roll pitch yaw]")
                continue
        except ValueError:
            print("format: x y z [roll pitch yaw]")
            continue

        x, y, z = vals[:3]
        if len(vals) == 6:
            target_rot = rpy_xyz_to_matrix(vals[3], vals[4], vals[5])
        else:
            target_rot = current[:3, :3]
        target_pose = pos_rot_to_se3(np.array([x, y, z], dtype=float), target_rot)
        ik = model.solve_ik_with_retry(target_pose, q_last, frame_id, ik_params, 10)
        if not ik.success:
            print(f"IK failed: error={ik.error:.4e}")
            continue

        duration = max(1.0, float(np.linalg.norm(target_pose[:3, 3] - current[:3, 3])) / LINEAR_SPEED)
        joint_traj = run_trajectory(viz, model, frame_id, q_last, ik.q, duration)
        if joint_traj:
            q_last = np.asarray(joint_traj[-1].q, dtype=float)
            viz.update(q_last)

    viz.neutral()
    print("\ndone.")


if __name__ == "__main__":
    main()
