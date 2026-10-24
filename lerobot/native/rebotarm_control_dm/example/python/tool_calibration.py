#!/usr/bin/env python3
"""Interactive 4-point TCP calibration with gravity-compensated free-drive."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

SOURCE_PYTHON = Path(__file__).resolve().parents[2] / "python"
if SOURCE_PYTHON.exists() and str(SOURCE_PYTHON) not in sys.path:
    sys.path.insert(0, str(SOURCE_PYTHON))

from rebotarm_control_dm.actuator import RobotArm
from rebotarm_control_dm.calibration import (
    FreeDrive,
    apply_tool_to_urdf,
    flange_pose,
    matrix_to_xyz_rpy_deg,
    solve_tcp_full,
    solve_tcp_position,
    tool_axis_from_pose,
    urdf_joint_origin_matrix,
)
from rebotarm_control_dm.kinematics import _URDF, load_robot_model
from rebotarm_control_dm.paths import default_calibration_dir
from _example_config import add_port_argument, config_with_port


def _wait_enter(prompt: str) -> bool:
    try:
        value = input(prompt).strip().lower()
    except EOFError:
        return False
    return value not in {"q", "quit", "exit"}


def _capture_flange_pose(arm, model, frame: str) -> np.ndarray:
    q = np.asarray(arm.get_positions(request=True), dtype=float)
    return flange_pose(model, q[: model.nq], frame)


def _print_pose_summary(label: str, pose: np.ndarray) -> None:
    xyz = pose[:3, 3]
    print(f"{label}: flange xyz = [{xyz[0]:+.4f}, {xyz[1]:+.4f}, {xyz[2]:+.4f}] m")


def _save_yaml(path: Path, transform: np.ndarray, residuals: dict[str, float], frame: str) -> None:
    xyz_m, rpy_deg = matrix_to_xyz_rpy_deg(transform)
    payload = {
        "frame": {
            "parent": frame,
            "child": "tool_tcp",
        },
        "xyz_m": [float(v) for v in xyz_m],
        "xyz_mm": [float(v * 1000.0) for v in xyz_m],
        "rpy_deg": [float(v) for v in rpy_deg],
        "T_flange_tool": [[float(v) for v in row] for row in transform],
        "mode": str(residuals.get("mode", "unknown")),
        "residual_mm": {
            key: float(value)
            for key, value in residuals.items()
            if isinstance(value, int | float | np.floating)
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "-c", default=None, help="Path to arm YAML config.")
    add_port_argument(parser)
    parser.add_argument("--frame", default="link6", help="Flange frame name. Default: link6.")
    parser.add_argument("--samples", type=int, default=4, help="Number of fixed-point touch samples.")
    parser.add_argument("--rate", type=float, default=200.0, help="Free-drive callback rate.")
    parser.add_argument("--kd", type=float, default=2.0, help="MIT damping during free-drive.")
    parser.add_argument("--gravity-scale", type=float, default=1.0, help="Gravity torque scale.")
    default_output = default_calibration_dir() / "tool_calibration.yaml"
    parser.add_argument("--output", default=str(default_output), help="Output YAML path.")
    parser.add_argument("--urdf-input", default=_URDF, help="Input URDF to copy and update.")
    parser.add_argument(
        "--urdf-output",
        default=None,
        help="Output calibrated URDF path. Defaults to --output with .urdf suffix.",
    )
    parser.add_argument("--urdf-joint", default="end_joint", help="Fixed joint to update in the URDF.")
    parser.add_argument(
        "--calibrate-orientation",
        action="store_true",
        help="Also calibrate TCP orientation with +Z/+X direction poses. Default: only calibrate TCP translation.",
    )
    args = parser.parse_args()

    if args.samples < 4:
        raise ValueError("--samples must be at least 4")

    model = load_robot_model()
    frames = model.all_frame_names()
    if args.frame not in frames:
        raise ValueError(f"unknown frame {args.frame!r}; available frames include: {frames}")

    arm = RobotArm(config_with_port(args.config, args.port))
    free_drive: FreeDrive | None = None
    output = Path(args.output)
    urdf_output = Path(args.urdf_output) if args.urdf_output else output.with_suffix(".urdf")

    print("=" * 72)
    print("  reBotArm TCP calibration")
    print("=" * 72)
    print(f"flange frame: {args.frame}")
    print(f"touch samples: {args.samples}")
    print(f"free-drive: rate={args.rate} Hz, kd={args.kd}, gravity_scale={args.gravity_scale}")
    print("Press Enter to capture each pose. Type q then Enter to abort.")
    print("-" * 72)

    try:
        arm.connect()
        print("[connect] OK")
        arm.enable()
        print("[enable] OK")

        touch_poses: list[np.ndarray] = []
        free_drive = FreeDrive(arm, model, rate=args.rate, kd=args.kd, gravity_scale=args.gravity_scale)
        free_drive.start()
        print("\n[free-drive] started. The arm should be movable by hand.")
        time.sleep(0.2)

        for idx in range(args.samples):
            prompt = (
                f"\nTouch the fixed reference point with the tool tip "
                f"from pose {idx + 1}/{args.samples}, then press Enter: "
            )
            if not _wait_enter(prompt):
                print("aborted")
                return
            pose = _capture_flange_pose(arm, model, args.frame)
            touch_poses.append(pose)
            _print_pose_summary(f"captured touch {idx + 1}", pose)

        u, world_point, res_max_mm, res_mean_mm = solve_tcp_position(touch_poses)
        transform = urdf_joint_origin_matrix(args.urdf_input, joint_name=args.urdf_joint)
        transform[:3, 3] = u
        residuals = {"max": res_max_mm, "mean": res_mean_mm, "mode": "position_only"}
        preserve_urdf_rpy = True
        print("\n[4-point position result]")
        print(f"  tcp in flange u [mm]: {(u * 1000.0).round(3).tolist()}")
        print(f"  fixed world point [m]: {world_point.round(6).tolist()}")
        print(f"  residual max/mean [mm]: {res_max_mm:.3f} / {res_mean_mm:.3f}")

        if args.calibrate_orientation:
            print("\nMove the tip back to the fixed reference point O, then pull along tool +Z by about 10 cm.")
            if not _wait_enter("Press Enter to capture the +Z direction pose: "):
                print("aborted")
                return
            z_pose = _capture_flange_pose(arm, model, args.frame)
            z_axis = tool_axis_from_pose(z_pose, u, world_point)
            print(f"captured +Z axis in flange: {z_axis.round(6).tolist()}")

            print("\nMove the tip back to O again, then pull along tool +X by about 10 cm.")
            print("Important: +X must be a different tool direction from +Z, not the same pull direction.")
            if not _wait_enter("Press Enter to capture the +X direction pose: "):
                print("aborted")
                return
            x_pose = _capture_flange_pose(arm, model, args.frame)
            x_axis = tool_axis_from_pose(x_pose, u, world_point)
            print(f"captured raw +X axis in flange: {x_axis.round(6).tolist()}")

            try:
                transform, residuals = solve_tcp_full(touch_poses, z_pose, x_pose)
                residuals["mode"] = "position_and_orientation"
                preserve_urdf_rpy = False
            except ValueError as exc:
                print(f"\n[orientation] failed: {exc}")
                print("[orientation] saved position-only calibration instead. Re-run with a real +X direction pose if orientation is needed.")
        else:
            print("\n[orientation] skipped. Keeping the existing URDF end_joint orientation and updating TCP translation only.")

        xyz_m, rpy_deg = matrix_to_xyz_rpy_deg(transform)
        _save_yaml(output, transform, residuals, args.frame)
        apply_tool_to_urdf(
            args.urdf_input,
            transform,
            urdf_output,
            joint_name=args.urdf_joint,
            preserve_rpy=preserve_urdf_rpy,
        )

        print("\n[result]")
        print(f"  xyz [m]: {xyz_m.round(6).tolist()}")
        print(f"  xyz [mm]: {(xyz_m * 1000.0).round(3).tolist()}")
        print(f"  rpy [deg]: {rpy_deg.round(3).tolist()}")
        print(f"  mode: {residuals.get('mode', 'unknown')}")
        print(f"  residual max/mean [mm]: {residuals['max']:.3f} / {residuals['mean']:.3f}")
        print("  T_flange_tool:")
        for row in transform:
            print("   ", " ".join(f"{float(v):+.8f}" for v in row))
        print(f"\n[saved] {output}")
        print(f"[saved] {urdf_output}")
        print("\nUse the generated URDF when loading RobotModel so end_link becomes the new TCP.")
        print("\n[free-drive] Calibration is saved; gravity compensation is still running.")
        print("[free-drive] Press Ctrl+C to disable motors, disconnect, and exit.")
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[exit] Ctrl+C received.")
    finally:
        if free_drive is not None:
            try:
                free_drive.stop()
            except Exception:
                pass
        else:
            try:
                arm.stop_control_loop()
            except Exception:
                pass
        try:
            arm.disable()
        except Exception:
            pass
        try:
            arm.disconnect()
        except Exception as exc:
            print(f"disconnect error: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
