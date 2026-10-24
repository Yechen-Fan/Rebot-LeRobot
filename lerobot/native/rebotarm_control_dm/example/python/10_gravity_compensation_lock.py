#!/usr/bin/env python3
"""Gravity compensation with end-effector velocity lock.

This mirrors reBotArm_control_py's lock demo, but uses the RT package's C++
Pinocchio bindings instead of Python pinocchio.

When the end-effector velocity is below thresholds, q_target stays locked.
When the arm is pushed fast enough, q_target is updated to the current pose.
"""
from __future__ import annotations

import argparse
import signal
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

SOURCE_PYTHON = Path(__file__).resolve().parents[2] / "python"
if SOURCE_PYTHON.exists() and str(SOURCE_PYTHON) not in sys.path:
    sys.path.insert(0, str(SOURCE_PYTHON))

from rebotarm_control_dm.actuator import RobotArm
from rebotarm_control_dm.dynamics import (
    compute_generalized_gravity,
    get_default_gravity,
    load_dynamics_model,
)
from rebotarm_control_dm.kinematics import _URDF, load_robot_model
from _example_config import add_port_argument, config_with_port


_running = True
END_LINK_LOAD_SCALE_WITH_GRIPPER = 0.7


def _sigint_handler(signum, frame) -> None:
    global _running
    print("\n[gravity_lock] stopping...")
    _running = False


def _str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def _end_link_load_urdf(scale: float) -> str:
    if scale < 0.0:
        raise ValueError("--end-link-load-scale must be >= 0")

    tree = ET.parse(_URDF)
    root = tree.getroot()
    end_link = root.find("./link[@name='end_link']")
    inertial = end_link.find("inertial") if end_link is not None else None
    if end_link is None or inertial is None:
        raise RuntimeError("URDF does not contain end_link inertial to scale")

    if scale == 0.0:
        end_link.remove(inertial)
    else:
        mass = inertial.find("mass")
        inertia = inertial.find("inertia")
        if mass is None or inertia is None:
            raise RuntimeError("URDF end_link inertial is missing mass or inertia")
        mass.set("value", str(float(mass.attrib["value"]) * scale))
        for key in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"):
            inertia.set(key, str(float(inertia.attrib[key]) * scale))

    tmp = tempfile.NamedTemporaryFile("wb", suffix=".urdf", delete=False)
    with tmp:
        tree.write(tmp, encoding="utf-8", xml_declaration=True)
    return tmp.name


def _load_gravity_model(use_gripper: bool):
    scale = END_LINK_LOAD_SCALE_WITH_GRIPPER if use_gripper else 0.0
    if scale == 1.0:
        return load_dynamics_model()
    tmp_urdf = _end_link_load_urdf(scale)
    try:
        return load_dynamics_model(tmp_urdf)
    finally:
        Path(tmp_urdf).unlink(missing_ok=True)


def _lock_command_vectors(arm, model_nq: int, q_target: np.ndarray, tau_g: np.ndarray, integral: np.ndarray, args):
    n = arm.num_joints
    pos = q_target.copy()
    vel = np.zeros(n, dtype=float)
    kp = np.full(n, args.kp, dtype=float)
    kd = np.full(n, args.kd, dtype=float)
    tau = np.zeros(n, dtype=float)
    tau[:model_nq] = tau_g
    tau += integral

    if n > model_nq:
        if args.use_gripper:
            kp[model_nq:] = args.gripper_kp
            kd[model_nq:] = args.gripper_kd
        else:
            kp[model_nq:] = 0.0
            kd[model_nq:] = 0.0
            tau[model_nq:] = 0.0
    return pos, vel, kp, kd, tau


def _release_mit_torque_hold(arm, frames: int = 10, dt_s: float = 0.02) -> None:
    """Send a short zero-torque MIT hold before leaving gravity compensation."""
    q_all = np.asarray(arm.get_positions(request=True), dtype=float)
    n = arm.num_joints
    if q_all.size != n:
        q_all = np.resize(q_all, n)
    zeros = np.zeros(n, dtype=float).tolist()
    pos = q_all.tolist()
    for _ in range(frames):
        arm.mit(
            pos=pos,
            vel=zeros,
            kp=zeros,
            kd=zeros,
            tau=zeros,
            request_feedback=False,
        )
        time.sleep(dt_s)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "-c", default=None, help="Path to arm YAML config.")
    add_port_argument(parser)
    parser.add_argument("--rate", type=float, default=200.0, help="Python control rate in Hz.")
    parser.add_argument("--vel-threshold", type=float, default=0.04, help="Linear velocity threshold.")
    parser.add_argument("--w-threshold", type=float, default=0.08, help="Angular velocity threshold.")
    parser.add_argument("--kp", type=float, default=8.0, help="MIT lock stiffness.")
    parser.add_argument("--kd", type=float, default=1.0, help="MIT lock damping.")
    parser.add_argument("--integral-limit", type=float, default=0.5, help="Integral torque clamp.")
    parser.add_argument(
        "--use_gripper",
        "--use-gripper",
        dest="use_gripper",
        type=_str_to_bool,
        default=True,
        metavar="{true,false}",
        help="Whether to include the fixed end_link gripper load in the gravity model.",
    )
    parser.add_argument("--gripper-kp", type=float, default=0.0, help="MIT stiffness for extra gripper joints.")
    parser.add_argument("--gripper-kd", type=float, default=1.0, help="MIT damping for extra gripper joints.")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _sigint_handler)

    kin_model = load_robot_model()
    dyn_model = _load_gravity_model(args.use_gripper)
    end_link_scale = END_LINK_LOAD_SCALE_WITH_GRIPPER if args.use_gripper else 0.0
    frame_id = kin_model.end_effector_frame_id()

    print("=" * 65)
    print("  reBotArm RT gravity compensation with velocity lock")
    print("=" * 65)
    print(f"[model] nq={dyn_model.nq}, nv={dyn_model.nv}")
    print(f"[gravity] {get_default_gravity()} m/s^2")
    print(f"[gripper/end_link load] scale={end_link_scale:.3f}")
    print(f"[thresholds] v={args.vel_threshold} m/s, w={args.w_threshold} rad/s")
    print("Ctrl+C to stop and disconnect.")
    print("-" * 65)

    arm = RobotArm(config_with_port(args.config, args.port))
    connected = False
    q_target: np.ndarray | None = None
    integral: np.ndarray | None = None
    counter = 0
    lock_counter = 0
    dt = 1.0 / args.rate

    try:
        arm.connect()
        connected = True
        print("[connect] OK")
        arm.enable()
        print("[enable] OK")

        q_target = np.asarray(arm.get_positions(request=True), dtype=float)
        integral = np.zeros_like(q_target)
        print(f"[target] initial lock: {np.degrees(q_target).round(2)} deg")

        if arm.num_joints < dyn_model.nq:
            raise ValueError(f"arm config has {arm.num_joints} joints, dynamics model requires {dyn_model.nq}")
        if args.use_gripper and arm.num_joints > dyn_model.nq:
            names = list(arm.joint_names)[dyn_model.nq:]
            print(f"[gripper] using extra arm joint(s) for gripper compensation: {names}")

        kp = np.full(arm.num_joints, args.kp, dtype=float)
        kd = np.full(arm.num_joints, args.kd, dtype=float)
        if arm.num_joints > dyn_model.nq:
            if args.use_gripper:
                kp[dyn_model.nq:] = args.gripper_kp
                kd[dyn_model.nq:] = args.gripper_kd
            else:
                kp[dyn_model.nq:] = 0.0
                kd[dyn_model.nq:] = 0.0
        arm.mode_mit(kp=kp.tolist(), kd=kd.tolist())
        print(f"[MIT mode] kp={args.kp}, kd={args.kd}")

        while _running:
            t0 = time.perf_counter()
            q = np.asarray(arm.get_positions(request=True), dtype=float)
            qd = np.asarray(arm.get_velocities(), dtype=float)
            q_model = q[: dyn_model.nq]
            qd_model = qd[: kin_model.nv]
            tau_g = np.asarray(compute_generalized_gravity(dyn_model, q_model), dtype=float)

            q_error = q_target - q
            integral += q_error * dt
            np.clip(integral, -args.integral_limit, args.integral_limit, out=integral)

            jacobian = np.asarray(kin_model.frame_jacobian(q_model, frame_id), dtype=float)
            v_spatial = jacobian @ qd_model
            v_norm = float(np.linalg.norm(v_spatial[:3]))
            w_norm = float(np.linalg.norm(v_spatial[3:]))

            if v_norm > args.vel_threshold or w_norm > args.w_threshold:
                q_target = q.copy()
                lock_counter = 0
                integral *= 0.9
            else:
                lock_counter += 1

            pos, vel, kp_cmd, kd_cmd, tau = _lock_command_vectors(arm, dyn_model.nq, q_target, tau_g, integral, args)

            arm.mit(
                pos=pos.tolist(),
                vel=vel.tolist(),
                kp=kp_cmd.tolist(),
                kd=kd_cmd.tolist(),
                tau=tau.tolist(),
                request_feedback=False,
            )

            counter += 1
            if counter % 20 == 0:
                state = "LOCKED" if lock_counter > 0 else "UPDATE"
                print(
                    f"[{counter:4d}] {state} "
                    f"err_max={float(np.max(np.abs(q_error))):.4f}rad "
                    f"v={v_norm:.4f}m/s w={w_norm:.4f}rad/s "
                    f"tau_g=" + " ".join(f"{float(t):+.3f}" for t in tau_g)
                )

            elapsed = time.perf_counter() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)
    finally:
        print("\n[stop] disconnecting...")
        if connected:
            try:
                _release_mit_torque_hold(arm)
            except Exception as exc:
                print(f"[stop] failed to release MIT torque cleanly: {exc}")
            arm.disconnect()
        print("[done] disconnected")


if __name__ == "__main__":
    main()
