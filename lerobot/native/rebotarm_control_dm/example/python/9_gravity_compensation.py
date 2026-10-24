#!/usr/bin/env python3
"""Gravity-compensation demo.

This example keeps Python callback control because each cycle computes a
dynamics feed-forward torque from the current joint state.

Use on hardware only:
    python example/python/9_gravity_compensation.py --port /dev/ttyACM0 [--config arm.yaml] [--rate 200]

Ctrl+C stops the control loop and disconnects.
"""
from __future__ import annotations

import argparse
import signal
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
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
from rebotarm_control_dm.kinematics import _URDF
from _example_config import add_port_argument, config_with_port


_running = True
END_LINK_LOAD_SCALE_WITH_GRIPPER = 0.7


def _sigint_handler(signum, frame) -> None:
    global _running
    print("\n[gravity_comp] stopping...")
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


def _mit_command_vectors(arm, model_nq: int, q_all: np.ndarray, tau_g: np.ndarray, args):
    n = arm.num_joints
    pos = q_all.copy()
    vel = np.zeros(n, dtype=float)
    kp = np.zeros(n, dtype=float)
    kd = np.ones(n, dtype=float)
    tau = np.zeros(n, dtype=float)
    tau[:model_nq] = tau_g

    # If a future arm config includes extra joints beyond the 6-DoF dynamics
    # model, keep them passive unless gripper compensation is explicitly enabled.
    if n > model_nq:
        if args.use_gripper:
            kp[model_nq:] = args.gripper_kp
            kd[model_nq:] = args.gripper_kd
        else:
            kd[model_nq:] = 0.0
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


def make_gravity_compensation_controller(model, args) -> Callable:
    counter = {"n": 0}
    model_nq = model.nq

    def controller(arm, dt: float) -> None:
        q_all = np.asarray(arm.get_positions(), dtype=float)
        if q_all.size < model_nq:
            raise ValueError(f"arm has {q_all.size} joints, dynamics model requires {model_nq}")
        q_model = q_all[:model_nq]
        tau_g = np.asarray(compute_generalized_gravity(model, q_model), dtype=float)

        pos, vel, kp, kd, tau = _mit_command_vectors(arm, model_nq, q_all, tau_g, args)

        arm.mit(
            pos=pos.tolist(),
            vel=vel.tolist(),
            kp=kp.tolist(),
            kd=kd.tolist(),
            tau=tau.tolist(),
            request_feedback=True,
        )

        counter["n"] += 1
        if counter["n"] % 20 == 0:
            tau_text = "  ".join(f"{float(t):+.3f}" for t in tau_g)
            print(f"[{counter['n']:4d}] tau_g = {tau_text}  N*m")

    return controller


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "-c", default=None, help="Path to arm YAML config.")
    add_port_argument(parser)
    parser.add_argument("--rate", type=float, default=None, help="Python callback loop rate in Hz.")
    parser.add_argument("--kp", type=float, default=2.0, help="MIT mode stiffness written before loop start.")
    parser.add_argument("--kd", type=float, default=1.0, help="MIT mode damping written before loop start.")
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

    model = _load_gravity_model(args.use_gripper)
    end_link_scale = END_LINK_LOAD_SCALE_WITH_GRIPPER if args.use_gripper else 0.0
    print("=" * 60)
    print("  reBotArm RT gravity-compensation demo")
    print("=" * 60)
    print(f"[model] nq={model.nq}, nv={model.nv}")
    print(f"[gravity] {get_default_gravity()} m/s^2")
    print(f"[gripper/end_link load] scale={end_link_scale:.3f}")
    print("Expected behavior: arm holds against gravity while remaining compliant.")
    print("Ctrl+C to stop and disconnect.")
    print("-" * 60)

    arm = RobotArm(config_with_port(args.config, args.port))
    rate = args.rate
    connected = False
    try:
        arm.connect()
        connected = True
        print("[connect] OK")
        arm.enable()
        print("[enable] OK")

        if arm.num_joints < model.nq:
            raise ValueError(f"arm config has {arm.num_joints} joints, dynamics model requires {model.nq}")
        if args.use_gripper and arm.num_joints > model.nq:
            names = list(arm.joint_names)[model.nq:]
            print(f"[gripper] using extra arm joint(s) for gripper compensation: {names}")

        kp = np.full(arm.num_joints, args.kp, dtype=float)
        kd = np.full(arm.num_joints, args.kd, dtype=float)
        if arm.num_joints > model.nq:
            if args.use_gripper:
                kp[model.nq:] = args.gripper_kp
                kd[model.nq:] = args.gripper_kd
            else:
                kp[model.nq:] = 0.0
                kd[model.nq:] = 0.0
        arm.mode_mit(kp=kp.tolist(), kd=kd.tolist())
        print(f"[MIT mode] kp={args.kp}, kd={args.kd}")

        if rate is None:
            rate = getattr(arm, "_rate", 200.0)
        arm.start_control_loop(make_gravity_compensation_controller(model, args), rate=rate)
        print(f"[control loop] started @ {rate} Hz")

        while _running:
            time.sleep(0.01)
    finally:
        print("\n[stop] disconnecting...")
        if connected:
            try:
                arm.stop_control_loop()
                _release_mit_torque_hold(arm)
            except Exception as exc:
                print(f"[stop] failed to release MIT torque cleanly: {exc}")
            arm.disconnect()
        print("[done] disconnected")


if __name__ == "__main__":
    main()
