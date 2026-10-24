#!/usr/bin/env python3
"""Single Damiao motor test through rebotarm_control_dm RobotArm.

This mirrors reBotArm_control_py's motorbridge example, but uses the RT
RobotArm API and controls one selected joint while holding the others.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

SOURCE_PYTHON = Path(__file__).resolve().parents[2] / "python"
if SOURCE_PYTHON.exists() and str(SOURCE_PYTHON) not in sys.path:
    sys.path.insert(0, str(SOURCE_PYTHON))

from rebotarm_control_dm.actuator import RobotArm
from _example_config import add_port_argument, config_with_port


def joint_index(names: list[str], joint: str) -> int:
    try:
        idx = int(joint)
    except ValueError:
        if joint not in names:
            raise ValueError(f"unknown joint {joint!r}; available: {names}") from None
        idx = names.index(joint)
    if idx < 0 or idx >= len(names):
        raise ValueError(f"joint index {idx} out of range 0..{len(names) - 1}")
    return idx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "-c", default=None, help="Path to arm YAML config.")
    add_port_argument(parser)
    parser.add_argument("--joint", default="0", help="Joint index or name to control. Default: 0.")
    parser.add_argument("--rate", type=float, default=150.0, help="RT loop rate for target-cache modes.")
    parser.add_argument("--rt-priority", type=int, default=0, help="Best-effort SCHED_FIFO priority.")
    parser.add_argument("--cpu", type=int, default=None, help="Optional CPU affinity.")
    parser.add_argument("--request-feedback", action="store_true", help="Request feedback from RT loop.")
    args = parser.parse_args()

    arm = RobotArm(config_with_port(args.config, args.port))
    try:
        arm.connect()
        names = list(arm.joint_names)
        idx = joint_index(names, args.joint)
        name = names[idx]
        print(f"connected: {arm.name}")
        print(f"joint: {idx} ({name}); all joints: {names}")

        target_pos = np.asarray(arm.get_positions(request=True), dtype=float)
        mit_kp = np.array([j.kp for j in arm._joints], dtype=float)
        mit_kd = np.array([j.kd for j in arm._joints], dtype=float)
        pv_vlim = np.array([j.vlim for j in arm._joints], dtype=float)
        mode = "mit"
        rt_started = False

        def start_rt_if_needed() -> None:
            nonlocal rt_started
            if rt_started:
                return
            arm.start_rt_loop(
                rate=args.rate,
                rt_priority=args.rt_priority,
                cpu=args.cpu,
                request_feedback=args.request_feedback,
            )
            rt_started = True
            print(f"RT loop started @ {args.rate} Hz")

        def do_enable() -> None:
            arm.enable()
            print("enabled")

        def do_disable() -> None:
            arm.disable()
            print("disabled")

        def do_set_zero() -> None:
            answer = input(f"Set current {name} as zero? Type YES: ").strip()
            if answer != "YES":
                print("aborted")
                return
            arm.set_zero_single(name)
            print("zero set")

        def do_mode(values: list[str]) -> None:
            nonlocal mode
            if not values:
                print("usage: mode <mit|posvel|vel>")
                return
            arm.stop_control_loop()
            if values[0].lower() == "mit":
                arm.mode_mit(kp=mit_kp.tolist(), kd=mit_kd.tolist())
                mode = "mit"
            elif values[0].lower() == "posvel":
                arm.mode_pos_vel(vlim=pv_vlim.tolist())
                mode = "posvel"
            elif values[0].lower() == "vel":
                arm.mode_vel()
                mode = "vel"
            else:
                print("available modes: mit / posvel / vel")
                return
            print(f"mode: {mode}")

        def do_state() -> None:
            pos, vel, torq = arm.get_state(request=True)
            print(
                f"{name}: pos={math.degrees(float(pos[idx])):+.4f}deg  "
                f"vel={math.degrees(float(vel[idx])):+.4f}deg/s  "
                f"torq={float(torq[idx]):+.4f}  mode={mode}"
            )

        def do_mit(values: list[str]) -> None:
            nonlocal mode, target_pos
            if not values:
                print("usage: mit <pos_deg> [vel_rad_s kp kd tau]")
                return
            if mode != "mit":
                do_mode(["mit"])
            target_pos[idx] = math.radians(float(values[0]))
            vel = np.zeros(arm.num_joints, dtype=float)
            tau = np.zeros(arm.num_joints, dtype=float)
            if len(values) > 1:
                vel[idx] = float(values[1])
            if len(values) > 2:
                mit_kp[idx] = float(values[2])
            if len(values) > 3:
                mit_kd[idx] = float(values[3])
            if len(values) > 4:
                tau[idx] = float(values[4])
            arm.set_targets(
                pos=target_pos.tolist(),
                vel=vel.tolist(),
                kp=mit_kp.tolist(),
                kd=mit_kd.tolist(),
                tau=tau.tolist(),
            )
            start_rt_if_needed()
            print(f"target {name}: {float(values[0]):+.2f} deg  kp={mit_kp[idx]:.2f} kd={mit_kd[idx]:.2f}")

        def do_posvel(values: list[str]) -> None:
            nonlocal mode, target_pos
            if not values:
                print("usage: posvel <pos_deg> [vlim_rad_s]")
                return
            if mode != "posvel":
                do_mode(["posvel"])
            target_pos[idx] = math.radians(float(values[0]))
            if len(values) > 1:
                pv_vlim[idx] = float(values[1])
            arm.set_targets(pos=target_pos.tolist(), vlim=pv_vlim.tolist())
            start_rt_if_needed()
            print(f"target {name}: {float(values[0]):+.2f} deg  vlim={pv_vlim[idx]:.3f} rad/s")

        def do_vel(values: list[str]) -> None:
            nonlocal mode
            if not values:
                print("usage: vel <vel_rad_s>")
                return
            if mode != "vel":
                do_mode(["vel"])
            vel = np.zeros(arm.num_joints, dtype=float)
            vel[idx] = float(values[0])
            arm.set_vel(vel.tolist())
            print(f"velocity {name}: {vel[idx]:+.3f} rad/s")

        commands = {
            "enable": do_enable,
            "disable": do_disable,
            "set_zero": do_set_zero,
            "mode": do_mode,
            "state": do_state,
            "mit": do_mit,
            "posvel": do_posvel,
            "vel": do_vel,
        }

        print("commands: enable / disable / set_zero / mode / mit / posvel / vel / state / q")
        print("examples: mit 10 0 20 2 0 | posvel 10 1.0 | vel 0.2")
        while True:
            try:
                line = input("> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n[exit]")
                break
            if not line:
                continue
            parts = line.split()
            cmd = parts[0].lower()
            values = parts[1:]
            if cmd in {"q", "quit", "exit"}:
                break
            fn = commands.get(cmd)
            if fn is None:
                print(f"unknown command: {cmd}")
                continue
            try:
                if cmd in {"enable", "disable", "set_zero", "state"}:
                    fn()
                else:
                    fn(values)
            except Exception as exc:
                print(f"error: {exc}")
    finally:
        try:
            arm.stop_control_loop()
            arm.disconnect()
        except Exception as exc:
            print(f"disconnect error: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
