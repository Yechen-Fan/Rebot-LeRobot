#!/usr/bin/env python3
"""Read Damiao POS_VEL gains from motor registers.

Damiao POS_VEL gain registers:
  25: vel_kp / KP_ASR
  26: vel_ki / KI_ASR
  27: pos_kp / KP_APR
  28: pos_ki / KI_APR
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

SOURCE_PYTHON = Path(__file__).resolve().parents[2] / "python"
if SOURCE_PYTHON.exists() and str(SOURCE_PYTHON) not in sys.path:
    sys.path.insert(0, str(SOURCE_PYTHON))

from rebotarm_control_dm.actuator import RobotArm


B601_JOINTS = [
    ("shoulder_pan", 0x01, 0x11, "4340P"),
    ("shoulder_lift", 0x02, 0x12, "4340P"),
    ("elbow_flex", 0x03, 0x13, "4340P"),
    ("wrist_flex", 0x04, 0x14, "4310"),
    ("wrist_yaw", 0x05, 0x15, "4310"),
    ("wrist_roll", 0x06, 0x16, "4310"),
    ("gripper", 0x07, 0x17, "4310"),
]


def make_b601_config(port: str, path: Path) -> Path:
    lines = [
        "# Temporary config for reading Damiao registers.",
        "name: reBotArmB601DMRegisterRead",
        f"channel: {port}",
        "rate: 150.0",
        "",
        "joints:",
    ]
    for name, motor_id, feedback_id, model in B601_JOINTS:
        lines.extend(
            [
                f"  - name: {name}",
                f"    motor_id: 0x{motor_id:02X}",
                f"    feedback_id: 0x{feedback_id:02X}",
                f"    model: \"{model}\"",
                "    vendor: \"damiao\"",
                "    POS_VEL:",
                "      vlim: 2.0",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def read_one(config: str | None, label: str, timeout_ms: int) -> None:
    arm = RobotArm(config)
    try:
        arm.connect()
        print(f"\n[{label}] {arm.name}")
        print(f"{'joint':<16} {'vel_kp':>10} {'vel_ki':>10} {'pos_kp':>10} {'pos_ki':>10}")
        gains = arm.get_pos_vel_gains(timeout_ms=timeout_ms)
        for name in arm.joint_names:
            vel_kp, vel_ki, pos_kp, pos_ki = gains[name]
            print(f"{name:<16} {vel_kp:10.6g} {vel_ki:10.6g} {pos_kp:10.6g} {pos_ki:10.6g}")
    finally:
        arm.disconnect(disable=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        "-c",
        action="append",
        default=None,
        help="Arm YAML config. Can be passed multiple times for left/right arms.",
    )
    parser.add_argument("--timeout-ms", type=int, default=300, help="Per-register read timeout.")
    parser.add_argument("--port", action="append", default=None, help="Port to read. Can be passed multiple times.")
    parser.add_argument("--left-port", default="/dev/ttyACM0", help="Default-bi left arm port.")
    parser.add_argument("--right-port", default="/dev/ttyACM1", help="Default-bi right arm port.")
    parser.add_argument(
        "--default-bi",
        action="store_true",
        help="Read B601 registers from --left-port and --right-port without requiring cached yaml.",
    )
    args = parser.parse_args()

    configs: list[tuple[str | None, str]] = []
    if args.config:
        configs.extend((config, config) for config in args.config)

    ports: list[tuple[str, str]] = []
    if args.default_bi:
        ports.extend([("left", args.left_port), ("right", args.right_port)])
    if args.port:
        ports.extend((f"port{idx}", port) for idx, port in enumerate(args.port, start=1))

    if ports:
        with tempfile.TemporaryDirectory(prefix="rebotarm-pd-read-") as tmp_dir:
            tmp_path = Path(tmp_dir)
            for label, port in ports:
                config_path = make_b601_config(port, tmp_path / f"{label}.yaml")
                configs.append((str(config_path), f"{label} {port}"))

            for config, label in configs:
                read_one(config, label, args.timeout_ms)
        return

    if not configs:
        configs = [(None, "package default arm.yaml")]

    for config, label in configs:
        read_one(config, label, args.timeout_ms)


if __name__ == "__main__":
    main()
