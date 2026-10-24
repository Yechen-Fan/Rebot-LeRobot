#!/usr/bin/env python3
"""ArmEndPos 交互控制示例（轨迹规划模式）。

用法:
    python example/python/8_arm_traj_control.py --port /dev/ttyACM0 [--config arm.yaml]

输入:
    x y z [roll pitch yaw] [duration]   目标末端位置（米 / 弧度 / 秒）
    q / quit / exit                     退出
    state                               当前状态
    pos / end_state                     当前末端位置

退出时 ArmEndPos.end() 会先 safe_home，再断开连接。
"""

import argparse
import sys
from pathlib import Path

SOURCE_PYTHON = Path(__file__).resolve().parents[2] / "python"
if SOURCE_PYTHON.exists() and str(SOURCE_PYTHON) not in sys.path:
    sys.path.insert(0, str(SOURCE_PYTHON))

from rebotarm_control_dm.actuator import RobotArm
from rebotarm_control_dm.controllers import ArmEndPos
from rebotarm_control_dm.kinematics import joint_to_pose, load_robot_model
from _example_config import add_port_argument, config_with_port


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "-c", default=None, help="Path to arm YAML config.")
    add_port_argument(parser)
    args = parser.parse_args()

    arm = RobotArm(config_with_port(args.config, args.port))
    arm_endpos_control = ArmEndPos(arm)
    model = load_robot_model()

    arm_endpos_control.start()
    print("--- 已启动末端位置控制器 ---\n")

    try:
        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                break

            if not line:
                continue
            if line.lower() in ("q", "quit", "exit"):
                break

            if line.lower() == "state":
                q, _, _ = arm.get_state()
                print(f"  当前关节 (rad): {[f'{v:+.3f}' for v in q]}")
                print(f"  overruns send/read: {arm.rt_send_overruns}/{arm.rt_read_overruns}")
                continue

            if line.lower() in ("pos", "end_state"):
                q, _, _ = arm.get_state()
                pos, rpy = joint_to_pose(model, q)
                px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
                rx, ry, rz = float(rpy[0]), float(rpy[1]), float(rpy[2])
                print(f"  pos=[{px:+.3f} {py:+.3f} {pz:+.3f}] m  rpy=[{rx:+.2f} {ry:+.2f} {rz:+.2f}] rad")
                continue

            try:
                vals = [float(v) for v in line.split()]
            except ValueError:
                print("  格式: x y z [roll pitch yaw] [duration]")
                continue

            if len(vals) not in (3, 6, 7):
                print("  格式: x y z [roll pitch yaw] [duration]")
                continue

            x, y, z = vals[0], vals[1], vals[2]
            roll = vals[3] if len(vals) >= 6 else 0.0
            pitch = vals[4] if len(vals) >= 6 else 0.0
            yaw = vals[5] if len(vals) >= 6 else 0.0
            duration = vals[6] if len(vals) >= 7 else 2.0

            ok = arm_endpos_control.move_to_traj(
                x=x, y=y, z=z,
                roll=roll, pitch=pitch, yaw=yaw,
                duration=duration,
            )
            print(f"  -> ({x:+.3f}, {y:+.3f}, {z:+.3f})  "
                  f"T={duration:.1f}s  {'ok' if ok else 'fail'}")
    finally:
        arm_endpos_control.end()
    print("\n完成。")


if __name__ == "__main__":
    main()
