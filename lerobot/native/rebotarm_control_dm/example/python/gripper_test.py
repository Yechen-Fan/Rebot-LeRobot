#!/usr/bin/env python3
"""Interactive gripper test tool."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

SOURCE_PYTHON = Path(__file__).resolve().parents[2] / "python"
if SOURCE_PYTHON.exists() and str(SOURCE_PYTHON) not in sys.path:
    sys.path.insert(0, str(SOURCE_PYTHON))

from rebotarm_control_dm.actuator import Gripper
from _example_config import add_port_argument, config_with_port

HELP = """
Gripper commands
----------------
z  - set current position as zero
m  - switch mode: MIT / POS_VEL / VEL
c  - send/update control command
s  - show current state
h  - show help
q  - stop loop, disable, disconnect
"""


class GripperTerminal:
    def __init__(self, cfg_path: str | None, rate: float) -> None:
        self.g = Gripper(cfg_path)
        self.g.enable()
        print(f"enabled, mode: {self.g.mode}")
        self._show_state()

        self._target_pos = self.g.get_position(request=True)
        self._target_vel = 0.0
        self._mit_tau = 0.0
        self._running = True

        self.g.start_control_loop(self._loop, rate=rate)
        print(f"control loop started @ {rate} Hz")

    def _loop(self, gripper, dt: float) -> None:
        if self.g.mode == "mit":
            self.g.mit(pos=self._target_pos, vel=self._target_vel, tau=self._mit_tau)
        elif self.g.mode == "pos_vel":
            self.g.pos_vel(pos=self._target_pos)
        elif self.g.mode == "vel":
            self.g.set_vel(vel=self._target_vel)

    def _show_state(self) -> None:
        pos, vel, torq = self.g.get_state(request=True)
        print(
            f"  pos={pos:+.4f} rad ({math.degrees(pos):+.2f} deg)  "
            f"vel={vel:+.4f} rad/s  torq={torq:+.4f} Nm  [mode={self.g.mode}]"
        )

    def run(self) -> None:
        print(HELP)
        while self._running:
            try:
                cmd = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                cmd = "q"

            if not cmd:
                continue

            if cmd == "q":
                print("stopping loop, disabling, disconnecting...")
                self.g.stop_control_loop()
                self.g.disable()
                self.g.disconnect()
                self._running = False
                break

            if cmd == "h":
                print(HELP)
            elif cmd == "s":
                self._show_state()
            elif cmd == "z":
                answer = input("Set current gripper position as zero? Type YES: ").strip()
                if answer == "YES":
                    self.g.set_zero()
            elif cmd == "m":
                print(f"current mode: {self.g.mode}; switch to: [0]MIT  [1]POS_VEL  [2]VEL")
                sel = input("  > ").strip()
                if sel == "0":
                    self.g.mode_mit()
                    print("mode: MIT")
                elif sel == "1":
                    self.g.mode_pos_vel()
                    print("mode: POS_VEL")
                elif sel == "2":
                    self.g.mode_vel()
                    print("mode: VEL")
                else:
                    print("invalid selection")
            elif cmd == "c":
                if self.g.mode == "mit":
                    try:
                        self._target_pos = float(input("  pos (rad): ").strip() or self._target_pos)
                        self._target_vel = float(input("  vel (rad/s) [0.0]: ").strip() or "0.0")
                        self._mit_tau = float(input("  tau (Nm) [0.0]: ").strip() or "0.0")
                    except ValueError:
                        print("invalid input")
                elif self.g.mode == "pos_vel":
                    try:
                        self._target_pos = float(input("  pos (rad): ").strip() or self._target_pos)
                    except ValueError:
                        print("invalid input")
                elif self.g.mode == "vel":
                    try:
                        self._target_vel = float(input("  vel (rad/s): ").strip() or "0.0")
                    except ValueError:
                        print("invalid input")
                self._show_state()
            else:
                print(f"unknown command: {cmd}; press h for help")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "-c", default=None, help="Path to gripper YAML config.")
    add_port_argument(parser)
    parser.add_argument("--rate", type=float, default=100.0, help="Python gripper loop rate.")
    args = parser.parse_args()
    GripperTerminal(config_with_port(args.config, args.port, gripper=True), args.rate).run()


if __name__ == "__main__":
    main()
