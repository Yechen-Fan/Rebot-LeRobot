"""actuator 模块 - 由 Rust 原生实现（直接调用 motorbridge vendor crates）。

接口与 reBotArm_control_py.actuator 一致：

    from rebotarm_control_dm.actuator import RobotArm

    arm = RobotArm()            # 默认读取包内 config/arm.yaml
    arm.connect()
    arm.disable()
    arm.set_zero()
    arm.enable()
    arm.mode_mit()
    arm.mit(pos=np.array([...]), kp=np.array([...]), kd=np.array([...]))

    arm.mode_pos_vel()
    arm.pos_vel(pos=np.array([...]))

    arm.mode_vel()
    arm.set_vel(vel=np.array([...]))

    arm.stop_control_loop()
    arm.disconnect()

RT 原生控制循环（GIL 全程释放，真正实时）：

    arm.mode_mit()
    arm.set_targets(pos=np.zeros(6))
    arm.start_rt_loop()         # Rust 后台线程按 set_targets 的目标驱动电机
    ...
    arm.stop_control_loop()
"""
from pathlib import Path

from rebotarm_control_dm import _native
from rebotarm_control_dm._native import JointCfg, GripperCfg

# 这个路径在 ./rebotarm_control_dm/python/rebotarm_control_dm/config/arm.yaml
_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def _append_serial_help(exc: Exception, cfg_path: str) -> Exception:
    msg = str(exc)
    if "open serial port" not in msg:
        return exc
    help_msg = (
        f"{msg}\n\n"
        "Serial port troubleshooting:\n"
        "  1. List current devices:\n"
        "       ls -l /dev/ttyACM* /dev/ttyUSB*\n"
        "       ls -l /dev/serial/by-id/\n"
        f"  2. Edit the YAML config used by this run:\n"
        f"       {cfg_path}\n"
        "     and set channel to the actual Damiao bridge, for example:\n"
        "       channel: /dev/ttyACM1    # replace with the port shown by ls\n"
        "  3. If the error is 'Device or resource busy', another process is still holding the port.\n"
        "     Stop the old process or check it with:\n"
        "       fuser -v /dev/ttyACM1    # replace with the busy port\n"
        "       lsof /dev/ttyACM1\n"
        "  4. If permissions are denied, allow access to the actual port:\n"
        "       sudo chmod 666 /dev/ttyACM1\n"
    )
    return ValueError(help_msg)


# 注意：PyO3 #[new] 对应 __new__（而非 __init__），构造发生在 __new__。
# 因此默认配置路径必须在 __new__ 注入；__init__ 仅吞掉参数避免 TypeError。

class RobotArm(_native.RobotArm):
    """机械臂控制句柄。cfg_path 为空时默认读取包内 config/arm.yaml。"""

    def __new__(cls, cfg_path: str | None = None):
        if cfg_path is None:
            cfg_path = str(_CONFIG_DIR / "arm.yaml")
        try:
            return super().__new__(cls, cfg_path)
        except ValueError as exc:
            raise _append_serial_help(exc, cfg_path) from exc

    def __init__(self, cfg_path: str | None = None) -> None:  # noqa: D401
        pass


class Gripper(_native.Gripper):
    """夹爪控制句柄。cfg_path 为空时默认读取包内 config/gripper.yaml。"""

    def __new__(cls, cfg_path: str | None = None):
        if cfg_path is None:
            cfg_path = str(_CONFIG_DIR / "gripper.yaml")
        try:
            return super().__new__(cls, cfg_path)
        except ValueError as exc:
            raise _append_serial_help(exc, cfg_path) from exc

    def __init__(self, cfg_path: str | None = None) -> None:
        pass


def load_cfg(path: str | None = None) -> dict:
    if path is None:
        path = str(_CONFIG_DIR / "arm.yaml")
    return _native.load_cfg(path)


def load_gripper_cfg(path: str | None = None) -> dict:
    if path is None:
        path = str(_CONFIG_DIR / "gripper.yaml")
    return _native.load_gripper_cfg(path)


__all__ = [
    "RobotArm",
    "JointCfg",
    "load_cfg",
    "Gripper",
    "GripperCfg",
    "load_gripper_cfg",
]
