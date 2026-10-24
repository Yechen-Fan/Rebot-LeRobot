"""rebotarm_control_dm - reBotArm 机械臂控制库。

底层全部原生实现，Python 仅作上层接口：
  - actuator（电机控制）  → Rust，直接调用 motorbridge vendor crates（_native）
  - kinematics/dynamics/trajectory/controllers → C++，直接调用 Pinocchio C++（_math）

接口与 reBotArm_control_py 对齐，可直接替换。
"""
from ._runtime import ensure_compatible_libstdcpp

ensure_compatible_libstdcpp()

from . import actuator
from . import kinematics
from . import dynamics
from . import trajectory
from . import controllers

__all__ = ["actuator", "kinematics", "dynamics", "trajectory", "controllers"]

try:
    from ._native import __version__  # type: ignore
except Exception:  # pragma: no cover
    __version__ = "0.1.0"
