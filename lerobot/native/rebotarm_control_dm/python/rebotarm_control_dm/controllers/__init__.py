"""控制器封装层 —— ArmEndPos 编排（C++ 计算 + 驱动 Rust actuator）。

Python 仅注入默认 URDF 路径后转交 C++ 实现。
"""
from rebotarm_control_dm._math import ArmEndPos as _ArmEndPos, TrajProfile
from rebotarm_control_dm.paths import default_urdf_path

_URDF = str(default_urdf_path())


class ArmEndPos(_ArmEndPos):
    """末端位置控制器。cfg/urdf 默认取包内 URDF。

    用法::
        arm = RobotArm()
        with ArmEndPos(arm) as ep:
            ep.move_to_ik(x=0.3, y=0.0, z=0.3)
            ep.move_to_traj(x=0.3, y=0.0, z=0.3, pitch=0.4, duration=2.0)
    """

    def __init__(self, arm, dt: float = 0.02, profile=TrajProfile.MIN_JERK) -> None:
        super().__init__(arm, _URDF, dt, profile)


__all__ = ["ArmEndPos"]
