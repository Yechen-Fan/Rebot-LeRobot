"""Kinematics 运动学库 —— 由 C++ (_math, Pinocchio) 实现，Python 仅 re-export。

接口与 reBotArm_control_py.kinematics 对齐。位姿在 Python 边界统一为 4×4 numpy 齐次矩阵
（不再暴露 pinocchio.SE3）。
"""
from rebotarm_control_dm._math import (
    RobotModel,
    IKParams,
    IKResult,
    compute_fk,
    joint_to_pose,
    compute_ik,
    pos_rot_to_se3,
    get_joint_names,
    get_joint_limits,
    get_frame_id,
    get_end_effector_frame_id,
    get_all_frame_names,
    load_robot_model as _load_robot_model_native,
)
from rebotarm_control_dm.paths import default_urdf_path, resolve_urdf_path

# 与 inverse_kinematics.py 的别名一致。
IKSolverParams = IKParams

_URDF = str(default_urdf_path())


def load_robot_model(urdf_path: str | None = None) -> RobotModel:
    """构建 reBot-DevArm 模型。

    ``urdf_path`` 为空时使用原始项目 URDF。传入文件名（例如
    ``"tool_calibration.urdf"``）且当前目录不存在该文件时，会自动到
    ``calibration/`` 目录下查找同名 URDF。
    """
    return _load_robot_model_native(str(resolve_urdf_path(urdf_path)))


__all__ = [
    "RobotModel",
    "load_robot_model",
    "get_joint_names",
    "get_joint_limits",
    "get_frame_id",
    "get_end_effector_frame_id",
    "get_all_frame_names",
    "compute_fk",
    "joint_to_pose",
    "compute_ik",
    "pos_rot_to_se3",
    "IKParams",
    "IKSolverParams",
    "IKResult",
]
