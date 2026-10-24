"""轨迹规划包 —— 由 C++ (_math) 实现，Python 仅 re-export。

SE(3) 测地线采样 + CLIK 关节空间跟踪。接口与 reBotArm_control_py.trajectory 对齐。
"""
from rebotarm_control_dm._math import (
    TrajProfile,
    TrajPlanParams,
    CartesianPoint,
    CartesianTrajectory,
    CartesianTrajectoryResult,
    plan_cartesian_geodesic_trajectory,
    CLIKParams,
    JointTrajectoryPoint,
    track_trajectory,
    TrajStats,
    plan_joint_space_trajectory,
    compute_traj_stats,
)

# 向后兼容别名（clik_tracker.py 中 IKParams 即 CLIK 参数）。
IKParams = CLIKParams

__all__ = [
    "TrajProfile",
    "TrajPlanParams",
    "CartesianPoint",
    "CartesianTrajectory",
    "CartesianTrajectoryResult",
    "plan_cartesian_geodesic_trajectory",
    "CLIKParams",
    "IKParams",
    "JointTrajectoryPoint",
    "track_trajectory",
    "TrajStats",
    "plan_joint_space_trajectory",
    "compute_traj_stats",
]
