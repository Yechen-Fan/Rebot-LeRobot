"""Dynamics 动力学库 —— 由 C++ (_math, Pinocchio) 实现，Python 仅 re-export。

接口与 reBotArm_control_py.dynamics 对齐；各计算函数以 RobotModel 为第一参。
"""
import numpy as np

from rebotarm_control_dm._math import (
    mass_matrix as compute_mass_matrix,
    coriolis_matrix as compute_coriolis_matrix,
    gravity_vector as compute_gravity_vector,
    nle as compute_nle,
    all_terms as compute_all_terms,
    inverse_dynamics as compute_inverse_dynamics,
    generalized_gravity as compute_generalized_gravity,
    static_torque as compute_static_torque,
    forward_dynamics as compute_forward_dynamics,
    forward_dynamics_from_nle,
    kinetic_energy as compute_kinetic_energy,
    potential_energy as compute_potential_energy,
    total_energy as compute_total_energy,
    center_of_mass as compute_center_of_mass,
    com_velocity as compute_com_velocity,
    centroidal_momentum as compute_centroidal_momentum,
    centroidal_matrix as compute_centroidal_matrix,
    rnea_derivatives as compute_rnea_derivatives,
    coriolis_derivatives as compute_coriolis_derivatives,
    generalized_gravity_derivatives as compute_generalized_gravity_derivatives,
    mass_matrix_derivatives as compute_mass_matrix_derivatives,
)
from rebotarm_control_dm.kinematics import load_robot_model

# 动力学模型与运动学共享同一 URDF 入口。
load_dynamics_model = load_robot_model

EARTH_GRAVITY = (0.0, 0.0, -9.81)
ZERO_GRAVITY = (0.0, 0.0, 0.0)


def get_default_gravity() -> np.ndarray:
    return np.array(EARTH_GRAVITY)


def set_gravity(model, gravity) -> None:
    model.set_gravity(np.asarray(gravity, dtype=float))


def get_gravity(model) -> np.ndarray:
    return np.asarray(model.get_gravity())


__all__ = [
    "load_dynamics_model",
    "get_default_gravity",
    "set_gravity",
    "get_gravity",
    "compute_mass_matrix",
    "compute_coriolis_matrix",
    "compute_gravity_vector",
    "compute_nle",
    "compute_all_terms",
    "compute_forward_dynamics",
    "forward_dynamics_from_nle",
    "compute_inverse_dynamics",
    "compute_generalized_gravity",
    "compute_static_torque",
    "compute_mass_matrix_derivatives",
    "compute_coriolis_derivatives",
    "compute_rnea_derivatives",
    "compute_generalized_gravity_derivatives",
    "compute_kinetic_energy",
    "compute_potential_energy",
    "compute_total_energy",
    "compute_center_of_mass",
    "compute_com_velocity",
    "compute_centroidal_matrix",
    "compute_centroidal_momentum",
]
