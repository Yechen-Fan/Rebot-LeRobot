// 动力学自由函数（对照 reBotArm_control_py/dynamics/*.py），均以 RobotModel 为第一参。
#pragma once
#include "rebotarm/robot_model.hpp"
#include <Eigen/Dense>
#include <tuple>
#include <vector>

namespace rebotarm {
namespace dyn {

Eigen::MatrixXd mass_matrix(const RobotModel& rm, const Eigen::VectorXd& q);                 // crba
Eigen::MatrixXd coriolis_matrix(const RobotModel& rm, const Eigen::VectorXd& q,
                                const Eigen::VectorXd& v);
Eigen::VectorXd gravity_vector(const RobotModel& rm, const Eigen::VectorXd& q);              // computeGeneralizedGravity
Eigen::VectorXd nle(const RobotModel& rm, const Eigen::VectorXd& q, const Eigen::VectorXd& v);
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXd>
all_terms(const RobotModel& rm, const Eigen::VectorXd& q, const Eigen::VectorXd& v);

Eigen::VectorXd inverse_dynamics(const RobotModel& rm, const Eigen::VectorXd& q,
                                 const Eigen::VectorXd& v, const Eigen::VectorXd& a);          // rnea
Eigen::VectorXd generalized_gravity(const RobotModel& rm, const Eigen::VectorXd& q);
Eigen::VectorXd static_torque(const RobotModel& rm, const Eigen::VectorXd& q);                // 零外力 == 重力
Eigen::VectorXd forward_dynamics(const RobotModel& rm, const Eigen::VectorXd& q,
                                 const Eigen::VectorXd& v, const Eigen::VectorXd& tau);        // aba
Eigen::VectorXd forward_dynamics_from_nle(const RobotModel& rm, const Eigen::VectorXd& q,
                                          const Eigen::VectorXd& v, const Eigen::VectorXd& tau);

double kinetic_energy(const RobotModel& rm, const Eigen::VectorXd& q, const Eigen::VectorXd& v);
double potential_energy(const RobotModel& rm, const Eigen::VectorXd& q);
double total_energy(const RobotModel& rm, const Eigen::VectorXd& q, const Eigen::VectorXd& v);

Eigen::Vector3d center_of_mass(const RobotModel& rm, const Eigen::VectorXd& q, bool center_zero = false);
Eigen::Vector3d com_velocity(const RobotModel& rm, const Eigen::VectorXd& q, const Eigen::VectorXd& v);
Eigen::Matrix<double, 6, 1> centroidal_momentum(const RobotModel& rm, const Eigen::VectorXd& q,
                                                 const Eigen::VectorXd& v);                   // ccrba -> hg
Eigen::MatrixXd centroidal_matrix(const RobotModel& rm, const Eigen::VectorXd& q,
                                  const Eigen::VectorXd& v);                                  // ccrba -> Ag

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>
rnea_derivatives(const RobotModel& rm, const Eigen::VectorXd& q, const Eigen::VectorXd& v,
                 const Eigen::VectorXd& a);
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd>
coriolis_derivatives(const RobotModel& rm, const Eigen::VectorXd& q, const Eigen::VectorXd& v);
Eigen::MatrixXd generalized_gravity_derivatives(const RobotModel& rm, const Eigen::VectorXd& q);
// dM/dq_j（中心差分，返回长度 nq 的 nv×nv 矩阵序列；Python 版本同语义）。
std::vector<Eigen::MatrixXd> mass_matrix_derivatives(const RobotModel& rm, const Eigen::VectorXd& q);

}  // namespace dyn
}  // namespace rebotarm
