#include "rebotarm/dynamics.hpp"

#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>
#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <pinocchio/algorithm/energy.hpp>
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/centroidal.hpp>

namespace rebotarm {
namespace dyn {

namespace pin = pinocchio;

Eigen::MatrixXd mass_matrix(const RobotModel& rm, const Eigen::VectorXd& q) {
  pin::crba(rm.model(), rm.data(), q);
  rm.data().M.triangularView<Eigen::StrictlyLower>() =
      rm.data().M.transpose().triangularView<Eigen::StrictlyLower>();  // crba 只填上三角
  return rm.data().M;
}

Eigen::MatrixXd coriolis_matrix(const RobotModel& rm, const Eigen::VectorXd& q,
                                const Eigen::VectorXd& v) {
  pin::computeCoriolisMatrix(rm.model(), rm.data(), q, v);
  return rm.data().C;
}

Eigen::VectorXd gravity_vector(const RobotModel& rm, const Eigen::VectorXd& q) {
  pin::computeGeneralizedGravity(rm.model(), rm.data(), q);
  return rm.data().g;
}

Eigen::VectorXd nle(const RobotModel& rm, const Eigen::VectorXd& q, const Eigen::VectorXd& v) {
  pin::nonLinearEffects(rm.model(), rm.data(), q, v);
  return rm.data().nle;
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXd>
all_terms(const RobotModel& rm, const Eigen::VectorXd& q, const Eigen::VectorXd& v) {
  pin::computeAllTerms(rm.model(), rm.data(), q, v);
  Eigen::MatrixXd M = rm.data().M;
  M.triangularView<Eigen::StrictlyLower>() =
      M.transpose().triangularView<Eigen::StrictlyLower>();
  return {M, rm.data().C, rm.data().g};
}

Eigen::VectorXd inverse_dynamics(const RobotModel& rm, const Eigen::VectorXd& q,
                                 const Eigen::VectorXd& v, const Eigen::VectorXd& a) {
  pin::rnea(rm.model(), rm.data(), q, v, a);
  return rm.data().tau;
}

Eigen::VectorXd generalized_gravity(const RobotModel& rm, const Eigen::VectorXd& q) {
  pin::computeGeneralizedGravity(rm.model(), rm.data(), q);
  return rm.data().g;
}

Eigen::VectorXd static_torque(const RobotModel& rm, const Eigen::VectorXd& q) {
  // 零外力下静力矩即广义重力。
  pin::computeGeneralizedGravity(rm.model(), rm.data(), q);
  return rm.data().g;
}

Eigen::VectorXd forward_dynamics(const RobotModel& rm, const Eigen::VectorXd& q,
                                 const Eigen::VectorXd& v, const Eigen::VectorXd& tau) {
  pin::aba(rm.model(), rm.data(), q, v, tau);
  return rm.data().ddq;
}

Eigen::VectorXd forward_dynamics_from_nle(const RobotModel& rm, const Eigen::VectorXd& q,
                                          const Eigen::VectorXd& v, const Eigen::VectorXd& tau) {
  pin::computeAllTerms(rm.model(), rm.data(), q, v);
  Eigen::MatrixXd M = rm.data().M;
  M.triangularView<Eigen::StrictlyLower>() =
      M.transpose().triangularView<Eigen::StrictlyLower>();
  return M.ldlt().solve(tau - rm.data().nle);
}

double kinetic_energy(const RobotModel& rm, const Eigen::VectorXd& q, const Eigen::VectorXd& v) {
  return pin::computeKineticEnergy(rm.model(), rm.data(), q, v);
}

double potential_energy(const RobotModel& rm, const Eigen::VectorXd& q) {
  return pin::computePotentialEnergy(rm.model(), rm.data(), q);
}

double total_energy(const RobotModel& rm, const Eigen::VectorXd& q, const Eigen::VectorXd& v) {
  return kinetic_energy(rm, q, v) + potential_energy(rm, q);
}

Eigen::Vector3d center_of_mass(const RobotModel& rm, const Eigen::VectorXd& q, bool center_zero) {
  pin::centerOfMass(rm.model(), rm.data(), q, !center_zero ? true : false);
  return rm.data().com[0];
}

Eigen::Vector3d com_velocity(const RobotModel& rm, const Eigen::VectorXd& q,
                             const Eigen::VectorXd& v) {
  // C++ 无 computeCentroidalVelocities；centerOfMass(q,v) 会填充 data.vcom[0]。
  pin::centerOfMass(rm.model(), rm.data(), q, v);
  return rm.data().vcom[0];
}

Eigen::Matrix<double, 6, 1> centroidal_momentum(const RobotModel& rm, const Eigen::VectorXd& q,
                                                const Eigen::VectorXd& v) {
  pin::ccrba(rm.model(), rm.data(), q, v);
  return rm.data().hg.toVector();
}

Eigen::MatrixXd centroidal_matrix(const RobotModel& rm, const Eigen::VectorXd& q,
                                  const Eigen::VectorXd& v) {
  pin::ccrba(rm.model(), rm.data(), q, v);
  return rm.data().Ag;
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>
rnea_derivatives(const RobotModel& rm, const Eigen::VectorXd& q, const Eigen::VectorXd& v,
                 const Eigen::VectorXd& a) {
  pin::computeRNEADerivatives(rm.model(), rm.data(), q, v, a);
  // dtau_da 即质量矩阵（上三角），补全为对称。
  Eigen::MatrixXd dtau_da = rm.data().M;
  dtau_da.triangularView<Eigen::StrictlyLower>() =
      dtau_da.transpose().triangularView<Eigen::StrictlyLower>();
  return {rm.data().dtau_dq, rm.data().dtau_dv, dtau_da};
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd>
coriolis_derivatives(const RobotModel& rm, const Eigen::VectorXd& q, const Eigen::VectorXd& v) {
  Eigen::VectorXd zero = Eigen::VectorXd::Zero(rm.nv());
  pin::computeRNEADerivatives(rm.model(), rm.data(), q, v, zero);
  return {rm.data().dtau_dq, rm.data().dtau_dv};
}

Eigen::MatrixXd generalized_gravity_derivatives(const RobotModel& rm, const Eigen::VectorXd& q) {
  Eigen::VectorXd zero = Eigen::VectorXd::Zero(rm.nv());
  pin::computeRNEADerivatives(rm.model(), rm.data(), q, zero, zero);
  return rm.data().dtau_dq;
}

std::vector<Eigen::MatrixXd> mass_matrix_derivatives(const RobotModel& rm,
                                                     const Eigen::VectorXd& q) {
  // dM/dq_j 中心差分（Pinocchio C++ 无单关节版 API；数值等价、与 Python 同语义）。
  const int nq = rm.nq();
  const double eps = 1e-6;
  std::vector<Eigen::MatrixXd> out;
  out.reserve(nq);
  for (int j = 0; j < nq; ++j) {
    Eigen::VectorXd qp = q, qm = q;
    qp[j] += eps;
    qm[j] -= eps;
    Eigen::MatrixXd Mp = mass_matrix(rm, qp);
    Eigen::MatrixXd Mm = mass_matrix(rm, qm);
    out.emplace_back((Mp - Mm) / (2.0 * eps));
  }
  return out;
}

}  // namespace dyn
}  // namespace rebotarm
