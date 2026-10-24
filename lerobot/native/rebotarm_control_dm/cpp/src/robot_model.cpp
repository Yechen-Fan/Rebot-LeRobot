#include "rebotarm/robot_model.hpp"
#include "rebotarm/se3_conv.hpp"

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/spatial/explog.hpp>

#include <cmath>
#include <random>
#include <stdexcept>

namespace rebotarm {

RobotModel::RobotModel(const std::string& urdf_path) : urdf_path_(urdf_path) {
  pinocchio::urdf::buildModel(urdf_path, model_);
  data_ = pinocchio::Data(model_);
}

std::vector<std::string> RobotModel::joint_names() const {
  std::vector<std::string> out;
  for (std::size_t i = 1; i < model_.names.size(); ++i) {
    if (model_.joints[i].idx_q() >= 0) out.push_back(model_.names[i]);
  }
  return out;
}

std::vector<std::pair<double, double>> RobotModel::joint_limits() const {
  std::vector<std::pair<double, double>> out;
  for (std::size_t i = 1; i < model_.joints.size(); ++i) {
    const int idx_q = model_.joints[i].idx_q();
    if (idx_q < 0) continue;
    double lo = model_.lowerPositionLimit[idx_q];
    double hi = model_.upperPositionLimit[idx_q];
    if (!std::isfinite(lo) && !std::isfinite(hi)) {
      out.emplace_back(-std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::infinity());
    } else {
      out.emplace_back(lo, hi);
    }
  }
  return out;
}

std::vector<std::string> RobotModel::all_frame_names() const {
  std::vector<std::string> out;
  for (const auto& f : model_.frames) out.push_back(f.name);
  return out;
}

int RobotModel::frame_id(const std::string& name) const {
  return static_cast<int>(model_.getFrameId(name));
}

int RobotModel::end_effector_frame_id() const {
  return static_cast<int>(model_.getFrameId("end_link"));
}

Eigen::VectorXd RobotModel::neutral() const { return pinocchio::neutral(model_); }

void RobotModel::set_gravity(const Eigen::Vector3d& g) { model_.gravity.linear(g); }
Eigen::Vector3d RobotModel::get_gravity() const { return model_.gravity.linear(); }

Eigen::VectorXd RobotModel::random_configuration() const {
  // 与 inverse_kinematics.py 的随机重试一致：limit 无穷时退化为 [-pi, pi]。
  static thread_local std::mt19937 rng{std::random_device{}()};
  Eigen::VectorXd q(model_.nq);
  for (int j = 0; j < model_.nq; ++j) {
    double lo = model_.lowerPositionLimit[j];
    double hi = model_.upperPositionLimit[j];
    if (!std::isfinite(lo)) lo = -M_PI;
    if (!std::isfinite(hi)) hi = M_PI;
    std::uniform_real_distribution<double> d(lo, hi);
    q[j] = d(rng);
  }
  return q;
}

Eigen::VectorXd RobotModel::clamp_config(const Eigen::VectorXd& q) const {
  Eigen::VectorXd out = q;
  for (int j = 0; j < model_.nq; ++j) {
    double lo = std::isfinite(model_.lowerPositionLimit[j]) ? model_.lowerPositionLimit[j] : 0.0;
    double hi = std::isfinite(model_.upperPositionLimit[j]) ? model_.upperPositionLimit[j] : 0.0;
    out[j] = std::min(std::max(out[j], lo), hi);
  }
  return out;
}

std::tuple<Eigen::Vector3d, Eigen::Matrix3d, Eigen::Matrix4d>
RobotModel::fk(const Eigen::VectorXd& q, const std::string& frame_name) const {
  if (q.size() != model_.nq)
    throw std::invalid_argument("q 维度必须为 nq");
  pinocchio::forwardKinematics(model_, data_, q);
  pinocchio::updateFramePlacements(model_, data_);
  const auto fid = frame_name.empty() ? model_.getFrameId("end_link")
                                      : model_.getFrameId(frame_name);
  const pinocchio::SE3& oMf = data_.oMf[fid];
  return {oMf.translation(), oMf.rotation(), oMf.toHomogeneousMatrix()};
}

Eigen::MatrixXd RobotModel::frame_jacobian_local(const Eigen::VectorXd& q, int frame_id) const {
  pinocchio::computeJointJacobians(model_, data_, q);
  pinocchio::updateFramePlacements(model_, data_);
  pinocchio::Data::Matrix6x J(6, model_.nv);
  J.setZero();
  pinocchio::getFrameJacobian(model_, data_, static_cast<pinocchio::FrameIndex>(frame_id),
                              pinocchio::LOCAL, J);
  return J;
}

std::pair<double, Eigen::Matrix<double, 6, 1>>
RobotModel::compute_error(const Eigen::VectorXd& q, const pinocchio::SE3& target,
                          int end_frame_id) const {
  pinocchio::forwardKinematics(model_, data_, q);
  pinocchio::updateFramePlacements(model_, data_);
  const pinocchio::SE3& T_cur = data_.oMf[static_cast<pinocchio::FrameIndex>(end_frame_id)];
  Eigen::Matrix<double, 6, 1> err = pinocchio::log6(T_cur.inverse() * target).toVector();
  return {err.norm(), err};
}

IKResult RobotModel::solve_ik(const Eigen::Matrix4d& target_h, const Eigen::VectorXd& q_init,
                              int end_frame_id, const IKParams& params) const {
  const pinocchio::SE3 target = se3_from_homogeneous(target_h);
  Eigen::VectorXd q = q_init;
  auto [prev_err, err] = compute_error(q, target, end_frame_id);

  for (int iteration = 0; iteration < params.max_iter; ++iteration) {
    if (prev_err < params.tolerance)
      return IKResult{q, true, prev_err, iteration};

    pinocchio::computeJointJacobians(model_, data_, q);
    pinocchio::updateFramePlacements(model_, data_);
    pinocchio::Data::Matrix6x J(6, model_.nv);
    J.setZero();
    pinocchio::getFrameJacobian(model_, data_, static_cast<pinocchio::FrameIndex>(end_frame_id),
                                pinocchio::LOCAL, J);

    const double lam = params.damping * std::max(1.0, prev_err * 10.0);
    Eigen::MatrixXd JJT = J * J.transpose();
    JJT.diagonal().array() += lam;
    Eigen::VectorXd dq = params.step_size * J.transpose() * JJT.ldlt().solve(err);

    double alpha = 1.0;
    bool improved = false;
    for (int k = 0; k < 4; ++k) {
      Eigen::VectorXd q_new = clamp_config(pinocchio::integrate(model_, q, alpha * dq));
      auto [new_err, err_new] = compute_error(q_new, target, end_frame_id);
      if (new_err < prev_err) {
        q = q_new;
        err = err_new;
        prev_err = new_err;
        improved = true;
        break;
      }
      alpha *= 0.5;
    }
    (void)improved;  // 线搜索全失败时保持当前 q 继续（与 Python 一致）
  }
  return IKResult{q, false, prev_err, params.max_iter};
}

IKResult RobotModel::solve_ik_with_retry(const Eigen::Matrix4d& target, const Eigen::VectorXd& q_seed,
                                         int end_frame_id, const IKParams& params,
                                         int max_retries) const {
  IKResult best = solve_ik(target, q_seed, end_frame_id, params);
  if (best.success) return best;
  for (int r = 0; r < max_retries; ++r) {
    IKResult res = solve_ik(target, random_configuration(), end_frame_id, params);
    if (res.error < best.error) best = res;
    if (best.success) break;
  }
  return best;
}

}  // namespace rebotarm
