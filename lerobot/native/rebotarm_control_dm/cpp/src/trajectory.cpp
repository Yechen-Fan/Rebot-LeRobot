#include "rebotarm/trajectory.hpp"
#include "rebotarm/se3_conv.hpp"

#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/spatial/explog.hpp>

#include <algorithm>
#include <cmath>

namespace rebotarm {
namespace traj {

namespace pin = pinocchio;

static double apply_profile(double t, TrajProfile profile, double accel_ratio) {
  t = std::max(0.0, std::min(1.0, t));
  switch (profile) {
    case TrajProfile::LINEAR:
      return t;
    case TrajProfile::MIN_JERK: {
      double t3 = t * t * t, t4 = t3 * t, t5 = t4 * t;
      return 10.0 * t3 - 15.0 * t4 + 6.0 * t5;
    }
    case TrajProfile::TRAPEZOID: {
      double ta = std::max(0.01, std::min(0.49, accel_ratio));
      double vm = 2.0 / (1.0 - ta);
      if (t <= ta) return 0.5 * vm / ta * t * t;
      if (t <= 1.0 - ta) return 0.5 * vm * ta + vm * (t - ta);
      double dt = 1.0 - t;
      return 1.0 - 0.5 * vm / ta * dt * dt;
    }
  }
  return t;
}

static pin::SE3 se3_interpolate(const pin::SE3& a, const pin::SE3& b, double s) {
  return a * pin::exp6(pin::log6(a.actInv(b)) * s);
}

CartesianTrajectoryResult plan_cartesian_geodesic_trajectory(const Eigen::Matrix4d& start_pose,
                                                             const Eigen::Matrix4d& end_pose,
                                                             double duration,
                                                             const TrajPlanParams& params) {
  if (duration <= 0.0) throw std::invalid_argument("duration 必须 > 0");
  const pin::SE3 a = se3_from_homogeneous(start_pose);
  const pin::SE3 b = se3_from_homogeneous(end_pose);

  CartesianTrajectoryResult res;
  int n = std::max(2, static_cast<int>(std::ceil(duration / params.dt)) + 1);
  double dt = duration / (n - 1);
  res.trajectory.points.reserve(n);
  for (int i = 0; i < n; ++i) {
    double t = i * dt;
    double s = apply_profile(t / duration, params.profile, params.accel_ratio);
    CartesianPoint p;
    p.time = t;
    p.pose = se3_interpolate(a, b, s).toHomogeneousMatrix();
    res.trajectory.points.push_back(std::move(p));
  }
  res.n_points = n;
  return res;
}

static Eigen::VectorXd joint_limit_grad(const pin::Model& model, const Eigen::VectorXd& q) {
  Eigen::VectorXd g = Eigen::VectorXd::Zero(model.nv);
  for (int i = 0; i < model.nv; ++i) {
    double lo = model.lowerPositionLimit[i], hi = model.upperPositionLimit[i];
    if (!std::isfinite(lo) || !std::isfinite(hi)) continue;
    double dl = q[i] - lo, dh = hi - q[i];
    if (dl > 1e-6 && dh > 1e-6) g[i] = (dh - dl) / (dl * dh);
  }
  return g;
}

static Eigen::VectorXd clamp_config(const pin::Model& model, const Eigen::VectorXd& q) {
  Eigen::VectorXd qc = q;
  for (int i = 0; i < model.nq; ++i) {
    double lo = std::isfinite(model.lowerPositionLimit[i]) ? model.lowerPositionLimit[i] : 0.0;
    double hi = std::isfinite(model.upperPositionLimit[i]) ? model.upperPositionLimit[i] : 0.0;
    if (std::isfinite(q[i]) && lo <= hi) qc[i] = std::min(std::max(q[i], lo), hi);
  }
  return qc;
}

std::vector<JointTrajectoryPoint> track_trajectory(const RobotModel& rm, int end_frame_id,
                                                   const CartesianTrajectory& traj,
                                                   const Eigen::VectorXd& q_init,
                                                   const CLIKParams& ik, double null_gain) {
  const pin::Model& model = rm.model();
  pin::Data& data = rm.data();
  Eigen::VectorXd q = q_init;
  std::vector<JointTrajectoryPoint> out;
  out.reserve(traj.points.size());

  for (const auto& pt : traj.points) {
    const pin::SE3 target = se3_from_homogeneous(pt.pose);
    bool converged = false;
    for (int it = 0; it < ik.max_iter; ++it) {
      pin::computeJointJacobians(model, data, q);
      pin::updateFramePlacements(model, data);
      const pin::SE3& oMf = data.oMf[static_cast<pin::FrameIndex>(end_frame_id)];
      Eigen::Matrix<double, 6, 1> err = pin::log6(oMf.actInv(target)).toVector();
      double err_norm = err.norm();
      if (err_norm < ik.tolerance) {
        converged = true;
        break;
      }
      pin::Data::Matrix6x J(6, model.nv);
      J.setZero();
      pin::getFrameJacobian(model, data, static_cast<pin::FrameIndex>(end_frame_id),
                            pin::LOCAL, J);
      double lam = ik.damping * std::max(1.0, err_norm * 10.0);
      Eigen::MatrixXd JJT = J * J.transpose();
      JJT.diagonal().array() += lam;
      Eigen::LDLT<Eigen::MatrixXd> ldlt(JJT);
      Eigen::VectorXd dq = ik.step_size * J.transpose() * ldlt.solve(err);
      if (null_gain > 0.0) {
        Eigen::VectorXd g = joint_limit_grad(model, q);
        dq += null_gain * (g - J.transpose() * ldlt.solve(J * g));
      }
      q = clamp_config(model, pin::integrate(model, q, dq));
    }
    JointTrajectoryPoint jp;
    jp.time = pt.time;
    jp.q = q;
    jp.ik_success = converged;
    out.push_back(std::move(jp));
  }
  return out;
}

std::vector<JointTrajectoryPoint> plan_joint_space_trajectory(
    const RobotModel& rm, int end_frame_id, const Eigen::VectorXd& q_start,
    const Eigen::VectorXd& q_end, double duration, const TrajPlanParams& params,
    const CLIKParams& ik, double null_gain) {
  auto [ps, rs, T_start] = rm.fk(q_start, "");
  auto [pe, re, T_end] = rm.fk(q_end, "");
  (void)ps; (void)rs; (void)pe; (void)re;
  auto cart = plan_cartesian_geodesic_trajectory(T_start, T_end, duration, params);
  return track_trajectory(rm, end_frame_id, cart.trajectory, q_start, ik, null_gain);
}

TrajStats compute_traj_stats(const RobotModel& rm, int end_frame_id,
                             const std::vector<JointTrajectoryPoint>& jt,
                             const Eigen::Matrix4d& T_start, const Eigen::Matrix4d& T_end,
                             double duration, const TrajPlanParams& params) {
  TrajStats stats;
  stats.total_points = static_cast<int>(jt.size());
  auto ref = plan_cartesian_geodesic_trajectory(T_start, T_end, duration, params);
  const auto& ref_pts = ref.trajectory.points;
  double sum_err = 0.0;
  for (std::size_t i = 0; i < jt.size(); ++i) {
    if (i >= ref_pts.size()) break;
    if (jt[i].ik_success) stats.success_count++;
    auto [p, r, oMf_h] = rm.fk(jt[i].q, "");
    (void)p; (void)r;
    pin::SE3 oMf = se3_from_homogeneous(oMf_h);
    pin::SE3 ref_pose = se3_from_homogeneous(ref_pts[i].pose);
    double err_norm = pin::log6(oMf.actInv(ref_pose)).toVector().norm();
    stats.max_ik_error = std::max(stats.max_ik_error, err_norm);
    sum_err += err_norm;
  }
  if (stats.total_points > 0) {
    stats.success_rate = static_cast<double>(stats.success_count) / stats.total_points;
    stats.avg_ik_error = sum_err / stats.total_points;
  }
  return stats;
}

}  // namespace traj
}  // namespace rebotarm
