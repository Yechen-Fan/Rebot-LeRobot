// 轨迹层（对照 reBotArm_control_py/trajectory/*.py）：SE(3) 测地线采样 + CLIK 跟踪 + 统计。
#pragma once
#include "rebotarm/robot_model.hpp"
#include <Eigen/Dense>
#include <string>
#include <vector>

namespace rebotarm {
namespace traj {

enum class TrajProfile { LINEAR, MIN_JERK, TRAPEZOID };

struct TrajPlanParams {
  double dt = 0.02;
  TrajProfile profile = TrajProfile::MIN_JERK;
  double accel_ratio = 0.25;
};

struct CartesianPoint {
  double time = 0.0;
  Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();  // 4×4 齐次（Python 边界）
};

struct CartesianTrajectory {
  std::vector<CartesianPoint> points;
  double duration() const { return points.empty() ? 0.0 : points.back().time; }
};

struct CartesianTrajectoryResult {
  CartesianTrajectory trajectory;
  int n_points = 0;
};

struct CLIKParams {
  int max_iter = 200;
  double tolerance = 1e-4;
  double damping = 1e-6;
  double step_size = 0.8;
};

struct JointTrajectoryPoint {
  double time = 0.0;
  Eigen::VectorXd q;
  bool ik_success = false;
};

struct TrajStats {
  int total_points = 0;
  int success_count = 0;
  double success_rate = 0.0;
  double max_ik_error = 0.0;
  double avg_ik_error = 0.0;
};

CartesianTrajectoryResult plan_cartesian_geodesic_trajectory(const Eigen::Matrix4d& start_pose,
                                                             const Eigen::Matrix4d& end_pose,
                                                             double duration,
                                                             const TrajPlanParams& params);

std::vector<JointTrajectoryPoint> track_trajectory(const RobotModel& rm, int end_frame_id,
                                                   const CartesianTrajectory& traj,
                                                   const Eigen::VectorXd& q_init,
                                                   const CLIKParams& ik_params, double null_gain);

std::vector<JointTrajectoryPoint> plan_joint_space_trajectory(
    const RobotModel& rm, int end_frame_id, const Eigen::VectorXd& q_start,
    const Eigen::VectorXd& q_end, double duration, const TrajPlanParams& params,
    const CLIKParams& ik_params, double null_gain);

TrajStats compute_traj_stats(const RobotModel& rm, int end_frame_id,
                             const std::vector<JointTrajectoryPoint>& jt,
                             const Eigen::Matrix4d& T_start, const Eigen::Matrix4d& T_end,
                             double duration, const TrajPlanParams& params);

}  // namespace traj
}  // namespace rebotarm
