// RobotModel：持有 pin::Model + pin::Data，提供 FK / Jacobian / IK / 帧与关节查询。
// 对照 reBotArm_control_py/kinematics/{robot_model,forward_kinematics,inverse_kinematics}.py。
#pragma once
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <Eigen/Dense>
#include <string>
#include <vector>
#include <utility>

namespace rebotarm {

struct IKParams {
  int max_iter = 1000;
  double tolerance = 1e-4;
  double step_size = 0.5;
  double damping = 1e-6;
};

struct IKResult {
  Eigen::VectorXd q;
  bool success = false;
  double error = 0.0;
  int iterations = 0;
};

class RobotModel {
public:
  explicit RobotModel(const std::string& urdf_path);

  int nq() const { return model_.nq; }
  int nv() const { return model_.nv; }
  const std::string& urdf_path() const { return urdf_path_; }

  std::vector<std::string> joint_names() const;
  std::vector<std::pair<double, double>> joint_limits() const;
  std::vector<std::string> all_frame_names() const;
  int frame_id(const std::string& name) const;
  int end_effector_frame_id() const;

  Eigen::VectorXd neutral() const;
  Eigen::VectorXd random_configuration() const;

  // 重力配置（写入 model.gravity，影响后续动力学）。
  void set_gravity(const Eigen::Vector3d& g);
  Eigen::Vector3d get_gravity() const;

  // FK：返回末端帧（或指定帧）的 (pos(3), rot(3x3), homogeneous(4x4))。
  std::tuple<Eigen::Vector3d, Eigen::Matrix3d, Eigen::Matrix4d>
  fk(const Eigen::VectorXd& q, const std::string& frame_name = "") const;

  // 帧的 6×nv 雅可比（LOCAL）。
  Eigen::MatrixXd frame_jacobian_local(const Eigen::VectorXd& q, int frame_id) const;

  // 阻尼最小二乘 CLIK（target 为 4×4 齐次矩阵）。
  IKResult solve_ik(const Eigen::Matrix4d& target, const Eigen::VectorXd& q_init,
                    int end_frame_id, const IKParams& params) const;

  IKResult solve_ik_with_retry(const Eigen::Matrix4d& target, const Eigen::VectorXd& q_seed,
                               int end_frame_id, const IKParams& params,
                               int max_retries = 8) const;

  // 暴露底层供 dynamics/trajectory 复用（同一进程内）。
  const pinocchio::Model& model() const { return model_; }
  pinocchio::Data& data() const { return data_; }

private:
  std::string urdf_path_;
  pinocchio::Model model_;
  mutable pinocchio::Data data_;

  Eigen::VectorXd clamp_config(const Eigen::VectorXd& q) const;
  std::pair<double, Eigen::Matrix<double, 6, 1>>
  compute_error(const Eigen::VectorXd& q, const pinocchio::SE3& target, int end_frame_id) const;
};

}  // namespace rebotarm
