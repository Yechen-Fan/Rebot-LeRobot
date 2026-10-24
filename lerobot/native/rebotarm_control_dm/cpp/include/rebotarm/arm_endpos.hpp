// ArmEndPos：末端位置编排控制器（IK + 轨迹）。对照 controllers/arm_endpos_controller.py。
// 计算（IK/轨迹）在 C++；通过 py::object 驱动 Rust actuator（RobotArm）。
// 采用 arm 的 RT-native 循环（set_targets + start_rt_loop），避免每 tick 跨语言回调。
#pragma once
#include "rebotarm/robot_model.hpp"
#include "rebotarm/trajectory.hpp"

#include <pybind11/pybind11.h>
#include <Eigen/Dense>
#include <atomic>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace rebotarm {

class ArmEndPos {
public:
  ArmEndPos(pybind11::object arm, const std::string& urdf_path, double dt,
            traj::TrajProfile profile);
  ~ArmEndPos();

  void start();
  void end();
  bool move_to_ik(double x, double y, double z, double roll, double pitch, double yaw);
  bool move_to_traj(double x, double y, double z, double roll, double pitch, double yaw,
                    double duration);
  void safe_home(std::optional<double> vlim);

private:
  pybind11::object arm_;
  RobotModel model_;
  int n_;
  int end_frame_id_;
  double dt_;
  double home_vel_ = 0.3;
  traj::TrajPlanParams traj_params_;
  IKParams ik_solver_params_;
  traj::CLIKParams clik_params_;
  bool running_ = false;

  std::thread send_thread_;
  std::atomic<bool> stop_send_{false};
  std::atomic<bool> moving_{false};

  Eigen::VectorXd current_q();                       // arm.get_state()[0]
  void set_target(const Eigen::VectorXd& q);         // arm.set_targets(q)
  void set_target(const Eigen::VectorXd& q, const Eigen::VectorXd& vlim);
  void stop_traj_thread();
  Eigen::Matrix4d target_pose(double x, double y, double z, double roll, double pitch,
                              double yaw) const;
};

}  // namespace rebotarm
