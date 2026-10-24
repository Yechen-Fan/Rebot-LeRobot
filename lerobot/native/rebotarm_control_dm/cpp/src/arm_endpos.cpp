#include "rebotarm/arm_endpos.hpp"

#include <pinocchio/math/rpy.hpp>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <chrono>
#include <cmath>
#include <cstdio>

namespace rebotarm {

namespace py = pybind11;
using namespace pybind11::literals;

ArmEndPos::ArmEndPos(py::object arm, const std::string& urdf_path, double dt,
                     traj::TrajProfile profile)
    : arm_(std::move(arm)), model_(urdf_path), dt_(dt) {
  n_ = arm_.attr("num_joints").cast<int>();
  end_frame_id_ = model_.end_effector_frame_id();
  traj_params_.dt = dt;
  traj_params_.profile = profile;
  ik_solver_params_ = IKParams{200, 1e-4, 0.5, 1e-6};
  clik_params_ = traj::CLIKParams{200, 1e-4, 1e-6, 0.8};
}

ArmEndPos::~ArmEndPos() { stop_traj_thread(); }

Eigen::VectorXd ArmEndPos::current_q() {
  py::gil_scoped_acquire gil;
  py::tuple st = arm_.attr("get_state")().cast<py::tuple>();
  return st[0].cast<Eigen::VectorXd>();
}

void ArmEndPos::set_target(const Eigen::VectorXd& q) {
  py::gil_scoped_acquire gil;
  arm_.attr("set_targets")(q);
}

void ArmEndPos::set_target(const Eigen::VectorXd& q, const Eigen::VectorXd& vlim) {
  py::gil_scoped_acquire gil;
  arm_.attr("set_targets")(q, "vlim"_a = vlim);
}

void ArmEndPos::stop_traj_thread() {
  stop_send_.store(true);
  if (send_thread_.joinable()) {
    // 释放 GIL 再 join（发送线程内部需要 GIL 调用 set_targets）。
    py::gil_scoped_release rel;
    send_thread_.join();
  }
  moving_.store(false);
}

Eigen::Matrix4d ArmEndPos::target_pose(double x, double y, double z, double roll, double pitch,
                                       double yaw) const {
  Eigen::Matrix4d M = Eigen::Matrix4d::Identity();
  M.block<3, 3>(0, 0) = pinocchio::rpy::rpyToMatrix(roll, pitch, yaw);
  M(0, 3) = x;
  M(1, 3) = y;
  M(2, 3) = z;
  return M;
}

void ArmEndPos::start() {
  arm_.attr("connect")();
  arm_.attr("mode_pos_vel")();
  arm_.attr("enable")();
  // 初始目标=当前位姿，避免启动跳变；随后启动 RT-native 循环。
  Eigen::VectorXd q = current_q();
  set_target(q);
  arm_.attr("start_rt_loop")();
  running_ = true;
}

void ArmEndPos::end() {
  if (!running_) return;
  safe_home(std::nullopt);
  {
    py::gil_scoped_acquire gil;
    arm_.attr("disconnect")();
  }
  running_ = false;
}

bool ArmEndPos::move_to_ik(double x, double y, double z, double roll, double pitch, double yaw) {
  if (!running_) return false;
  stop_traj_thread();
  Eigen::VectorXd q_curr = current_q();
  Eigen::Matrix4d target = target_pose(x, y, z, roll, pitch, yaw);
  IKResult r = model_.solve_ik(target, q_curr, end_frame_id_, ik_solver_params_);
  if (!r.success) {
    std::printf("[ArmEndPos/IK] IK 未收敛 err=%.3e\n", r.error);
    return false;
  }
  set_target(r.q);
  return true;
}

bool ArmEndPos::move_to_traj(double x, double y, double z, double roll, double pitch, double yaw,
                             double duration) {
  if (!running_) return false;
  Eigen::VectorXd q_start = current_q();
  Eigen::Matrix4d target = target_pose(x, y, z, roll, pitch, yaw);
  IKResult ik = model_.solve_ik(target, q_start, end_frame_id_, ik_solver_params_);
  if (!ik.success) {
    std::printf("[ArmEndPos/Traj] IK 失败 err=%.4f\n", ik.error);
    return false;
  }
  auto [ps, rs, T_start] = model_.fk(q_start, "");
  auto [pe, re, T_end] = model_.fk(ik.q, "");
  (void)ps; (void)rs; (void)pe; (void)re;
  if (duration <= 0.0) {
    double dist = (target.block<3, 1>(0, 3) - T_start.block<3, 1>(0, 3)).norm();
    duration = std::max(1.0, dist / 0.1);
  }
  auto cart = traj::plan_cartesian_geodesic_trajectory(T_start, T_end, duration, traj_params_);
  auto jt = traj::track_trajectory(model_, end_frame_id_, cart.trajectory, q_start, clik_params_,
                                   0.1);
  if (jt.empty()) {
    std::printf("[ArmEndPos/Traj] 轨迹为空\n");
    return false;
  }

  std::vector<Eigen::VectorXd> pts;
  pts.reserve(jt.size());
  for (auto& p : jt) pts.push_back(p.q);

  stop_traj_thread();
  stop_send_.store(false);
  moving_.store(true);
  const double interval = duration / static_cast<double>(pts.size());
  send_thread_ = std::thread([this, pts, interval]() {
    for (const auto& q : pts) {
      if (stop_send_.load()) break;
      set_target(q);  // 内部获取 GIL
      std::this_thread::sleep_for(std::chrono::duration<double>(interval));
    }
    moving_.store(false);
  });
  return true;
}

void ArmEndPos::safe_home(std::optional<double> vlim) {
  if (!running_) return;
  const double v = vlim.value_or(home_vel_);
  stop_traj_thread();
  Eigen::VectorXd zero = Eigen::VectorXd::Zero(n_);
  Eigen::VectorXd vlim_vec = Eigen::VectorXd::Constant(n_, v);
  set_target(zero, vlim_vec);

  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (true) {
    Eigen::VectorXd q = current_q();
    if (q.cwiseAbs().maxCoeff() < 0.01) break;
    if (std::chrono::steady_clock::now() > deadline) {
      std::printf("[ArmEndPos] safe_home 超时\n");
      break;
    }
    std::this_thread::sleep_for(std::chrono::duration<double>(dt_));
  }
}

}  // namespace rebotarm
