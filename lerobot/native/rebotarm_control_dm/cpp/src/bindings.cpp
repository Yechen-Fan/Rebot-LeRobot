// _math：rebotarm_control_dm 原生数学层（Pinocchio C++ via pybind11）。
// 暴露与 reBotArm_control_py 的 kinematics 同名的类型与函数，Python 侧纯 re-export。
#include "rebotarm/robot_model.hpp"
#include "rebotarm/dynamics.hpp"
#include "rebotarm/trajectory.hpp"
#include "rebotarm/arm_endpos.hpp"
#include "rebotarm/se3_conv.hpp"

#include <pinocchio/math/rpy.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <optional>
#include <string>

namespace py = pybind11;
using rebotarm::IKParams;
using rebotarm::IKResult;
using rebotarm::RobotModel;

// ── 便捷自由函数（镜像 kinematics/*.py） ───────────────────────────────────────

static Eigen::Matrix4d pos_rot_to_se3(const Eigen::Vector3d& pos,
                                      std::optional<Eigen::Matrix3d> rot,
                                      double roll, double pitch, double yaw) {
  Eigen::Matrix3d R = rot ? *rot : pinocchio::rpy::rpyToMatrix(roll, pitch, yaw);
  Eigen::Matrix4d M = Eigen::Matrix4d::Identity();
  M.block<3, 3>(0, 0) = R;
  M.block<3, 1>(0, 3) = pos;
  return M;
}

static py::tuple compute_fk(const RobotModel& rm, const Eigen::VectorXd& q,
                            const std::string& frame_name) {
  auto [pos, rot, homog] = rm.fk(q, frame_name);
  return py::make_tuple(pos, rot, homog);
}

static py::tuple joint_to_pose(const RobotModel& rm, const Eigen::VectorXd& q,
                               const std::string& frame_name) {
  auto [pos, rot, homog] = rm.fk(q, frame_name);
  (void)homog;
  Eigen::Vector3d euler = pinocchio::rpy::matrixToRpy(rot);
  return py::make_tuple(pos, euler);
}

static IKResult compute_ik(RobotModel& rm, std::optional<Eigen::VectorXd> q_init,
                           const Eigen::Vector3d& target_pos,
                           std::optional<Eigen::Matrix3d> target_rot, double roll, double pitch,
                           double yaw, const std::string& frame_name, IKParams params) {
  Eigen::Matrix4d target = pos_rot_to_se3(target_pos, target_rot, roll, pitch, yaw);
  Eigen::VectorXd q0 = q_init ? *q_init : rm.neutral();
  const int fid = frame_name.empty() ? rm.end_effector_frame_id() : rm.frame_id(frame_name);
  return rm.solve_ik(target, q0, fid, params);
}

PYBIND11_MODULE(_math, m) {
  m.doc() = "rebotarm_control_dm 原生数学层（Pinocchio C++）";

  py::class_<IKParams>(m, "IKParams")
      .def(py::init<>())
      .def(py::init([](int mi, double tol, double ss, double d) {
             IKParams p; p.max_iter = mi; p.tolerance = tol; p.step_size = ss; p.damping = d; return p;
           }),
           py::arg("max_iter") = 1000, py::arg("tolerance") = 1e-4,
           py::arg("step_size") = 0.5, py::arg("damping") = 1e-6)
      .def_readwrite("max_iter", &IKParams::max_iter)
      .def_readwrite("tolerance", &IKParams::tolerance)
      .def_readwrite("step_size", &IKParams::step_size)
      .def_readwrite("damping", &IKParams::damping);

  py::class_<IKResult>(m, "IKResult")
      .def_readonly("q", &IKResult::q)
      .def_readonly("success", &IKResult::success)
      .def_readonly("error", &IKResult::error)
      .def_readonly("iterations", &IKResult::iterations)
      .def("__repr__", [](const IKResult& r) {
        return "IKResult(success=" + std::string(r.success ? "True" : "False") +
               ", error=" + std::to_string(r.error) +
               ", iterations=" + std::to_string(r.iterations) + ")";
      });

  py::class_<RobotModel>(m, "RobotModel")
      .def(py::init<const std::string&>(), py::arg("urdf_path"))
      .def_property_readonly("nq", &RobotModel::nq)
      .def_property_readonly("nv", &RobotModel::nv)
      .def("joint_names", &RobotModel::joint_names)
      .def("joint_limits", &RobotModel::joint_limits)
      .def("all_frame_names", &RobotModel::all_frame_names)
      .def("frame_id", &RobotModel::frame_id, py::arg("name"))
      .def("end_effector_frame_id", &RobotModel::end_effector_frame_id)
      .def("neutral", &RobotModel::neutral)
      .def("random_configuration", &RobotModel::random_configuration)
      .def("set_gravity", &RobotModel::set_gravity, py::arg("gravity"))
      .def("get_gravity", &RobotModel::get_gravity)
      .def("fk", [](const RobotModel& rm, const Eigen::VectorXd& q,
                    const std::string& frame_name) { return compute_fk(rm, q, frame_name); },
           py::arg("q"), py::arg("frame_name") = "")
      .def("frame_jacobian", &RobotModel::frame_jacobian_local, py::arg("q"), py::arg("frame_id"))
      .def("solve_ik", &RobotModel::solve_ik, py::arg("target"), py::arg("q_init"),
           py::arg("end_frame_id"), py::arg("params") = IKParams())
      .def("solve_ik_with_retry", &RobotModel::solve_ik_with_retry, py::arg("target"),
           py::arg("q_seed"), py::arg("end_frame_id"), py::arg("params") = IKParams(),
           py::arg("max_retries") = 8);

  // 模块级函数（镜像 kinematics/*.py 名称，供 Python 纯 re-export）
  m.def("load_robot_model", [](const std::string& urdf_path) { return RobotModel(urdf_path); },
        py::arg("urdf_path"));
  m.def("get_joint_names", [](const RobotModel& rm) { return rm.joint_names(); });
  m.def("get_joint_limits", [](const RobotModel& rm) { return rm.joint_limits(); });
  m.def("get_frame_id", [](const RobotModel& rm, const std::string& n) { return rm.frame_id(n); });
  m.def("get_end_effector_frame_id",
        [](const RobotModel& rm) { return rm.end_effector_frame_id(); });
  m.def("get_all_frame_names", [](const RobotModel& rm) { return rm.all_frame_names(); });

  m.def("pos_rot_to_se3", &pos_rot_to_se3, py::arg("pos"), py::arg("rot") = std::nullopt,
        py::arg("roll") = 0.0, py::arg("pitch") = 0.0, py::arg("yaw") = 0.0);
  m.def("compute_fk", &compute_fk, py::arg("model"), py::arg("q"), py::arg("frame_name") = "");
  m.def("joint_to_pose", &joint_to_pose, py::arg("model"), py::arg("q"),
        py::arg("frame_name") = "");
  // 注：C++ 无全局默认模型，故 model 作为第一参（Python shim 可注入默认模型）。
  // 注：C++ 无全局默认模型，故 model 作为第一参（Python shim 可注入默认模型）。
  // q_init 必填但可为 None；target_pos 必填，故二者均不给默认值。
  m.def("compute_ik", &compute_ik, py::arg("model"), py::arg("q_init"), py::arg("target_pos"),
        py::arg("target_rot") = std::nullopt, py::arg("roll") = 0.0, py::arg("pitch") = 0.0,
        py::arg("yaw") = 0.0, py::arg("frame_name") = "end_link", py::arg("params") = IKParams());

  // ── 动力学（镜像 dynamics/*.py，第一参为 RobotModel） ──────────────────────
  namespace d = rebotarm::dyn;
  m.def("mass_matrix", &d::mass_matrix, py::arg("model"), py::arg("q"));
  m.def("coriolis_matrix", &d::coriolis_matrix, py::arg("model"), py::arg("q"), py::arg("v"));
  m.def("gravity_vector", &d::gravity_vector, py::arg("model"), py::arg("q"));
  m.def("nle", &d::nle, py::arg("model"), py::arg("q"), py::arg("v"));
  m.def("all_terms", &d::all_terms, py::arg("model"), py::arg("q"), py::arg("v"));
  m.def("inverse_dynamics", &d::inverse_dynamics, py::arg("model"), py::arg("q"), py::arg("v"),
        py::arg("a"));
  m.def("generalized_gravity", &d::generalized_gravity, py::arg("model"), py::arg("q"));
  m.def("static_torque", &d::static_torque, py::arg("model"), py::arg("q"));
  m.def("forward_dynamics", &d::forward_dynamics, py::arg("model"), py::arg("q"), py::arg("v"),
        py::arg("tau"));
  m.def("forward_dynamics_from_nle", &d::forward_dynamics_from_nle, py::arg("model"), py::arg("q"),
        py::arg("v"), py::arg("tau"));
  m.def("kinetic_energy", &d::kinetic_energy, py::arg("model"), py::arg("q"), py::arg("v"));
  m.def("potential_energy", &d::potential_energy, py::arg("model"), py::arg("q"));
  m.def("total_energy", &d::total_energy, py::arg("model"), py::arg("q"), py::arg("v"));
  m.def("center_of_mass", &d::center_of_mass, py::arg("model"), py::arg("q"),
        py::arg("center_zero") = false);
  m.def("com_velocity", &d::com_velocity, py::arg("model"), py::arg("q"), py::arg("v"));
  m.def("centroidal_momentum", &d::centroidal_momentum, py::arg("model"), py::arg("q"),
        py::arg("v"));
  m.def("centroidal_matrix", &d::centroidal_matrix, py::arg("model"), py::arg("q"), py::arg("v"));
  m.def("rnea_derivatives", &d::rnea_derivatives, py::arg("model"), py::arg("q"), py::arg("v"),
        py::arg("a"));
  m.def("coriolis_derivatives", &d::coriolis_derivatives, py::arg("model"), py::arg("q"),
        py::arg("v"));
  m.def("generalized_gravity_derivatives", &d::generalized_gravity_derivatives, py::arg("model"),
        py::arg("q"));
  m.def("mass_matrix_derivatives", &d::mass_matrix_derivatives, py::arg("model"), py::arg("q"));

  // ── 轨迹（镜像 trajectory/*.py） ──────────────────────────────────────────
  namespace t = rebotarm::traj;
  py::enum_<t::TrajProfile>(m, "TrajProfile")
      .value("LINEAR", t::TrajProfile::LINEAR)
      .value("MIN_JERK", t::TrajProfile::MIN_JERK)
      .value("TRAPEZOID", t::TrajProfile::TRAPEZOID);

  py::class_<t::TrajPlanParams>(m, "TrajPlanParams")
      .def(py::init([](double dt, t::TrajProfile pf, double ar) {
             t::TrajPlanParams p; p.dt = dt; p.profile = pf; p.accel_ratio = ar; return p;
           }),
           py::arg("dt") = 0.02, py::arg("profile") = t::TrajProfile::MIN_JERK,
           py::arg("accel_ratio") = 0.25)
      .def_readwrite("dt", &t::TrajPlanParams::dt)
      .def_readwrite("profile", &t::TrajPlanParams::profile)
      .def_readwrite("accel_ratio", &t::TrajPlanParams::accel_ratio);

  py::class_<t::CartesianPoint>(m, "CartesianPoint")
      .def_readonly("time", &t::CartesianPoint::time)
      .def_readonly("pose", &t::CartesianPoint::pose);

  py::class_<t::CartesianTrajectory>(m, "CartesianTrajectory")
      .def("points", [](const t::CartesianTrajectory& c) { return c.points; })
      .def("duration", &t::CartesianTrajectory::duration);

  py::class_<t::CartesianTrajectoryResult>(m, "CartesianTrajectoryResult")
      .def_readonly("trajectory", &t::CartesianTrajectoryResult::trajectory)
      .def_readonly("n_points", &t::CartesianTrajectoryResult::n_points);

  py::class_<t::CLIKParams>(m, "CLIKParams")
      .def(py::init([](int mi, double tol, double d, double ss) {
             t::CLIKParams p; p.max_iter = mi; p.tolerance = tol; p.damping = d; p.step_size = ss;
             return p;
           }),
           py::arg("max_iter") = 200, py::arg("tolerance") = 1e-4, py::arg("damping") = 1e-6,
           py::arg("step_size") = 0.8)
      .def_readwrite("max_iter", &t::CLIKParams::max_iter)
      .def_readwrite("tolerance", &t::CLIKParams::tolerance)
      .def_readwrite("damping", &t::CLIKParams::damping)
      .def_readwrite("step_size", &t::CLIKParams::step_size);

  py::class_<t::JointTrajectoryPoint>(m, "JointTrajectoryPoint")
      .def_readonly("time", &t::JointTrajectoryPoint::time)
      .def_readonly("q", &t::JointTrajectoryPoint::q)
      .def_readonly("ik_success", &t::JointTrajectoryPoint::ik_success);

  py::class_<t::TrajStats>(m, "TrajStats")
      .def_readonly("total_points", &t::TrajStats::total_points)
      .def_readonly("success_count", &t::TrajStats::success_count)
      .def_readonly("success_rate", &t::TrajStats::success_rate)
      .def_readonly("max_ik_error", &t::TrajStats::max_ik_error)
      .def_readonly("avg_ik_error", &t::TrajStats::avg_ik_error);

  m.def("plan_cartesian_geodesic_trajectory", &t::plan_cartesian_geodesic_trajectory,
        py::arg("start_pose"), py::arg("end_pose"), py::arg("duration"),
        py::arg("params") = t::TrajPlanParams());
  m.def("track_trajectory", &t::track_trajectory, py::arg("model"), py::arg("end_frame_id"),
        py::arg("traj"), py::arg("q_init"), py::arg("ik_params") = t::CLIKParams(),
        py::arg("null_gain") = 0.0);
  m.def("plan_joint_space_trajectory", &t::plan_joint_space_trajectory, py::arg("model"),
        py::arg("end_frame_id"), py::arg("q_start"), py::arg("q_end"), py::arg("duration"),
        py::arg("params") = t::TrajPlanParams(), py::arg("ik_params") = t::CLIKParams(),
        py::arg("null_gain") = 0.0);
  m.def("compute_traj_stats", &t::compute_traj_stats, py::arg("model"), py::arg("end_frame_id"),
        py::arg("jt"), py::arg("T_start"), py::arg("T_end"), py::arg("duration"),
        py::arg("params") = t::TrajPlanParams());

  // ── ArmEndPos 编排（驱动 Rust actuator） ──────────────────────────────────
  py::class_<rebotarm::ArmEndPos>(m, "ArmEndPos")
      .def(py::init<py::object, const std::string&, double, t::TrajProfile>(), py::arg("arm"),
           py::arg("urdf_path"), py::arg("dt") = 0.02,
           py::arg("profile") = t::TrajProfile::MIN_JERK)
      .def("start", &rebotarm::ArmEndPos::start)
      .def("end", &rebotarm::ArmEndPos::end)
      .def("move_to_ik", &rebotarm::ArmEndPos::move_to_ik, py::arg("x"), py::arg("y"), py::arg("z"),
           py::arg("roll") = 0.0, py::arg("pitch") = 0.0, py::arg("yaw") = 0.0)
      .def("move_to_traj", &rebotarm::ArmEndPos::move_to_traj, py::arg("x"), py::arg("y"),
           py::arg("z"), py::arg("roll") = 0.0, py::arg("pitch") = 0.0, py::arg("yaw") = 0.0,
           py::arg("duration") = 2.0)
      .def("safe_home", &rebotarm::ArmEndPos::safe_home, py::arg("vlim") = std::nullopt)
      .def("__enter__", [](py::object self) { return self; })
      .def("__exit__", [](rebotarm::ArmEndPos& s, py::args) { s.end(); });
}
