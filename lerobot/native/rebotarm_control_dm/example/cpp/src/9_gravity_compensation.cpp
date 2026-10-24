#include "example_common.hpp"
#include "rebotarm/dynamics.hpp"

#include <exception>
#include <iostream>

int main(int argc, char** argv) {
  try {
    if (example::has_flag(argc, argv, "--help") || example::has_flag(argc, argv, "-h")) {
      std::cout << "Usage: ./9_gravity_compensation --port /dev/ttyACM0 --rate 200 "
                   "--use_gripper=true [--kp 0.0] [--kd 1.0]\n";
      return 0;
    }

    const double rate = example::parse_rate(argc, argv, 200.0);
    const bool use_gripper = example::parse_bool_arg(argc, argv, "--use_gripper", true);
    const float arm_kp = static_cast<float>(example::arg_double(argc, argv, "--kp", 0.0));
    const float arm_kd = static_cast<float>(example::arg_double(argc, argv, "--kd", 1.0));
    const float gripper_kp =
        static_cast<float>(example::arg_double(argc, argv, "--gripper-kp", 0.0));
    const float gripper_kd =
        static_cast<float>(example::arg_double(argc, argv, "--gripper-kd", 1.0));
    auto gravity_urdf = example::gravity_urdf_for_gripper(argc, argv, use_gripper);
    rebotarm::RobotModel model(gravity_urdf.path);
    const int model_nq = model.nq();

    std::cout << "C++ gravity demo backend: C++/Pinocchio librebotarm_math.so.\n";
    std::cout << "use_gripper=" << (use_gripper ? "true" : "false")
              << "; end_link inertial scale=" << gravity_urdf.end_link_scale << "\n";
    std::cout << "MIT command gains: arm kp=" << arm_kp << ", kd=" << arm_kd
              << "; gripper kp=" << gripper_kp << ", kd=" << gripper_kd << "\n";
    std::cout << "Ctrl+C to stop and disconnect.\n";
    example::install_signal_handler();

    auto arm = example::B601Arm::open(example::parse_port(argc, argv));
    arm.enable();
    if (use_gripper) {
      arm.ensure_all_mode(motorbridge::Mode::MIT);
    } else {
      arm.ensure_arm_mode(motorbridge::Mode::MIT);
    }
    const int count = use_gripper ? example::kAllDof : example::kArmDof;

    while (!example::stop_requested()) {
      const auto tick = std::chrono::steady_clock::now();
      const auto q = arm.positions_or_zero();
      Eigen::VectorXd q_model(model_nq);
      for (int i = 0; i < model_nq; ++i) q_model[i] = q.size() > i ? q[i] : 0.0f;
      const Eigen::VectorXd tau_g = rebotarm::dyn::generalized_gravity(model, q_model);
      for (int i = 0; i < count; ++i) {
        const float tau = i < tau_g.size() ? static_cast<float>(tau_g[i]) : 0.0f;
        const bool is_gripper = i >= model_nq;
        const float kp = is_gripper ? gripper_kp : arm_kp;
        const float kd = is_gripper ? gripper_kd : arm_kd;
        arm.motors[i].send_mit(q[i], 0.0f, kp, kd, tau);
      }
      example::sleep_to_rate(tick, rate);
    }

    arm.close();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
