#include "example_common.hpp"
#include "rebotarm/dynamics.hpp"

#include <exception>
#include <iostream>

int main(int argc, char** argv) {
  try {
    if (example::has_flag(argc, argv, "--help") || example::has_flag(argc, argv, "-h")) {
      std::cout << "Usage: ./10_gravity_compensation_lock --port /dev/ttyACM0 --rate 200 "
                   "--use_gripper=true [--lock-kp 8.0] [--lock-kd 1.0]\n";
      return 0;
    }

    const double rate = example::parse_rate(argc, argv, 200.0);
    const bool use_gripper = example::parse_bool_arg(argc, argv, "--use_gripper", true);
    const float vel_threshold =
        static_cast<float>(example::arg_double(argc, argv, "--vel-threshold", 0.04));
    const float lock_kp = static_cast<float>(example::arg_double(
        argc, argv, "--lock-kp", example::arg_double(argc, argv, "--kp", 8.0)));
    const float lock_kd = static_cast<float>(example::arg_double(
        argc, argv, "--lock-kd", example::arg_double(argc, argv, "--kd", 1.0)));
    auto gravity_urdf = example::gravity_urdf_for_gripper(argc, argv, use_gripper);
    rebotarm::RobotModel model(gravity_urdf.path);

    std::cout << "C++ gravity-lock demo backend: C++/Pinocchio librebotarm_math.so.\n";
    std::cout << "use_gripper=" << (use_gripper ? "true" : "false")
              << "; end_link inertial scale=" << gravity_urdf.end_link_scale
              << "; vel-threshold=" << vel_threshold << "\n";
    std::cout << "MIT lock gains: kp=" << lock_kp << ", kd=" << lock_kd << "\n";
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
    std::vector<float> target = arm.positions_or_zero();

    while (!example::stop_requested()) {
      const auto tick = std::chrono::steady_clock::now();
      const auto states = arm.states();
      std::vector<float> current = target;
      float max_vel = 0.0f;
      for (int i = 0; i < count; ++i) {
        if (i < static_cast<int>(states.size()) && states[i]) {
          current[i] = states[i]->pos;
          max_vel = std::max(max_vel, std::abs(states[i]->vel));
        }
      }
      if (max_vel > vel_threshold) target = current;

      Eigen::VectorXd q_model(model.nq());
      for (int i = 0; i < model.nq(); ++i) q_model[i] = current.size() > i ? current[i] : 0.0f;
      const Eigen::VectorXd tau_g = rebotarm::dyn::generalized_gravity(model, q_model);
      for (int i = 0; i < count; ++i) {
        const float tau = i < tau_g.size() ? static_cast<float>(tau_g[i]) : 0.0f;
        arm.motors[i].send_mit(target[i], 0.0f, lock_kp, lock_kd, tau);
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
