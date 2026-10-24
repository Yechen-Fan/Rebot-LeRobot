#include "example_common.hpp"

#include <exception>
#include <iostream>

int main(int argc, char** argv) {
  try {
    if (example::has_flag(argc, argv, "--help") || example::has_flag(argc, argv, "-h")) {
      std::cout << "Usage: ./7_arm_ik_control --port /dev/ttyACM0 --rate 150 [--duration 2.0]\n";
      return 0;
    }

    const double rate = example::parse_rate(argc, argv, example::kDefaultRateHz);
    const double duration = example::arg_double(argc, argv, "--duration", 2.0);
    rebotarm::RobotModel model(example::urdf_arg(argc, argv));
    rebotarm::IKParams params;
    params.max_iter = 2000;
    params.tolerance = 1e-4;
    params.step_size = 0.5;
    params.damping = 1e-6;

    auto arm = example::B601Arm::open(example::parse_port(argc, argv));
    arm.enable();
    arm.ensure_all_mode(motorbridge::Mode::POS_VEL);
    std::cout << "Connected. Input target x y z in meters, state, or q.\n";
    std::cout << "Backend: C++/Pinocchio librebotarm_math.so.\n";

    while (true) {
      const auto line_opt = example::prompt("target xyz > ");
      if (!line_opt) break;
      const std::string line = *line_opt;
      if (line == "q" || line == "quit" || line == "exit") break;
      if (line == "state") {
        arm.print_state();
        continue;
      }
      if (line.empty()) continue;
      const auto values = example::parse_floats(line);
      if (values.size() < 3) {
        std::cout << "need x y z\n";
        continue;
      }

      const auto start = arm.positions_or_zero();
      Eigen::VectorXd seed = arm.arm_positions_or_zero();
      const auto [current_pos, current_rot, current_T] = model.fk(seed, "");
      (void)current_pos;
      (void)current_rot;
      Eigen::Matrix4d target_T = current_T;
      target_T.block<3, 1>(0, 3) = Eigen::Vector3d(values[0], values[1], values[2]);
      const auto result =
          model.solve_ik_with_retry(target_T, seed, model.end_effector_frame_id(), params, 8);
      std::cout << "  [" << (result.success ? "converged" : "not converged")
                << "] iterations=" << result.iterations << " error=" << result.error << "\n";
      example::print_vector("  q(deg):", example::q_degrees(result.q), 2);

      auto end = start;
      for (int i = 0; i < example::kArmDof; ++i) end[i] = static_cast<float>(result.q[i]);
      example::move_pos_vel_path(arm, start, end, duration, rate);
    }

    arm.close();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
