#include "example_common.hpp"

#include <iostream>

int main(int argc, char** argv) {
  if (example::has_flag(argc, argv, "--help") || example::has_flag(argc, argv, "-h")) {
    std::cout << "Usage: ./6_ik_test [--urdf path/to/robot.urdf]\n";
    return 0;
  }

  rebotarm::RobotModel model(example::urdf_arg(argc, argv));
  const Eigen::VectorXd neutral = model.neutral();
  const auto [neutral_pos, neutral_rot, neutral_T] = model.fk(neutral, "");
  (void)neutral_pos;
  (void)neutral_rot;

  rebotarm::IKParams params;
  params.max_iter = 1000;
  params.tolerance = 1e-4;
  params.step_size = 0.5;
  params.damping = 1e-6;

  std::cout << "C++ IK demo. Input target pose:\n";
  std::cout << "  x y z                    (meters, keeps neutral orientation)\n";
  std::cout << "  x y z roll pitch yaw     (meters + radians)\n";
  std::cout << "q / quit / exit to stop.\n\n";

  Eigen::VectorXd seed = neutral;
  std::string line;
  while (true) {
    std::cout << "target pose > ";
    if (!std::getline(std::cin, line)) break;
    if (line == "q" || line == "quit" || line == "exit") break;
    if (line.empty()) continue;

    try {
      const auto values = example::parse_numbers(line);
      if (values.size() != 3 && values.size() != 6) {
        std::cout << "need x y z or x y z roll pitch yaw\n";
        continue;
      }
      Eigen::Matrix4d target = neutral_T;
      target.block<3, 1>(0, 3) = Eigen::Vector3d(values[0], values[1], values[2]);
      if (values.size() == 6) {
        target = example::pose_from_xyz_rpy(values[0], values[1], values[2], values[3],
                                            values[4], values[5]);
      }

      const auto result =
          model.solve_ik_with_retry(target, seed, model.end_effector_frame_id(), params, 8);
      std::cout << "  [" << (result.success ? "converged" : "not converged")
                << "] iterations=" << result.iterations << " error=" << result.error << "\n";
      example::print_vector("  q(deg):", example::q_degrees(result.q), 2);
      example::print_pose(model, result.q);
      seed = result.q;
    } catch (const std::exception& e) {
      std::cout << "invalid input: " << e.what() << "\n";
    }
  }
  return 0;
}
