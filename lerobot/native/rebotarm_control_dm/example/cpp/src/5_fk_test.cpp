#include "example_common.hpp"

#include <iostream>

int main(int argc, char** argv) {
  if (example::has_flag(argc, argv, "--help") || example::has_flag(argc, argv, "-h")) {
    std::cout << "Usage: ./5_fk_test [--urdf path/to/robot.urdf]\n";
    return 0;
  }

  rebotarm::RobotModel model(example::urdf_arg(argc, argv));
  std::cout << "C++ FK demo. Input " << model.nq() << " joint angles in degrees.\n";
  std::cout << "Backend: rebotarm::RobotModel + Pinocchio C++.\n";
  std::cout << "examples: 0 0 0 0 0 0 | 45 -30 15 -60 90 180\n";
  std::cout << "q / quit / exit to stop.\n\n";

  std::string line;
  while (true) {
    std::cout << "joint angles > ";
    if (!std::getline(std::cin, line)) break;
    if (line == "q" || line == "quit" || line == "exit") break;
    if (line.empty()) continue;
    try {
      const auto values = example::parse_numbers(line);
      const Eigen::VectorXd q = example::q_from_degrees(values, model.nq());
      example::print_pose(model, q);
    } catch (const std::exception& e) {
      std::cout << "invalid input: " << e.what() << "\n";
    }
  }
  return 0;
}
