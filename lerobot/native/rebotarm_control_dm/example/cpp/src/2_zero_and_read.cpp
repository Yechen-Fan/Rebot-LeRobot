#include "example_common.hpp"

#include <exception>
#include <iostream>

int main(int argc, char** argv) {
  try {
    if (example::has_flag(argc, argv, "--help") || example::has_flag(argc, argv, "-h")) {
      std::cout << "Usage: ./2_zero_and_read --port /dev/ttyACM0 [--skip-zero]\n";
      return 0;
    }

    auto arm = example::B601Arm::open(example::parse_port(argc, argv));
    std::cout << "connected: " << arm.port << "\n";

    if (!example::has_flag(argc, argv, "--skip-zero")) {
      std::cout << "This will set the current pose as zero for all B601 motors.\n";
      std::cout << "Type YES to continue.\n";
      if (example::prompt("confirm> ").value_or("") == "YES") {
        arm.disable();
        for (int i = 0; i < static_cast<int>(arm.motors.size()); ++i) {
          arm.motors[i].set_zero_position();
          std::cout << "zero set: " << example::b601_joints()[i].name << "\n";
        }
      } else {
        std::cout << "zero skipped\n";
      }
    }

    std::cout << "Press Enter to read state again, q to quit.\n";
    while (true) {
      arm.print_state();
      const auto line = example::prompt("> ");
      if (!line || *line == "q" || *line == "quit" || *line == "exit") break;
    }

    arm.close();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
