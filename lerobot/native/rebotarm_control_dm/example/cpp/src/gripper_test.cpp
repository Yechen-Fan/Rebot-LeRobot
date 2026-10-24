#include "example_common.hpp"

#include <exception>
#include <iostream>
#include <sstream>

int main(int argc, char** argv) {
  try {
    if (example::has_flag(argc, argv, "--help") || example::has_flag(argc, argv, "-h")) {
      std::cout << "Usage: ./gripper_test --port /dev/ttyACM0\n";
      return 0;
    }

    auto arm = example::B601Arm::open(example::parse_port(argc, argv));
    auto& gripper = arm.motors[6];
    arm.enable();
    gripper.ensure_mode(motorbridge::Mode::FORCE_POS, 300);

    std::cout << "Gripper commands: open / close / pos <deg> / forcepos <deg> [vlim] [ratio] / posvel <deg> [vlim] / state / q\n";
    while (true) {
      const auto line_opt = example::prompt("> ");
      if (!line_opt) break;
      const std::string line = *line_opt;
      if (line == "q" || line == "quit" || line == "exit") break;
      if (line == "state") {
        arm.print_state();
        continue;
      }
      std::istringstream in(line);
      std::string cmd;
      in >> cmd;
      if (cmd == "open") {
        gripper.send_force_pos(0.0f, 5.2359877f, 0.05f);
      } else if (cmd == "close") {
        gripper.send_force_pos(example::deg_to_rad_f32(-270.0), 5.2359877f, 0.05f);
      } else if (cmd == "pos") {
        double pos = 0.0;
        in >> pos;
        gripper.send_force_pos(example::deg_to_rad_f32(pos), 5.2359877f, 0.05f);
      } else if (cmd == "forcepos" || cmd == "force_pos") {
        double pos = 0.0;
        float vlim = 5.2359877f;
        float ratio = 0.05f;
        in >> pos;
        if (!(in >> vlim)) vlim = 5.2359877f;
        if (!(in >> ratio)) ratio = 0.05f;
        gripper.send_force_pos(example::deg_to_rad_f32(pos), vlim, ratio);
      } else if (cmd == "posvel" || cmd == "pos_vel") {
        gripper.ensure_mode(motorbridge::Mode::POS_VEL, 300);
        double pos = 0.0;
        float vlim = 5.2359877f;
        in >> pos;
        if (!(in >> vlim)) vlim = 5.2359877f;
        gripper.send_pos_vel(example::deg_to_rad_f32(pos), vlim);
      } else {
        std::cout << "unknown command\n";
      }
    }

    arm.close();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
