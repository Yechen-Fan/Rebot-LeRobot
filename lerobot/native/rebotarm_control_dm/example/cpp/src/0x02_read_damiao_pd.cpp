#include "example_common.hpp"

#include <exception>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct Target {
  std::string label;
  std::string port;
};

void usage() {
  std::cout
      << "Read Damiao POS_VEL gain registers from B601 motors.\n"
      << "Usage:\n"
      << "  ./0x02_read_damiao_pd --default-bi [--timeout-ms 300]\n"
      << "  ./0x02_read_damiao_pd --port /dev/ttyACM0 [--port /dev/ttyACM1]\n"
      << "\n"
      << "Registers: 25 vel_kp, 26 vel_ki, 27 pos_kp, 28 pos_ki\n";
}

void read_target(const Target& target, uint32_t timeout_ms) {
  std::cout << "\n[" << target.label << "] " << target.port << "\n";
  auto controller = example::open_controller(target.port);
  std::cout << std::left << std::setw(16) << "joint" << std::right << std::setw(12)
            << "vel_kp" << std::setw(12) << "vel_ki" << std::setw(12) << "pos_kp"
            << std::setw(12) << "pos_ki" << "\n";

  for (const auto& joint : example::b601_joints()) {
    auto motor = controller->add_damiao_motor(joint.motor_id, joint.feedback_id, joint.model);
    std::cout << std::left << std::setw(16) << joint.name;
    for (const uint8_t rid : {25, 26, 27, 28}) {
      try {
        std::cout << std::right << std::setw(12) << std::fixed << std::setprecision(6)
                  << motor.get_register_f32(rid, timeout_ms);
      } catch (const std::exception& e) {
        std::string err = "ERR";
        std::cout << std::right << std::setw(12) << err;
      }
    }
    std::cout << "\n";
  }

  try {
    controller->close_bus();
  } catch (...) {
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (example::has_flag(argc, argv, "--help") || example::has_flag(argc, argv, "-h")) {
      usage();
      return 0;
    }

    const uint32_t timeout_ms =
        static_cast<uint32_t>(example::arg_int(argc, argv, "--timeout-ms", 300));
    std::vector<Target> targets;
    if (example::has_flag(argc, argv, "--default-bi")) {
      targets.push_back({"left", example::arg_value(argc, argv, "--left-port", "/dev/ttyACM0")});
      targets.push_back({"right", example::arg_value(argc, argv, "--right-port", "/dev/ttyACM1")});
    }
    int index = 1;
    for (const auto& port : example::arg_values(argc, argv, "--port")) {
      targets.push_back({"port" + std::to_string(index++), port});
    }
    if (targets.empty()) {
      usage();
      throw std::runtime_error("no target specified; pass --default-bi or --port");
    }
    for (const auto& target : targets) read_target(target, timeout_ms);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
