#include "example_common.hpp"

#include <atomic>
#include <exception>
#include <iostream>
#include <mutex>
#include <thread>

int main(int argc, char** argv) {
  try {
    if (example::has_flag(argc, argv, "--help") || example::has_flag(argc, argv, "-h")) {
      std::cout << "Usage: ./3_mit_control --port /dev/ttyACM0 --rate 150\n";
      return 0;
    }

    const double rate = example::parse_rate(argc, argv, example::kDefaultRateHz);
    auto arm = example::B601Arm::open(example::parse_port(argc, argv));
    arm.enable();
    arm.ensure_all_mode(motorbridge::Mode::MIT);

    std::mutex mutex;
    std::vector<float> target = arm.positions_or_zero();
    std::vector<float> kp = example::default_kp();
    std::vector<float> kd = example::default_kd();
    std::atomic_bool running{true};

    std::thread loop([&]() {
      while (running.load(std::memory_order_relaxed)) {
        const auto tick = std::chrono::steady_clock::now();
        std::vector<float> target_copy;
        std::vector<float> kp_copy;
        std::vector<float> kd_copy;
        {
          std::lock_guard<std::mutex> lock(mutex);
          target_copy = target;
          kp_copy = kp;
          kd_copy = kd;
        }
        for (int i = 0; i < static_cast<int>(arm.motors.size()); ++i) {
          try {
            arm.motors[i].send_mit(target_copy[i], 0.0f, kp_copy[i], kd_copy[i], 0.0f);
          } catch (...) {
          }
        }
        example::sleep_to_rate(tick, rate);
      }
    });

    std::cout << "MIT loop started at " << rate << " Hz.\n";
    std::cout << "Input: q1 ... q6 [gripper] [kp kd], state, or q. Angles are degrees.\n";
    while (true) {
      const auto line_opt = example::prompt("> ");
      if (!line_opt) break;
      const std::string line = *line_opt;
      if (line == "q" || line == "quit" || line == "exit") break;
      if (line == "state") {
        arm.print_state();
        continue;
      }
      if (line.empty()) continue;
      const auto values = example::parse_floats(line);
      if (values.size() < 6) {
        std::cout << "need at least 6 joint angles\n";
        continue;
      }
      std::lock_guard<std::mutex> lock(mutex);
      for (int i = 0; i < std::min(example::kAllDof, static_cast<int>(values.size())); ++i) {
        target[i] = example::deg_to_rad_f32(values[i]);
      }
      if (values.size() >= example::kAllDof + 1) {
        std::fill(kp.begin(), kp.end(), static_cast<float>(values[example::kAllDof]));
      }
      if (values.size() >= example::kAllDof + 2) {
        std::fill(kd.begin(), kd.end(), static_cast<float>(values[example::kAllDof + 1]));
      }
    }

    running.store(false, std::memory_order_relaxed);
    loop.join();
    arm.close();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
