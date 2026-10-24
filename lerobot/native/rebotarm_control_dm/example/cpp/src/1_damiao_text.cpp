#include "example_common.hpp"

#include <exception>
#include <iostream>

int main(int argc, char** argv) {
  try {
    return example::run_single_motor_console(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
