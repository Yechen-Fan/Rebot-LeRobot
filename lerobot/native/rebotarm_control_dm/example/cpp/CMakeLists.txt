cmake_minimum_required(VERSION 3.16)
project(rebotarm_control_dm_cpp_examples LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

get_filename_component(REBOTARM_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(REBOTARM_CPP_DIR "${REBOTARM_ROOT}/cpp")
set(MOTORBRIDGE_ROOT "${REBOTARM_ROOT}/../motorbridge")

if(NOT PINOCCHIO_PREFIX)
  if(DEFINED ENV{PINOCCHIO_PREFIX})
    set(PINOCCHIO_PREFIX "$ENV{PINOCCHIO_PREFIX}")
  elseif(DEFINED ENV{CONDA_PREFIX} AND EXISTS "$ENV{CONDA_PREFIX}/include/pinocchio")
    set(PINOCCHIO_PREFIX "$ENV{CONDA_PREFIX}")
  elseif(EXISTS "/usr/local/include/pinocchio")
    set(PINOCCHIO_PREFIX "/usr/local")
  elseif(EXISTS "/usr/include/pinocchio")
    set(PINOCCHIO_PREFIX "/usr")
  endif()
endif()

if(NOT PINOCCHIO_PREFIX OR NOT EXISTS "${PINOCCHIO_PREFIX}/include/pinocchio")
  message(FATAL_ERROR
    "Pinocchio C++ was not found. Activate the conda environment or set "
    "-DPINOCCHIO_PREFIX=<prefix>.")
endif()

find_package(Eigen3 QUIET CONFIG)
if(NOT Eigen3_FOUND)
  find_path(EIGEN3_INCLUDE_DIR Eigen/Dense
    PATHS "${PINOCCHIO_PREFIX}/include/eigen3" /usr/include/eigen3 /usr/local/include/eigen3
    REQUIRED)
endif()

find_library(REBOTARM_MATH_LIB NAMES rebotarm_math
  HINTS "${REBOTARM_ROOT}/python/rebotarm_control_dm" REQUIRED)
find_library(CONDA_STDCXX_LIB NAMES stdc++
  HINTS "${PINOCCHIO_PREFIX}/lib" NO_DEFAULT_PATH)
find_library(MOTORBRIDGE_ABI_LIB NAMES motor_abi
  HINTS
    "${MOTORBRIDGE_ROOT}/target/release"
    "${MOTORBRIDGE_ROOT}/target/debug")
find_library(PINOCCHIO_DEFAULT_LIB NAMES pinocchio_default pinocchio
  HINTS "${PINOCCHIO_PREFIX}/lib" "${PINOCCHIO_PREFIX}/lib/x86_64-linux-gnu" REQUIRED)
get_filename_component(PINO_LIBDIR "${PINOCCHIO_DEFAULT_LIB}" DIRECTORY)
if(MOTORBRIDGE_ABI_LIB)
  get_filename_component(MOTORBRIDGE_ABI_LIBDIR "${MOTORBRIDGE_ABI_LIB}" DIRECTORY)
else()
  set(MOTORBRIDGE_ABI_LIBDIR "")
endif()

add_library(rebotarm_example_common INTERFACE)
target_include_directories(rebotarm_example_common INTERFACE
  "${CMAKE_CURRENT_LIST_DIR}/include"
  "${REBOTARM_CPP_DIR}/include"
  "${MOTORBRIDGE_ROOT}/bindings/cpp/include"
  "${MOTORBRIDGE_ROOT}/motor_abi/include"
  "${PINOCCHIO_PREFIX}/include")
target_link_libraries(rebotarm_example_common INTERFACE "${REBOTARM_MATH_LIB}")
if(CONDA_STDCXX_LIB)
  target_link_libraries(rebotarm_example_common INTERFACE "${CONDA_STDCXX_LIB}")
endif()
if(Eigen3_FOUND)
  target_link_libraries(rebotarm_example_common INTERFACE Eigen3::Eigen)
else()
  target_include_directories(rebotarm_example_common INTERFACE "${EIGEN3_INCLUDE_DIR}")
endif()

function(add_rebotarm_cpp_example name)
  add_executable(${name} "src/${name}.cpp")
  target_link_libraries(${name} PRIVATE rebotarm_example_common)
  set_target_properties(${name} PROPERTIES
    BUILD_RPATH "${REBOTARM_ROOT}/python/rebotarm_control_dm;${PINO_LIBDIR};${MOTORBRIDGE_ABI_LIBDIR}"
    INSTALL_RPATH "${REBOTARM_ROOT}/python/rebotarm_control_dm;${PINO_LIBDIR};${MOTORBRIDGE_ABI_LIBDIR}")
  target_link_options(${name} PRIVATE
    "-Wl,-rpath-link,${PINOCCHIO_PREFIX}/lib"
    "-Wl,-rpath,${PINOCCHIO_PREFIX}/lib")
endfunction()

function(add_rebotarm_cpp_hardware_example name)
  if(NOT MOTORBRIDGE_ABI_LIB)
    message(STATUS "motor_abi library not found; skip ${name}. Build it with: cargo build --manifest-path ${MOTORBRIDGE_ROOT}/motor_abi/Cargo.toml --release")
    return()
  endif()
  add_rebotarm_cpp_example(${name})
  target_link_libraries(${name} PRIVATE "${MOTORBRIDGE_ABI_LIB}")
endfunction()

add_rebotarm_cpp_hardware_example(0x01damiao_test)
add_rebotarm_cpp_hardware_example(1_damiao_text)
add_rebotarm_cpp_hardware_example(0x02_read_damiao_pd)
add_rebotarm_cpp_hardware_example(2_zero_and_read)
add_rebotarm_cpp_hardware_example(3_mit_control)
add_rebotarm_cpp_hardware_example(4_pos_vel_control)
add_rebotarm_cpp_example(5_fk_test)
add_rebotarm_cpp_example(6_ik_test)
add_rebotarm_cpp_hardware_example(7_arm_ik_control)
add_rebotarm_cpp_hardware_example(8_arm_traj_control)
add_rebotarm_cpp_hardware_example(9_gravity_compensation)
add_rebotarm_cpp_hardware_example(10_gravity_compensation_lock)
add_rebotarm_cpp_hardware_example(gripper_test)
