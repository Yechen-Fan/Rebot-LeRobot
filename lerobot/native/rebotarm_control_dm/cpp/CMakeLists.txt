cmake_minimum_required(VERSION 3.16)
project(rebotarm_math LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ---------------------------------------------------------------------------
# 自动探测 Pinocchio C++ 前缀（conda-forge / 系统安装；不依赖 ROS）。
# 优先级：-DPINOCCHIO_PREFIX > $PINOCCHIO_PREFIX > $CONDA_PREFIX > /usr/local > /usr
# 仅链接 C++ 核心库，绕开 pinocchioConfig.cmake（其会拖入 eigenpy/Boost.Python）。
# ---------------------------------------------------------------------------
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
    "未找到 Pinocchio C++。请先 `conda install -c conda-forge pinocchio`，"
    "或显式指定 -DPINOCCHIO_PREFIX=<prefix>。")
endif()
message(STATUS "Pinocchio prefix: ${PINOCCHIO_PREFIX}")
list(APPEND CMAKE_PREFIX_PATH "${PINOCCHIO_PREFIX}")

# pybind11（经 -Dpybind11_DIR=$(python -m pybind11 --cmakedir)）。
find_package(pybind11 REQUIRED)

# Eigen：优先 Eigen3Config，找不到则在前缀/系统路径搜索头文件。
find_package(Eigen3 QUIET CONFIG)
if(NOT Eigen3_FOUND)
  find_path(EIGEN3_INCLUDE_DIR Eigen/Dense
    PATHS "${PINOCCHIO_PREFIX}/include/eigen3" /usr/include/eigen3 /usr/local/include/eigen3
    REQUIRED)
  message(STATUS "Eigen include: ${EIGEN3_INCLUDE_DIR}")
endif()

# Pinocchio 库：自动适配 lib 与 lib/x86_64-linux-gnu。
find_library(PINOCCHIO_DEFAULT_LIB NAMES pinocchio_default pinocchio
  HINTS "${PINOCCHIO_PREFIX}/lib" "${PINOCCHIO_PREFIX}/lib/x86_64-linux-gnu" REQUIRED)
find_library(PINOCCHIO_PARSERS_LIB NAMES pinocchio_parsers
  HINTS "${PINOCCHIO_PREFIX}/lib" "${PINOCCHIO_PREFIX}/lib/x86_64-linux-gnu")
get_filename_component(PINO_LIBDIR "${PINOCCHIO_DEFAULT_LIB}" DIRECTORY)
message(STATUS "Pinocchio lib dir: ${PINO_LIBDIR}")
find_library(URDFDOM_MODEL_LIB NAMES urdfdom_model
  HINTS "${PINOCCHIO_PREFIX}/lib" "${PINOCCHIO_PREFIX}/lib/x86_64-linux-gnu"
        /usr/local/lib /usr/lib/x86_64-linux-gnu)
find_library(CONSOLE_BRIDGE_LIB NAMES console_bridge
  HINTS "${PINOCCHIO_PREFIX}/lib" "${PINOCCHIO_PREFIX}/lib/x86_64-linux-gnu"
        /usr/local/lib /usr/lib/x86_64-linux-gnu)
find_library(TINYXML2_LIB NAMES tinyxml2
  HINTS "${PINOCCHIO_PREFIX}/lib" "${PINOCCHIO_PREFIX}/lib/x86_64-linux-gnu"
        /usr/local/lib /usr/lib/x86_64-linux-gnu)

set(REBOTARM_MATH_CORE_SOURCES
  src/robot_model.cpp
  src/dynamics.cpp
  src/trajectory.cpp
  src/math_c_api.cpp
)

add_library(rebotarm_math SHARED
  ${REBOTARM_MATH_CORE_SOURCES}
)
target_include_directories(rebotarm_math PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${PINOCCHIO_PREFIX}/include
)
target_compile_definitions(rebotarm_math PUBLIC PINOCCHIO_WITH_URDFDOM)
target_link_libraries(rebotarm_math PUBLIC ${PINOCCHIO_DEFAULT_LIB})
if(PINOCCHIO_PARSERS_LIB)
  target_link_libraries(rebotarm_math PUBLIC ${PINOCCHIO_PARSERS_LIB})
endif()
if(URDFDOM_MODEL_LIB)
  target_link_libraries(rebotarm_math PUBLIC ${URDFDOM_MODEL_LIB})
endif()
if(CONSOLE_BRIDGE_LIB)
  target_link_libraries(rebotarm_math PUBLIC ${CONSOLE_BRIDGE_LIB})
endif()
if(TINYXML2_LIB)
  target_link_libraries(rebotarm_math PUBLIC ${TINYXML2_LIB})
endif()
if(Eigen3_FOUND)
  target_link_libraries(rebotarm_math PUBLIC Eigen3::Eigen)
else()
  target_include_directories(rebotarm_math PUBLIC ${EIGEN3_INCLUDE_DIR})
endif()
set_target_properties(rebotarm_math PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../python/rebotarm_control_dm
  OUTPUT_NAME "rebotarm_math"
  BUILD_RPATH "${PINO_LIBDIR}"
  INSTALL_RPATH "${PINO_LIBDIR}")

pybind11_add_module(_math
  src/bindings.cpp
  src/arm_endpos.cpp
)
target_include_directories(_math PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${PINOCCHIO_PREFIX}/include
)
target_compile_definitions(_math PRIVATE PINOCCHIO_WITH_URDFDOM)
target_link_libraries(_math PRIVATE rebotarm_math)
if(Eigen3_FOUND)
  target_link_libraries(_math PRIVATE Eigen3::Eigen)
else()
  target_include_directories(_math PRIVATE ${EIGEN3_INCLUDE_DIR})
endif()

# 直接把 .so 输出到 Python 包目录（被 maturin 一并打包）；RPATH 指向 Pinocchio 库目录。
set_target_properties(_math PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../python/rebotarm_control_dm
  OUTPUT_NAME "_math"
  BUILD_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/../python/rebotarm_control_dm;${PINO_LIBDIR}"
  INSTALL_RPATH "$ORIGIN;${PINO_LIBDIR}")
