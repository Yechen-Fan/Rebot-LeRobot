// pin::SE3 ↔ 4×4 齐次矩阵互转，使 Python 边界完全不暴露 pinocchio 类型。
#pragma once
#include <pinocchio/spatial/se3.hpp>
#include <Eigen/Dense>

namespace rebotarm {

inline pinocchio::SE3 se3_from_homogeneous(const Eigen::Matrix4d& M) {
  return pinocchio::SE3(M.block<3, 3>(0, 0), M.block<3, 1>(0, 3));
}

inline Eigen::Matrix4d homogeneous_from_se3(const pinocchio::SE3& T) {
  return T.toHomogeneousMatrix();
}

}  // namespace rebotarm
