#include "rebotarm/math_c_api.h"

#include "rebotarm/dynamics.hpp"
#include "rebotarm/robot_model.hpp"

#include <Eigen/Dense>
#include <pinocchio/math/rpy.hpp>

#include <exception>
#include <memory>
#include <string>

struct RebotarmMathModel {
  std::unique_ptr<rebotarm::RobotModel> model;
};

static thread_local std::string g_last_error;

static int fail(const std::string& message) {
  g_last_error = message;
  return -1;
}

static bool valid_model(const RebotarmMathModel* model) {
  return model != nullptr && model->model != nullptr;
}

static Eigen::VectorXd vector_from_ptr(const double* data, int len) {
  Eigen::VectorXd out(len);
  for (int i = 0; i < len; ++i) out[i] = data[i];
  return out;
}

static Eigen::Matrix4d matrix_from_row_major(const double* data) {
  Eigen::Matrix4d out;
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c) out(r, c) = data[r * 4 + c];
  return out;
}

static void matrix_to_row_major(const Eigen::Matrix4d& in, double* out) {
  if (!out) return;
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c) out[r * 4 + c] = in(r, c);
}

extern "C" {

RebotarmMathModel* rebotarm_math_model_new(const char* urdf_path) {
  try {
    if (!urdf_path) {
      fail("urdf_path is null");
      return nullptr;
    }
    auto* handle = new RebotarmMathModel;
    handle->model = std::make_unique<rebotarm::RobotModel>(urdf_path);
    g_last_error.clear();
    return handle;
  } catch (const std::exception& e) {
    fail(e.what());
  } catch (...) {
    fail("unknown exception");
  }
  return nullptr;
}

void rebotarm_math_model_free(RebotarmMathModel* model) { delete model; }

int rebotarm_math_model_nq(const RebotarmMathModel* model) {
  if (!valid_model(model)) return -1;
  return model->model->nq();
}

int rebotarm_math_model_nv(const RebotarmMathModel* model) {
  if (!valid_model(model)) return -1;
  return model->model->nv();
}

int rebotarm_math_end_frame_id(const RebotarmMathModel* model) {
  try {
    if (!valid_model(model)) return fail("model is null");
    return model->model->end_effector_frame_id();
  } catch (const std::exception& e) {
    return fail(e.what());
  }
}

int rebotarm_math_frame_id(const RebotarmMathModel* model, const char* frame_name) {
  try {
    if (!valid_model(model)) return fail("model is null");
    if (!frame_name) return fail("frame_name is null");
    return model->model->frame_id(frame_name);
  } catch (const std::exception& e) {
    return fail(e.what());
  }
}

int rebotarm_math_neutral(const RebotarmMathModel* model, double* out_q, int out_len) {
  try {
    if (!valid_model(model)) return fail("model is null");
    if (!out_q) return fail("out_q is null");
    const Eigen::VectorXd q = model->model->neutral();
    if (out_len < q.size()) return fail("out_q too small");
    for (int i = 0; i < q.size(); ++i) out_q[i] = q[i];
    return 0;
  } catch (const std::exception& e) {
    return fail(e.what());
  }
}

int rebotarm_math_fk(const RebotarmMathModel* model,
                     const double* q,
                     int q_len,
                     const char* frame_name,
                     double* out_xyz,
                     double* out_rpy,
                     double* out_T_row_major_4x4) {
  try {
    if (!valid_model(model)) return fail("model is null");
    if (!q) return fail("q is null");
    if (q_len != model->model->nq()) return fail("q length mismatch");
    const std::string frame = frame_name ? frame_name : "";
    const auto [pos, rot, T] = model->model->fk(vector_from_ptr(q, q_len), frame);
    if (out_xyz) {
      out_xyz[0] = pos.x();
      out_xyz[1] = pos.y();
      out_xyz[2] = pos.z();
    }
    if (out_rpy) {
      const Eigen::Vector3d rpy = pinocchio::rpy::matrixToRpy(rot);
      out_rpy[0] = rpy.x();
      out_rpy[1] = rpy.y();
      out_rpy[2] = rpy.z();
    }
    matrix_to_row_major(T, out_T_row_major_4x4);
    return 0;
  } catch (const std::exception& e) {
    return fail(e.what());
  }
}

int rebotarm_math_ik(const RebotarmMathModel* model,
                     const double* target_T_row_major_4x4,
                     const double* q_seed,
                     int q_len,
                     int frame_id,
                     int max_iter,
                     double tolerance,
                     double step_size,
                     double damping,
                     double* out_q,
                     RebotarmIkResult* out_result) {
  try {
    if (!valid_model(model)) return fail("model is null");
    if (!target_T_row_major_4x4) return fail("target matrix is null");
    if (!q_seed) return fail("q_seed is null");
    if (!out_q) return fail("out_q is null");
    if (q_len != model->model->nq()) return fail("q length mismatch");
    rebotarm::IKParams params;
    params.max_iter = max_iter;
    params.tolerance = tolerance;
    params.step_size = step_size;
    params.damping = damping;
    const auto result = model->model->solve_ik_with_retry(
        matrix_from_row_major(target_T_row_major_4x4),
        vector_from_ptr(q_seed, q_len),
        frame_id,
        params,
        8);
    if (result.q.size() != q_len) return fail("IK result length mismatch");
    for (int i = 0; i < q_len; ++i) out_q[i] = result.q[i];
    if (out_result) {
      out_result->success = result.success ? 1 : 0;
      out_result->error = result.error;
      out_result->iterations = result.iterations;
    }
    return 0;
  } catch (const std::exception& e) {
    return fail(e.what());
  }
}

int rebotarm_math_generalized_gravity(const RebotarmMathModel* model,
                                      const double* q,
                                      int q_len,
                                      double* out_tau,
                                      int out_len) {
  try {
    if (!valid_model(model)) return fail("model is null");
    if (!q) return fail("q is null");
    if (!out_tau) return fail("out_tau is null");
    if (q_len != model->model->nq()) return fail("q length mismatch");
    const Eigen::VectorXd tau = rebotarm::dyn::generalized_gravity(*model->model, vector_from_ptr(q, q_len));
    if (out_len < tau.size()) return fail("out_tau too small");
    for (int i = 0; i < tau.size(); ++i) out_tau[i] = tau[i];
    return 0;
  } catch (const std::exception& e) {
    return fail(e.what());
  }
}

const char* rebotarm_math_last_error(void) { return g_last_error.c_str(); }

}
