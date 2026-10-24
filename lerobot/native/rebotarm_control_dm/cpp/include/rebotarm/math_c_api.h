#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RebotarmMathModel RebotarmMathModel;

typedef struct RebotarmIkResult {
  int success;
  double error;
  int iterations;
} RebotarmIkResult;

RebotarmMathModel* rebotarm_math_model_new(const char* urdf_path);
void rebotarm_math_model_free(RebotarmMathModel* model);

int rebotarm_math_model_nq(const RebotarmMathModel* model);
int rebotarm_math_model_nv(const RebotarmMathModel* model);
int rebotarm_math_end_frame_id(const RebotarmMathModel* model);
int rebotarm_math_frame_id(const RebotarmMathModel* model, const char* frame_name);

int rebotarm_math_neutral(const RebotarmMathModel* model, double* out_q, int out_len);

int rebotarm_math_fk(
    const RebotarmMathModel* model,
    const double* q,
    int q_len,
    const char* frame_name,
    double* out_xyz,
    double* out_rpy,
    double* out_T_row_major_4x4);

int rebotarm_math_ik(
    const RebotarmMathModel* model,
    const double* target_T_row_major_4x4,
    const double* q_seed,
    int q_len,
    int frame_id,
    int max_iter,
    double tolerance,
    double step_size,
    double damping,
    double* out_q,
    RebotarmIkResult* out_result);

int rebotarm_math_generalized_gravity(
    const RebotarmMathModel* model,
    const double* q,
    int q_len,
    double* out_tau,
    int out_len);

const char* rebotarm_math_last_error(void);

#ifdef __cplusplus
}
#endif
