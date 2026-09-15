/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Op schema definitions and ROCm dispatch for aiter-sourced kernels.
// On ROCm builds, the CUDA dispatch key routes to adapter functions that
// convert at::Tensor -> aiter_tensor_t and call the aiter implementations.
// On CUDA builds, only the schema is registered (no impl); a Python-side
// @torch.library.impl could provide an alternative backend.

#include <torch/library.h>

#ifdef USE_ROCM
#include <mslk/aiter/aiter_bridge.h>
#include <rmsnorm_quant.h>
#include <hip/hip_runtime.h>

namespace {

at::Tensor aiter_rmsnorm_impl(
    const at::Tensor& input,
    const at::Tensor& weight,
    double epsilon) {
  auto out = torch::empty_like(input);

  auto a_out = mslk::aiter::to_aiter_tensor(out);
  auto a_input = mslk::aiter::to_aiter_tensor(input);
  auto a_weight = mslk::aiter::to_aiter_tensor(weight);

  aiter::rmsnorm(a_out, a_input, a_weight, epsilon);
  return out;
}

at::Tensor aiter_add_rmsnorm_impl(
    const at::Tensor& input,
    const at::Tensor& residual,
    const at::Tensor& weight,
    double epsilon) {
  auto out = torch::empty_like(input);
  auto residual_out = torch::empty_like(residual);

  auto a_out = mslk::aiter::to_aiter_tensor(out);
  auto a_input = mslk::aiter::to_aiter_tensor(input);
  auto a_residual_in = mslk::aiter::to_aiter_tensor(residual);
  auto a_residual_out = mslk::aiter::to_aiter_tensor(residual_out);
  auto a_weight = mslk::aiter::to_aiter_tensor(weight);

  aiter::add_rmsnorm(
      a_out, a_input, a_residual_in, a_residual_out, a_weight, epsilon);
  return out;
}

} // namespace
#endif // USE_ROCM

namespace mslk::aiter_ops {

TORCH_LIBRARY_FRAGMENT(mslk, m) {
#ifdef USE_ROCM
  m.def(
      "aiter_rmsnorm(Tensor input, Tensor weight, float epsilon) -> Tensor");
  m.def(
      "aiter_add_rmsnorm(Tensor input, Tensor residual, "
      "Tensor weight, float epsilon) -> Tensor");
  m.def(
      "aiter_topk_softmax(Tensor(a!) topk_weights, "
      "Tensor(b!) topk_indices, "
      "Tensor(c!) token_expert_indices, "
      "Tensor gating_output, "
      "bool renormalize) -> ()");
#endif
}

#ifdef USE_ROCM
TORCH_LIBRARY_IMPL(mslk, CUDA, m) {
  m.impl("aiter_rmsnorm", aiter_rmsnorm_impl);
  m.impl("aiter_add_rmsnorm", aiter_add_rmsnorm_impl);
  // aiter_topk_softmax impl is in asm_topksoftmax.cu
}
#endif

} // namespace mslk::aiter_ops
