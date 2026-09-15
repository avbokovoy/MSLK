/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Originally from ROCm/aiter (MIT License)
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

// Assembly kernel wrapper for topk_softmax.  Loads a pre-compiled HSACO
// code object (.co file) at runtime and launches it via hipModuleLaunchKernel.
// The config header is generated at build time by aiter's hsa/codegen.py.

#ifdef USE_ROCM

#include <ATen/ATen.h>
#include <mslk/aiter/aiter_bridge.h>
#include <torch/library.h>

#include "aiter_hip_common.h"
#include "aiter_tensor.h"
#include "asm_topksoftmax_configs.hpp"

namespace {

struct __attribute__((packed)) KernelArgs {
  void* ptr_T;
  p2 _p0;
  void* ptr_W;
  p2 _p1;
  void* ptr_A;
  p2 _p2;
  unsigned int batch;
  p3 _p4;
  unsigned int expert;
  p3 _p5;
  unsigned int topk;
  p3 _p6;
  unsigned int renormalize;
  p3 _p7;
  unsigned int out_stride;
  p3 _p8;
};

std::pair<std::string, int> get_heuristic_kernel_topksoftmax(
    std::string arch_id,
    std::string dtype,
    int max_subm,
    int num_experts,
    int topk,
    CFG* cfgs) {
  int subm = -1;
  std::string kernelName;
  for (const auto& el : *cfgs) {
    if (el.first.find(arch_id) != 0)
      continue;
    const auto& cfg = el.second;
    if (cfg.dtype != dtype || cfg.subm > max_subm ||
        cfg.num_experts != num_experts || cfg.topk != topk)
      continue;
    if (cfg.subm > subm) {
      kernelName = el.first;
      subm = cfg.subm;
    }
  }
  AITER_CHECK(
      !kernelName.empty(),
      __func__,
      ": cannot get heuristic kernel!"
      " arch_id:",
      arch_id,
      " dtype:",
      dtype,
      " max_subm:",
      max_subm,
      " num_experts:",
      num_experts,
      " topk:",
      topk);
  return {kernelName, subm};
}

void aiter_topk_softmax_impl(
    at::Tensor& topk_weights,
    at::Tensor& topk_indices,
    at::Tensor& token_expert_indices,
    const at::Tensor& gating_output,
    bool renormalize) {
  // Convert to aiter types for shape/dtype queries
  auto a_gating = mslk::aiter::to_aiter_tensor(gating_output);
  auto a_topk_w = mslk::aiter::to_aiter_tensor(topk_weights);

  std::string arch_id = get_gpu_arch();
  const uint num_experts = a_gating.size(-1);
  const uint num_tokens = a_gating.numel() / num_experts;
  const uint topk = a_topk_w.size(-1);
  const uint out_stride = a_topk_w.stride(0);
  const uint MAX_SUBM = num_tokens < 10000 ? 4 : 12;

  std::string dtype;
  if (a_gating.dtype() == AITER_DTYPE_fp32)
    dtype = "fp32";
  else if (a_gating.dtype() == AITER_DTYPE_bf16)
    dtype = "bf16";
  else
    AITER_CHECK(
        false,
        __func__,
        ": unsupported gating_output dtype: ",
        AiterDtype_to_str(a_gating.dtype()));

  AITER_CHECK(
      topk_weights.scalar_type() == c10::ScalarType::Float,
      __func__,
      ": topk_weights must be float32");
  AITER_CHECK(
      topk_indices.scalar_type() == c10::ScalarType::Int,
      __func__,
      ": topk_indices must be int32");

  KernelArgs args;
  size_t arg_size = sizeof(args);
  args.ptr_T = topk_indices.data_ptr();
  args.ptr_W = topk_weights.data_ptr();
  args.ptr_A = gating_output.data_ptr();
  args.batch = num_tokens;
  args.expert = num_experts;
  args.topk = topk;
  args.renormalize = renormalize ? 1 : 0;
  args.out_stride = out_stride * 4;

  CFG* config_map = &cfg_topksoftmax;
  static SynchronizedCache<std::string_view, AiterAsmKernel> impl_ptr_map;
  AiterAsmKernel* impl_ptr = nullptr;
  auto [kernelName, subm] = get_heuristic_kernel_topksoftmax(
      arch_id, dtype, MAX_SUBM, num_experts, topk, config_map);

  auto it = config_map->find(kernelName);
  AITER_CHECK(
      it != config_map->end(), __func__, " not find kernel " + kernelName);

  const auto& cfg = it->second;
  const char* name = cfg.knl_name.c_str();
  const char* co_name = cfg.co_name.c_str();

  impl_ptr = &impl_ptr_map.get_or_create(
      name, [&]() { return AiterAsmKernel(name, co_name); });

  const HipDeviceGuard device_guard(gating_output.device().index());

  uint gdx = (num_tokens + subm - 1) / subm;
  AITER_CHECK(gdx >> 31 == 0, "num_tokens too large: ", num_tokens);
  hipStream_t stream = c10::hip::getCurrentHIPStream().stream();
  impl_ptr->launch_kernel(
      {&args,
       &arg_size,
       static_cast<int>(gdx),
       1,
       1,
       256,
       1,
       1,
       stream});
}

} // namespace

namespace mslk::aiter_asm {

TORCH_LIBRARY_IMPL(mslk, CUDA, m) {
  m.impl("aiter_topk_softmax", aiter_topk_softmax_impl);
}

} // namespace mslk::aiter_asm

#endif // USE_ROCM
