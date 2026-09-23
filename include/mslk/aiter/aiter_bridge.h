/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Bridge between MSLK's at::Tensor world and aiter's aiter_tensor_t world.
// Include this header in any MSLK C++ file that needs to call aiter kernels.
// Only used in ROCm builds (all consumers are gated in MslkDefault.cmake).

#pragma once

#include <ATen/ATen.h>
#include <aiter_tensor.h>

namespace mslk::aiter {

inline AiterDtype torch_to_aiter_dtype(c10::ScalarType st) {
  switch (st) {
    case c10::ScalarType::Float:
      return AITER_DTYPE_fp32;
    case c10::ScalarType::Half:
      return AITER_DTYPE_fp16;
    case c10::ScalarType::BFloat16:
      return AITER_DTYPE_bf16;
    case c10::ScalarType::Float8_e4m3fnuz:
      return AITER_DTYPE_fp8;
    case c10::ScalarType::Int:
      return AITER_DTYPE_i32;
    case c10::ScalarType::Long:
      return AITER_DTYPE_i64;
    case c10::ScalarType::Short:
      return AITER_DTYPE_i16;
    case c10::ScalarType::Char:
      return AITER_DTYPE_i8;
    case c10::ScalarType::Byte:
      return AITER_DTYPE_u8;
    default:
      TORCH_CHECK(
          false, "Unsupported dtype for aiter bridge: ", toString(st));
  }
}

inline aiter_tensor_t to_aiter_tensor(const at::Tensor& t) {
  TORCH_CHECK(t.dim() <= 8, "aiter_tensor_t supports at most 8 dimensions");
  aiter_tensor_t a{};
  a.ptr = t.data_ptr();
  a.numel_ = static_cast<size_t>(t.numel());
  a.ndim = t.dim();
  for (int i = 0; i < t.dim(); i++) {
    a.shape[i] = t.size(i);
    a.strides[i] = t.stride(i);
  }
  a.dtype_ = torch_to_aiter_dtype(t.scalar_type());
  a.device_id = t.device().index();
  return a;
}

} // namespace mslk::aiter
