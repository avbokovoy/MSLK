# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""Meta (shape-inference) registrations for aiter-sourced C++ ops.

Each registration is guarded by ``hasattr(torch.ops.mslk, ...)`` so the
module is importable even when the C++ library was built without ROCm.
"""

import torch


if hasattr(torch.ops.mslk, "aiter_rmsnorm"):

    @torch.library.impl("mslk::aiter_rmsnorm", "Meta")
    def _rmsnorm_meta(
        input: torch.Tensor, weight: torch.Tensor, epsilon: float
    ) -> torch.Tensor:
        return torch.empty_like(input)


if hasattr(torch.ops.mslk, "aiter_add_rmsnorm"):

    @torch.library.impl("mslk::aiter_add_rmsnorm", "Meta")
    def _add_rmsnorm_meta(
        input: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        epsilon: float,
    ) -> torch.Tensor:
        return torch.empty_like(input)


if hasattr(torch.ops.mslk, "aiter_topk_softmax"):

    @torch.library.impl("mslk::aiter_topk_softmax", "Meta")
    def _topk_softmax_meta(
        topk_weights: torch.Tensor,
        topk_indices: torch.Tensor,
        token_expert_indices: torch.Tensor,
        gating_output: torch.Tensor,
        renormalize: bool,
    ) -> None:
        pass
