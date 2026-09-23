# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Originally from ROCm/aiter (MIT License)
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# pyre-unsafe

import torch


def topk_softmax(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = True,
) -> None:
    """Top-K softmax using aiter's hand-tuned assembly kernel (ROCm only).

    Computes top-K selection and softmax over expert gating scores in a single
    fused GPU kernel.  The kernel is a pre-compiled HSACO code object loaded at
    runtime via ``hipModuleLoad``.

    All output tensors are written **in-place**.

    Args:
        topk_weights: Output float32 tensor of shape (num_tokens, topk).
        topk_indices: Output int32 tensor of shape (num_tokens, topk).
        token_expert_indices: Output int32 tensor of shape (num_tokens, topk).
        gating_output: Input tensor of shape (num_tokens, num_experts),
            dtype float32 or bfloat16.
        renormalize: Whether to renormalize the softmax weights.
    """
    torch.ops.mslk.aiter_topk_softmax(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
    )
