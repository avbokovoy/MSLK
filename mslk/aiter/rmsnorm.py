# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Originally from ROCm/aiter (MIT License)
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# pyre-unsafe

import torch


def rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """RMSNorm using aiter's HIP kernel (ROCm only).

    Args:
        input: Input tensor of shape (..., hidden_size).
        weight: Learnable scale of shape (hidden_size,).
        epsilon: Small constant for numerical stability.

    Returns:
        Normalized tensor with the same shape as *input*.
    """
    return torch.ops.mslk.aiter_rmsnorm(input, weight, epsilon)


def add_rmsnorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Fused residual-add + RMSNorm using aiter's HIP kernel (ROCm only).

    Args:
        input: Input tensor of shape (..., hidden_size).
        residual: Residual tensor of the same shape.
        weight: Learnable scale of shape (hidden_size,).
        epsilon: Small constant for numerical stability.

    Returns:
        Normalized tensor with the same shape as *input*.
    """
    return torch.ops.mslk.aiter_add_rmsnorm(input, residual, weight, epsilon)
