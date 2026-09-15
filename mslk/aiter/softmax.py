# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""Thin wrapper that delegates to aiter's Triton softmax kernel.

This demonstrates the runtime-dependency pattern for integrating aiter's
Python/Triton kernels: ``import aiter`` and call the op directly, then
register it under the ``mslk`` torch dispatch namespace so it is accessible
as ``torch.ops.mslk.aiter_softmax`` and participates in torch.compile / Meta.

No source code is vendored — aiter must be installed as a Python package.
"""

import torch

from aiter.ops.triton.softmax import softmax as _aiter_softmax


def softmax(x: torch.Tensor) -> torch.Tensor:
    """Row-wise softmax using aiter's online-softmax Triton kernel.

    Args:
        x: 2-D input tensor (n_rows, n_cols) on GPU.

    Returns:
        Tensor with the same shape, softmax applied along the last dimension.
    """
    return _aiter_softmax(x)


# ---------------------------------------------------------------------------
# Register under ``torch.ops.mslk`` so it participates in torch dispatch.
# ---------------------------------------------------------------------------

torch.library.define(
    "mslk::aiter_softmax",
    "(Tensor x) -> Tensor",
)


@torch.library.impl("mslk::aiter_softmax", "Meta")
def _aiter_softmax_meta(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


if torch.version.hip is not None:

    @torch.library.impl("mslk::aiter_softmax", "CUDA")
    def _aiter_softmax_rocm(x: torch.Tensor) -> torch.Tensor:
        return _aiter_softmax(x)
