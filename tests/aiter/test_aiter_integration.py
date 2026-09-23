# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Integration tests for aiter kernels exposed through the mslk namespace.

These tests verify that all three kernel categories (Triton, C++/HIP,
assembly) are callable through ``torch.ops.mslk.*`` and produce correct
results.  All tests require a ROCm GPU and are skipped otherwise.
"""

import unittest

import torch

_IS_ROCM = torch.version.hip is not None and torch.cuda.is_available()


def _skip_unless_rocm(reason: str = "ROCm required"):
    return unittest.skipUnless(_IS_ROCM, reason)


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------


def _reference_rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    return (x_normed * weight).to(x.dtype)


def _reference_topk_softmax(
    gating: torch.Tensor, topk: int, renormalize: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.softmax(gating.float(), dim=-1)
    topk_weights, topk_indices = torch.topk(scores, topk, dim=-1)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights, topk_indices.to(torch.int32)


# ---------------------------------------------------------------------------
# Test: Triton softmax
# ---------------------------------------------------------------------------


@_skip_unless_rocm()
class TestAiterSoftmax(unittest.TestCase):
    def test_basic(self) -> None:
        from mslk.aiter.softmax import softmax

        x = torch.randn(128, 256, device="cuda", dtype=torch.float32)
        result = softmax(x)
        expected = torch.nn.functional.softmax(x, dim=-1)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_large_cols(self) -> None:
        from mslk.aiter.softmax import softmax

        x = torch.randn(32, 8192, device="cuda", dtype=torch.float32)
        result = softmax(x)
        expected = torch.nn.functional.softmax(x, dim=-1)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_dispatch(self) -> None:
        """Verify the op is callable through torch.ops.mslk dispatch."""
        x = torch.randn(64, 128, device="cuda", dtype=torch.float32)
        result = torch.ops.mslk.aiter_softmax(x)
        expected = torch.nn.functional.softmax(x, dim=-1)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Test: C++/HIP RMSNorm
# ---------------------------------------------------------------------------


@_skip_unless_rocm()
@unittest.skipUnless(
    _IS_ROCM
    and hasattr(torch.ops, "mslk")
    and hasattr(torch.ops.mslk, "aiter_rmsnorm"),
    "aiter_rmsnorm op not registered (mslk.so built without USE_ROCM?)",
)
class TestAiterRmsnorm(unittest.TestCase):
    def test_basic(self) -> None:
        from mslk.aiter.rmsnorm import rmsnorm

        hidden = 4096
        x = torch.randn(8, hidden, device="cuda", dtype=torch.bfloat16)
        w = torch.ones(hidden, device="cuda", dtype=torch.bfloat16)
        eps = 1e-6

        result = rmsnorm(x, w, eps)
        expected = _reference_rmsnorm(x, w, eps)
        torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)

    def test_nonunit_weight(self) -> None:
        from mslk.aiter.rmsnorm import rmsnorm

        hidden = 2048
        x = torch.randn(4, hidden, device="cuda", dtype=torch.float16)
        w = torch.randn(hidden, device="cuda", dtype=torch.float16).abs() + 0.1
        eps = 1e-5

        result = rmsnorm(x, w, eps)
        expected = _reference_rmsnorm(x, w, eps)
        torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# Test: Assembly topk_softmax
# ---------------------------------------------------------------------------


@_skip_unless_rocm()
@unittest.skipUnless(
    _IS_ROCM
    and hasattr(torch.ops, "mslk")
    and hasattr(torch.ops.mslk, "aiter_topk_softmax"),
    "aiter_topk_softmax op not registered (mslk.so built without USE_ROCM?)",
)
class TestAiterTopkSoftmax(unittest.TestCase):
    def test_basic(self) -> None:
        from mslk.aiter.topk_softmax import topk_softmax

        num_tokens = 64
        num_experts = 128
        topk_val = 8

        gating = torch.randn(
            num_tokens, num_experts, device="cuda", dtype=torch.float32
        )
        topk_weights = torch.empty(
            num_tokens, topk_val, device="cuda", dtype=torch.float32
        )
        topk_indices = torch.empty(
            num_tokens, topk_val, device="cuda", dtype=torch.int32
        )
        token_expert_indices = torch.empty(
            num_tokens, topk_val, device="cuda", dtype=torch.int32
        )

        topk_softmax(topk_weights, topk_indices, token_expert_indices, gating, True)

        ref_weights, ref_indices = _reference_topk_softmax(gating, topk_val, True)

        # The selected experts should match (order may vary within a row).
        for i in range(num_tokens):
            asm_set = set(topk_indices[i].tolist())
            ref_set = set(ref_indices[i].tolist())
            self.assertEqual(asm_set, ref_set, f"row {i}: expert sets differ")

        # Weights should sum to ~1.0 per row when renormalized.
        row_sums = topk_weights.sum(dim=-1)
        torch.testing.assert_close(
            row_sums,
            torch.ones_like(row_sums),
            atol=1e-4,
            rtol=1e-4,
        )


if __name__ == "__main__":
    unittest.main()
