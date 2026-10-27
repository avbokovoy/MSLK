# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dropout tests for the FlyDSL attention ops.

flydsl.FwOp draws CK's Philox dropout mask (mslk.attention.flydsl.dropout_philox)
from a (seed, offset) reserved on the default generator exactly as CK's forward
does, and stores it as rng_state. flydsl.BwOp regenerates the same mask from
rng_state on gfx950, and since the mask is CK's, ck.BwOp accepts the same
rng_state. Scoped to the op's dropout subset: dense self-attention
(q_len == kv_len), f16/bf16, multiple heads, head_dim in {64, 128}.
"""

import numpy as np
import pytest
import torch
from mslk.attention import fmha
from mslk.attention.fmha.common import Context
from scipy.stats import binomtest  # type: ignore

from .test_mem_eff_attention import _vec_binom_test
from .utils import assert_allclose, ref_attention_for_test, rocm_only


def _generator_state():
    gen = torch.cuda.default_generators[torch.cuda.current_device()]
    return gen.initial_seed(), gen.get_offset()


def _drop_mask(batch_size, num_heads, q_len, kv_len, p, seed, offset):
    from mslk.attention.flydsl.dropout_philox import philox_dropout_keep_mask

    keep = philox_dropout_keep_mask(
        batch_size, num_heads, q_len, kv_len, p, seed, offset, torch.device("cuda")
    )
    return keep.to(torch.float32)


def _keep_probability(p):
    # CK keeps a random byte <= floor((1 - p) * 255), i.e. (threshold + 1) / 256.
    from mslk.attention.flydsl.dropout_philox import philox_keep_threshold

    return (philox_keep_threshold(p) + 1) / 256


def _ck_reference_randval(batch, heads, q_len, kv_len, seed, offset):
    """Transcription of ck_tile/host/reference/reference_batched_dropout_randval.hpp."""
    from mslk.attention.flydsl import dropout_philox as dp

    m32 = 0xFFFFFFFF

    def philox(counter_lo, subsequence):
        c = [counter_lo & m32, counter_lo >> 32, subsequence & m32, subsequence >> 32]
        k = [seed & m32, seed >> 32]
        for _ in range(dp.PHILOX_ROUNDS):
            p0, p1 = dp.PHILOX_M0 * c[0], dp.PHILOX_M1 * c[2]
            c = [
                ((p1 >> 32) ^ c[1] ^ k[0]) & m32,
                p1 & m32,
                ((p0 >> 32) ^ c[3] ^ k[1]) & m32,
                p0 & m32,
            ]
            k = [(k[0] + dp.PHILOX_W0) & m32, (k[1] + dp.PHILOX_W1) & m32]
        return np.array(c, dtype="<u4").view(np.uint8)

    out = np.zeros((batch, heads, q_len, kv_len), dtype=np.uint8)
    for b in range(batch):
        for h in range(heads):
            for tm in range((q_len + 31) // 32):
                for tn in range((kv_len + 31) // 32):
                    for lane in range(64):
                        counter = (offset + (b * heads + h) * 64 + lane) & (2**64 - 1)
                        rand = philox(counter, tm | tn << 32)
                        for r in range(16):
                            i = 16 * (r // 8) + 8 * (lane // 32) + r % 8
                            m, n = tm * 32 + i, tn * 32 + lane % 32
                            if m < q_len and n < kv_len:
                                out[b, h, m, n] = rand[r]
    return out


@rocm_only
@pytest.mark.parametrize(
    "shape,seed,offset",
    [
        ((1, 1, 32, 32), 0, 0),
        ((2, 3, 70, 45), 0x123456789ABCDEF0, 0xFFFFFF80),  # counter carries into hi
        ((1, 2, 33, 64), 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFF00000040),
    ],
)
@pytest.mark.parametrize("p", [0.1, 0.5])
def test_philox_keep_mask_matches_ck(shape, seed, offset, p):
    from mslk.attention.flydsl.dropout_philox import philox_keep_threshold

    ref = _ck_reference_randval(*shape, seed, offset) <= philox_keep_threshold(p)
    keep = _drop_mask(*shape, p, seed, offset).cpu().numpy().astype(bool)
    assert (keep == ref).all(), f"{int((keep != ref).sum())} mismatching elements"


@rocm_only
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("attn_bias", [None, fmha.attn_bias.LowerTriangularMask()])
@pytest.mark.parametrize("seed", [42, 124])
@pytest.mark.parametrize("p", [0.3, 0.7])
@pytest.mark.parametrize("k_len", [64, 128])
@pytest.mark.parametrize("h_len", [1, 3])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seq_len", [32, 256])
def test_dropout_flydsl(seq_len, batch_size, h_len, k_len, p, seed, attn_bias, dtype):
    op = fmha.flydsl.FwOp
    device = "cuda"
    scale = 3
    # Dense self-attention only (q_len == kv_len). BMHK so we exercise >1 head.
    q_len = kv_len = seq_len

    query = (
        torch.randn((batch_size, q_len, h_len, k_len), device=device, dtype=dtype)
        * scale
    )
    key = (
        torch.randn((batch_size, kv_len, h_len, k_len), device=device, dtype=dtype)
        * scale
    )
    value = (
        torch.randn((batch_size, kv_len, h_len, k_len), device=device, dtype=dtype)
        * scale
    )

    if not op.supports(fmha.Inputs(query, key, value, attn_bias, p, None)):
        pytest.skip(f"{op.NAME}: unsupported input")

    torch.manual_seed(seed)
    out = fmha.memory_efficient_attention(
        query, key, value, attn_bias, p, op=(op, None)
    )
    torch.manual_seed(seed)
    out2 = fmha.memory_efficient_attention(
        query, key, value, attn_bias, p, op=(op, None)
    )
    assert_allclose(out, out2, "dropout reproducibility")

    # Correctness: the op reserves its Philox (seed, offset) from the default
    # generator, so reseeding to `seed` and rebuilding the mask from the same
    # generator state reproduces the op's exact mask; apply it to a head-by-head
    # reference and compare. Dropout applies to the attention weights (not the
    # output), so this masked reference is the only faithful check.
    torch.manual_seed(seed)
    mask = _drop_mask(batch_size, h_len, q_len, kv_len, p, *_generator_state())
    ref = torch.stack(
        [
            ref_attention_for_test(
                query[:, :, h], key[:, :, h], value[:, :, h], attn_bias, mask[:, h], p
            )
            for h in range(h_len)
        ],
        dim=2,
    )
    # A wrong mask diverges in thousands of elements; heavy dropout (p=0.7) +
    # causal + bf16 leaves a handful of near-floor outputs at the bf16 floor
    # (~0.04) that can exceed tolerance, so allow a tiny fraction of outliers while
    # still catching any mask mismatch.
    of = out.float()
    close = torch.isclose(of, ref.float(), atol=5e-2, rtol=2e-2)
    n_bad = int((~close).sum())
    assert n_bad <= max(8, of.numel() // 2000), (
        f"{n_bad}/{of.numel()} elements exceed tolerance (likely wrong dropout mask)"
    )

    # Statistical: each trial uses a fresh, non-overlapping Philox counter range.
    num_trials = 1000
    p_val_tol = 1e-6
    keep_prob = _keep_probability(p)
    masks = []
    for trial in range(num_trials):
        offset = trial * batch_size * h_len * 64
        masks.append(
            _drop_mask(batch_size, h_len, q_len, kv_len, p, seed, offset).cpu()
        )
    masks = torch.stack(masks, dim=0)
    p_value = binomtest(int(masks.sum()), masks.numel(), p=keep_prob).pvalue
    assert p_value > p_val_tol, p_value
    # Per-element check allows a few spurious outliers (~N*1e-6 expected at this tol).
    masks = masks.sum(0).flatten()
    p_values = _vec_binom_test(masks.numpy(), num_trials, p=keep_prob)
    n_outliers = int((p_values <= p_val_tol).sum())
    assert n_outliers <= max(1, masks.numel() // 50000), (
        f"{n_outliers} per-element binomial outliers of {masks.numel()}"
    )


@rocm_only
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("p", [0.0, 0.3, 0.7])
@pytest.mark.parametrize("k_len", [64, 128])
@pytest.mark.parametrize("h_len", [1, 3])
@pytest.mark.parametrize("seq_len", [32, 256])
def test_dropout_backward_flydsl(seq_len, h_len, k_len, p, dtype):
    # The FlyDSL forward draws CK's Philox mask and stores CK's rng_state, so
    # ck.BwOp must regenerate the identical mask. Compare the full fw+bw against
    # CK end-to-end under the same seed.
    if not fmha.ck.BwOp.is_available():
        pytest.skip("ck.BwOp unavailable")
    flyF = fmha.flydsl.FwOp
    device = "cuda"
    scale = 3
    q_len = kv_len = seq_len

    def mk():
        torch.manual_seed(0)
        q = torch.randn((1, q_len, h_len, k_len), device=device, dtype=dtype) * scale
        k = torch.randn((1, kv_len, h_len, k_len), device=device, dtype=dtype) * scale
        v = torch.randn((1, kv_len, h_len, k_len), device=device, dtype=dtype) * scale
        return q, k, v

    q, k, v = mk()
    inp = fmha.Inputs(query=q, key=k, value=v, attn_bias=None, p=p)
    if not flyF.supports(inp):
        pytest.skip(f"{flyF.NAME}: unsupported input")

    grad_out = torch.randn((1, q_len, h_len, k_len), device=device, dtype=dtype)

    torch.manual_seed(77)
    out_f, ctx_f = flyF.apply(inp, needs_gradient=True)
    if p != 0.0:
        assert ctx_f.rng_state is not None
        assert ctx_f.rng_state.dtype == torch.int64
        assert tuple(ctx_f.rng_state.shape) == (2,)
        assert ctx_f.op_bw is fmha.flydsl.BwOp
    g_f = fmha.ck.BwOp.apply(ctx_f, inp, grad_out)

    torch.manual_seed(77)
    out_c, ctx_c = fmha.ck.FwOp.apply(inp, needs_gradient=True)
    g_c = fmha.ck.BwOp.apply(ctx_c, inp, grad_out)

    if dtype is torch.float16:
        # f16 (10-bit mantissa): flydslF+ck.BwOp matches ck end-to-end per element.
        # atol 5e-2 covers the handful of near-zero elements at the f16 floor for
        # kv>=256 (cancellation in the backward); rtol governs the signal.
        assert_allclose(out_f.float(), out_c.float(), "fwd", atol=5e-2, rtol=2e-2)
        assert_allclose(g_f.dq.float(), g_c.dq.float(), "dq", atol=5e-2, rtol=2e-2)
        assert_allclose(g_f.dk.float(), g_c.dk.float(), "dk", atol=5e-2, rtol=2e-2)
        assert_allclose(g_f.dv.float(), g_c.dv.float(), "dv", atol=5e-2, rtol=2e-2)
    else:
        # bf16 (8-bit mantissa): the attention backward is cancellation-dominated at
        # near-zero gradients (and outputs), where bf16 has large per-element error
        # regardless of kernel. Compare the aggregate relative L2 norm instead:
        # robust to those outliers, but a wrong dropout mask still blows the norm up
        # well past the thresholds.
        def _rel_l2(a, b):
            return (a - b).float().norm() / b.float().norm().clamp_min(1e-6)

        for name, ga, gb, tol in (
            ("fwd", out_f, out_c, 5e-2),
            ("dq", g_f.dq, g_c.dq, 1e-1),
            ("dk", g_f.dk, g_c.dk, 1e-1),
            ("dv", g_f.dv, g_c.dv, 1e-1),
        ):
            rel = _rel_l2(ga, gb)
            assert rel < tol, f"{name}: relative L2 {rel:.4f} exceeds {tol}"


@rocm_only
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("p", [0.0, 0.3])
@pytest.mark.parametrize("hq_len", [1, 3])
@pytest.mark.parametrize("g_len", [2])
@pytest.mark.parametrize("seq_len", [32, 128])
def test_dropout_backward_bmghk_flydsl(seq_len, g_len, hq_len, p, dtype):
    # BMGHK (5-D) dropout with gradients: the op folds G into H, runs the dense
    # dropout forward, then re-wraps the 5-D context. That re-wrap must PRESERVE the
    # philox rng_state (regression guard) so ck.BwOp can regenerate the 5-D mask
    # instead of raising. Compare the full fw+bw against ck end to end.
    if not fmha.ck.BwOp.is_available():
        pytest.skip("ck.BwOp unavailable")
    flyF = fmha.flydsl.FwOp
    device = "cuda"
    scale = 3
    k_len = 128
    q_len = kv_len = seq_len

    def mk():
        torch.manual_seed(0)
        qs = (1, q_len, g_len, hq_len, k_len)
        kvs = (1, kv_len, g_len, hq_len, k_len)
        return (
            torch.randn(qs, device=device, dtype=dtype) * scale,
            torch.randn(kvs, device=device, dtype=dtype) * scale,
            torch.randn(kvs, device=device, dtype=dtype) * scale,
        )

    q, k, v = mk()
    inp = fmha.Inputs(query=q, key=k, value=v, attn_bias=None, p=p)
    if not flyF.supports(inp):
        pytest.skip(f"{flyF.NAME}: unsupported input")

    grad_out = torch.randn((1, q_len, g_len, hq_len, k_len), device=device, dtype=dtype)

    torch.manual_seed(77)
    out_f, ctx_f = flyF.apply(inp, needs_gradient=True)
    assert out_f.shape == q.shape
    # LSE is unflattened back to the (G, Hq) head split.
    assert tuple(ctx_f.lse.shape) == (1, g_len, hq_len, q_len), ctx_f.lse.shape
    if p != 0.0:
        # The 5-D re-wrap must keep rng_state, else ck.BwOp gets None and raises.
        assert ctx_f.rng_state is not None
        assert ctx_f.op_bw is fmha.flydsl.BwOp

    # ck.BwOp runs on BMHK, so fold G into H exactly as the op's autograd wrapper
    # does and drive the backward with the preserved rng_state. This confirms
    # ck.BwOp can regenerate the 5-D dropout mask (the regression the fix guards)
    # rather than getting rng_state=None and raising.
    qf, kf, vf = q.flatten(2, 3), k.flatten(2, 3), v.flatten(2, 3)
    inp_flat = fmha.Inputs(query=qf, key=kf, value=vf, attn_bias=None, p=p)
    grad_flat = grad_out.flatten(2, 3)

    def _bwd(out_x, lse_x, rng_x):
        ctx = Context(
            lse=lse_x.flatten(1, 2),
            out=out_x.flatten(2, 3),
            op_bw=fmha.ck.BwOp,
            rng_state=rng_x,
        )
        return fmha.ck.BwOp.apply(ctx, inp_flat, grad_flat)

    g_f = _bwd(out_f, ctx_f.lse, ctx_f.rng_state)

    torch.manual_seed(77)
    out_c, ctx_c = fmha.ck.FwOp.apply(inp, needs_gradient=True)
    g_c = _bwd(out_c, ctx_c.lse, ctx_c.rng_state)

    if dtype is torch.float16:
        assert_allclose(out_f.float(), out_c.float(), "fwd", atol=5e-2, rtol=2e-2)
        assert_allclose(g_f.dq.float(), g_c.dq.float(), "dq", atol=5e-2, rtol=2e-2)
        assert_allclose(g_f.dk.float(), g_c.dk.float(), "dk", atol=5e-2, rtol=2e-2)
        assert_allclose(g_f.dv.float(), g_c.dv.float(), "dv", atol=5e-2, rtol=2e-2)
    else:

        def _rel_l2(a, b):
            return (a - b).float().norm() / b.float().norm().clamp_min(1e-6)

        for name, ga, gb, tol in (
            ("fwd", out_f, out_c, 5e-2),
            ("dq", g_f.dq, g_c.dq, 1e-1),
            ("dk", g_f.dk, g_c.dk, 1e-1),
            ("dv", g_f.dv, g_c.dv, 1e-1),
        ):
            rel = _rel_l2(ga, gb)
            assert rel < tol, f"{name}: relative L2 {rel:.4f} exceeds {tol}"


def _is_gfx950():
    return "gfx950" in torch.cuda.get_device_properties(0).gcnArchName


@rocm_only
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("p", [0.3, 0.7])
@pytest.mark.parametrize("k_len", [64, 128])
@pytest.mark.parametrize("seq_len", [32, 200])
def test_dropout_training_flydsl(seq_len, k_len, p, causal, dtype):
    # Training-mode dropout through the public API: dispatch picks flydsl.FwOp,
    # which pins flydsl.BwOp to regenerate the mask from rng_state.
    if not _is_gfx950():
        pytest.skip("flydsl dropout backward is gfx950-only")
    device, B, H = "cuda", 2, 3
    torch.manual_seed(0)
    q, k, v, grad = (
        torch.randn((B, seq_len, H, k_len), device=device, dtype=dtype)
        for _ in range(4)
    )
    q, k, v = (t.requires_grad_(True) for t in (q, k, v))
    bias = fmha.attn_bias.LowerTriangularMask() if causal else None
    seed, offset = _generator_state()
    out = fmha.memory_efficient_attention(q, k, v, bias, p)
    out.backward(grad)

    mask = _drop_mask(B, H, seq_len, seq_len, p, seed, offset)
    qr, kr, vr = (t.detach().float().requires_grad_(True) for t in (q, k, v))
    scores = torch.einsum("bqhd,bkhd->bhqk", qr, kr) * k_len**-0.5
    if causal:
        future = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(future, float("-inf"))
    probs = scores.softmax(-1) * mask / (1 - p)
    ref = torch.einsum("bhqk,bkhd->bqhd", probs, vr)
    ref.backward(grad.float())

    # A wrong or mismatched forward/backward mask gives O(1) relative errors.
    def _rel_l2(a, b):
        return float((a.float() - b).norm() / b.norm().clamp_min(1e-6))

    tol = 2e-2 if dtype is torch.bfloat16 else 5e-3
    for name, got, want in (
        ("out", out, ref),
        ("dq", q.grad, qr.grad),
        ("dk", k.grad, kr.grad),
        ("dv", v.grad, vr.grad),
    ):
        err = _rel_l2(got, want)
        assert err < tol, f"{name}: relative L2 {err:.3e} exceeds {tol}"
