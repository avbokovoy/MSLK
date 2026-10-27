# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# pyre-unsafe

"""CK-compatible Philox dropout keep-mask for the FlyDSL attention kernels.

The mask is a pure function of (seed, offset, batch, head, q_row, kv_col), the
same one CK's fmha kernels use (ck_tile philox_rand.hpp and
host/reference/reference_batched_dropout_randval.hpp):

* P is split into 32x32 tiles. Tile (tm, tn) of head h in batch b draws 64
  Philox4x32 (7 rounds) results with counter = (offset + (b * H + h) * 64 + lane,
  subsequence = tm | tn << 32) and key = seed.
* Byte r of lane `lane` is the random value at row 16 * (r // 8) + 8 * (lane // 32)
  + r % 8 and column lane % 32 of the tile.
* An element is kept when its byte <= floor((1 - p) * 255); kept values are
  scaled by 1 / (1 - p).

Because the mask depends only on (seed, offset), a backward regenerates exactly
the forward's mask, and CK's backward accepts the same rng_state.
"""

import functools
from typing import Tuple

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import range_constexpr
from mslk.flydsl import buffer_ops

PHILOX_W0 = 0x9E3779B9
PHILOX_W1 = 0xBB67AE85
PHILOX_M0 = 0xD2511F53
PHILOX_M1 = 0xCD9E8D57
PHILOX_ROUNDS = 7
# Philox results per 32x32 tile of P (64 lanes x 16 bytes).
PHILOX_PER_TILE = 64
_BLOCK = 256


def philox_keep_threshold(p: float) -> int:
    """Largest random byte that keeps an element (CK's p_undrop_in_uint8_t)."""
    return int((1.0 - p) * 255.0 // 1)


def _mulhilo(a, b):
    prod = a.to(fx.Uint64) * b.to(fx.Uint64)
    return (prod >> 32).to(fx.Uint32), prod.to(fx.Uint32)


def philox_4x32(c0, c1, c2, c3, k0, k1):
    """ck_tile::philox::get_philox_4x32 on Uint32 values."""
    for i in range(PHILOX_ROUNDS):
        hi0, lo0 = _mulhilo(fx.Uint32(PHILOX_M0), c0)
        hi1, lo1 = _mulhilo(fx.Uint32(PHILOX_M1), c2)
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
        if i < PHILOX_ROUNDS - 1:
            k0 = k0 + fx.Uint32(PHILOX_W0)
            k1 = k1 + fx.Uint32(PHILOX_W1)
    return c0, c1, c2, c3


@functools.lru_cache(maxsize=None)
def _build_keep_mask_launcher():
    @flyc.kernel(known_block_size=[_BLOCK, 1, 1])
    def philox_keep_mask_kernel(
        Mask: fx.Tensor,  # [B * H * M * N] uint8 keep-mask (0/1)
        n_items: fx.Int32,  # B * H * tiles_m * tiles_n * 64
        seqlen_q: fx.Int32,
        seqlen_k: fx.Int32,
        mask_bytes: fx.Int32,
        seed_lo: fx.Int32,
        seed_hi: fx.Int32,
        offset_lo: fx.Int32,
        offset_hi: fx.Int32,
        threshold: fx.Int32,
    ):
        idx = fx.Index(fx.block_idx.x) * _BLOCK + fx.Index(fx.thread_idx.x)
        if idx < fx.Index(n_items):
            tiles_m = (fx.Index(seqlen_q) + 31) // 32
            tiles_n = (fx.Index(seqlen_k) + 31) // 32
            lane = idx % PHILOX_PER_TILE
            t = idx // PHILOX_PER_TILE
            tn = t % tiles_n
            t = t // tiles_n
            tm = t % tiles_m
            bh = t // tiles_m
            offset = (fx.Uint32(offset_hi).to(fx.Uint64) << 32) | fx.Uint32(
                offset_lo
            ).to(fx.Uint64)
            ctr = offset + fx.Uint64(fx.Int64(bh * PHILOX_PER_TILE + lane))
            words = philox_4x32(
                ctr.to(fx.Uint32),
                (ctr >> 32).to(fx.Uint32),
                fx.Uint32(fx.Int32(tm)),
                fx.Uint32(fx.Int32(tn)),
                fx.Uint32(seed_lo),
                fx.Uint32(seed_hi),
            )
            # Exact-size resource: masked-off (out-of-tile) stores are dropped.
            rsrc = buffer_ops.create_buffer_resource(
                Mask, max_size=False, num_records_bytes=fx.Int32(mask_bytes)
            )
            col = tn * 32 + lane % 32
            row_base = tm * 32 + (lane // 32) * 8
            thr = fx.Uint32(threshold)
            for r in range_constexpr(16):
                rand = (words[r // 4] >> ((r % 4) * 8)) & fx.Uint32(0xFF)
                keep = (rand <= thr).select(fx.Int32(1), fx.Int32(0))
                row = row_base + (16 * (r // 8) + r % 8)
                valid = (row < fx.Index(seqlen_q)) & (col < fx.Index(seqlen_k))
                off = (bh * fx.Index(seqlen_q) + row) * fx.Index(seqlen_k) + col
                buffer_ops.buffer_store(
                    keep.to(fx.Int8), rsrc, fx.Int32(off), mask=valid
                )

    @flyc.jit
    def launch(
        Mask: fx.Tensor,
        n_items: fx.Int32,
        seqlen_q: fx.Int32,
        seqlen_k: fx.Int32,
        mask_bytes: fx.Int32,
        seed_lo: fx.Int32,
        seed_hi: fx.Int32,
        offset_lo: fx.Int32,
        offset_hi: fx.Int32,
        threshold: fx.Int32,
        stream: fx.Stream,
    ):
        n_blocks = (fx.Index(n_items) + _BLOCK - 1) // _BLOCK
        philox_keep_mask_kernel(
            Mask,
            n_items,
            seqlen_q,
            seqlen_k,
            mask_bytes,
            seed_lo,
            seed_hi,
            offset_lo,
            offset_hi,
            threshold,
        ).launch(grid=(fx.Int32(n_blocks), 1, 1), block=(_BLOCK, 1, 1), stream=stream)

    return launch


def _split_u64(v: int) -> Tuple[int, int]:
    # Kernel args are i32; pass the two's-complement halves.
    lo, hi = v & 0xFFFFFFFF, (v >> 32) & 0xFFFFFFFF
    return tuple(x - (1 << 32) if x >= (1 << 31) else x for x in (lo, hi))


def philox_dropout_keep_mask(
    batch: int,
    heads: int,
    seqlen_q: int,
    seqlen_k: int,
    p: float,
    seed: int,
    offset: int,
    device: torch.device,
) -> torch.Tensor:
    """uint8 [batch, heads, seqlen_q, seqlen_k] keep-mask (1 = keep)."""
    mask = torch.empty(
        (batch, heads, seqlen_q, seqlen_k), dtype=torch.uint8, device=device
    )
    if mask.numel() == 0:
        return mask
    if mask.numel() >= 1 << 31:
        raise ValueError(
            "philox_dropout_keep_mask: the keep-mask must hold < 2**31 elements, "
            f"got {mask.numel()}"
        )
    tiles = ((seqlen_q + 31) // 32) * ((seqlen_k + 31) // 32)
    seed_lo, seed_hi = _split_u64(seed)
    offset_lo, offset_hi = _split_u64(offset)
    with torch.cuda.device(device):
        _build_keep_mask_launcher()(
            mask.view(-1),
            batch * heads * tiles * PHILOX_PER_TILE,
            seqlen_q,
            seqlen_k,
            mask.numel(),
            seed_lo,
            seed_hi,
            offset_lo,
            offset_hi,
            philox_keep_threshold(p),
            torch.cuda.current_stream(device),
        )
    return mask


def reserve_philox_seed_offset(
    device: torch.device, batch: int, heads: int, seqlen_q: int, seqlen_k: int
) -> Tuple[int, int]:
    """Draw (seed, offset) from the device's default generator like CK does.

    Mirrors the CK forward's gen->philox_cuda_state((B + 3) * (H + 1) * (M + 1) *
    (N + 1)) so FlyDSL and CK dropout calls never reuse each other's counters.
    """
    index = device.index if device.index is not None else torch.cuda.current_device()
    gen = torch.cuda.default_generators[index]
    increment = (batch + 3) * (heads + 1) * (seqlen_q + 1) * (seqlen_k + 1)
    increment = (increment + 3) // 4 * 4  # CUDAGeneratorImpl rounds to 4
    seed, offset = gen.initial_seed(), gen.get_offset()
    gen.set_offset(offset + increment)
    return seed, offset
