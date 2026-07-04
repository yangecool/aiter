# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon grouped row-reduce for gfx1201 MoE scatter-combine.

Ported from gfx1250 version. One workgroup per group sums K*B rows
in-register (no cross-wave communication) into out[g, :N], with optional
external residual fold-in.

gfx1250 → gfx1201 substitutions:
  - TDM async bulk-load → gl.load directly to registers (no shared memory)
  - gl.amd.gfx1250.buffer_load → gl.amd.rdna4.buffer_load
  - gl.amd.gfx1250.buffer_store → gl.amd.rdna4.buffer_store
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def reduce_grouped_gluon(
    X,              # [B, M, N] (flattened to [B*M, N])
    Out,            # [num_groups, N]
    InIndx,         # [num_groups, K] int
    Residual,       # [num_groups, N] external residual (dummy ptr if unused)
    stride_xm,
    stride_om,
    stride_on,
    stride_res_m,
    stride_res_n,
    M,
    N: gl.constexpr,
    NPAD: gl.constexpr,  # next_pow2(N)
    B: gl.constexpr,
    K: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    HAS_EXT_RESIDUAL: gl.constexpr,
):
    group = gl.program_id(0)
    gl.static_assert(NPAD >= 32, "NPAD must be >= 32")
    gl.static_assert(
        NPAD % (NUM_WARPS * 32) == 0, "NPAD must be a multiple of NUM_WARPS*32"
    )

    SIZE_N: gl.constexpr = NPAD // (NUM_WARPS * 32)
    BLKN: gl.constexpr = gl.BlockedLayout(
        [1, SIZE_N], [1, 32], [1, NUM_WARPS], [1, 0]
    )

    offs_n = gl.arange(0, NPAD, layout=gl.SliceLayout(0, BLKN))
    mask_n = offs_n[None, :] < N

    # Load rows directly from global memory and accumulate (no TDM).
    acc = gl.zeros([1, NPAD], dtype=gl.float32, layout=BLKN)
    for i in gl.static_range(K):
        idx_i = gl.load(InIndx + group * K + i)
        for b in gl.static_range(B):
            row = b * M + idx_i
            row_data = gl.load(
                X + row * stride_xm + offs_n[None, :],
                mask=mask_n,
                other=0.0,
            )
            acc += row_data.to(gl.float32)

    o_offs = group * stride_om + offs_n[None, :] * stride_on

    # Fold in the external residual before writeback.
    if HAS_EXT_RESIDUAL:
        r_offs = group * stride_res_m + offs_n[None, :] * stride_res_n
        res = gl.amd.rdna4.buffer_load(Residual, r_offs, mask=mask_n, other=0.0)
        acc += res.to(gl.float32)

    gl.amd.rdna4.buffer_store(acc.to(Out.dtype.element_ty), Out, o_offs, mask=mask_n)


def reduce_grouped_gluon_num_warps(npad: int) -> int:
    """Pick the largest wave count W in {8,4,2,1} with npad % (W*32) == 0."""
    for w in (8, 4, 2, 1):
        if npad % (w * 32) == 0:
            return w
    return 1
