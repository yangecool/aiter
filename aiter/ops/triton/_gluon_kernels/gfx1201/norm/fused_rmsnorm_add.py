# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""RMSNorm + residual add for gfx1201.

Ported from gfx1250 version. One workgroup per tile (BLOCK_SIZE_M rows)
computes RMSNorm(x1 + residual) * weights and writes back.

gfx1250 → gfx1201 substitutions:
  - TDM async_load/async_store → gl.load/buffer_store (skip shared memory)
  - TDM async_wait → removed (sync)
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _rmsnorm_op(row, weights, n_cols, epsilon):
    row_norm = row * row
    row_norm = gl.sum(row_norm, axis=-1, keep_dims=True)
    norm_factor = gl.rsqrt((row_norm / n_cols) + epsilon)
    return row * norm_factor * weights


@gluon.jit
def _gluon_fused_rms_kernel(
    x1_ptr,
    w1_ptr,
    res1_ptr,
    out1_ptr,
    out_res1_ptr,
    eps1,
    M,
    N,
    x1_stride_m,
    res1_stride_m,
    out1_stride_m,
    out_res1_stride_m,
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    FIRST_INPUT_RES: gl.constexpr,
):
    start_pid = gl.program_id(0)
    row_start = start_pid * BLOCK_SIZE_M

    gLayout2D: gl.constexpr = gl.BlockedLayout(
        [1, 8], [1, 32], [1, 4], [1, 0]
    )
    gLayoutN: gl.constexpr = gl.SliceLayout(0, gLayout2D)

    offs_m = gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, gLayout2D))
    offs_n = gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, gLayout2D))
    offs_n1d = gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, gLayoutN))

    # --- Load x1 directly from global (no shared memory / TDM) ---
    x1_offs = (
        (row_start + offs_m[:, None]) * x1_stride_m + offs_n[None, :]
    )
    x1 = gl.load(x1_ptr + x1_offs).to(gl.float32)

    # --- Load + fold residual (optional) ---
    if FIRST_INPUT_RES:
        res1_offs = (
            (row_start + offs_m[:, None]) * res1_stride_m + offs_n[None, :]
        )
        res1_loaded = gl.load(res1_ptr + res1_offs).to(gl.float32)
        x1 = x1 + res1_loaded

        # Store residual output back (needs shared memory for layout conversion
        # if the input and output are the same tensor to avoid race).
        # For simplicity, use buffer_store with same offsets.
        out_res_offs = (
            (row_start + offs_m[:, None]) * out_res1_stride_m + offs_n[None, :]
        )
        gl.amd.rdna4.buffer_store(
            x1.to(out_res1_ptr.dtype.element_ty), out_res1_ptr, out_res_offs
        )

    # --- Load weights ---
    w1 = gl.load(w1_ptr + offs_n1d).to(gl.float32)
    w1 = w1.reshape(1, BLOCK_SIZE_N)
    w1 = gl.convert_layout(w1, gLayout2D)

    # --- RMSNorm ---
    norm1 = _rmsnorm_op(x1, w1, N, eps1)

    # --- Store output ---
    out_offs = (
        (row_start + offs_m[:, None]) * out1_stride_m + offs_n[None, :]
    )
    gl.amd.rdna4.buffer_store(
        norm1.to(out1_ptr.dtype.element_ty), out1_ptr, out_offs
    )
