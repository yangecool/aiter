// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// opus_gemm_arch_gfx1201.cuh — gfx1201 dispatch + launcher.
//
// Provides:
//   opus_dispatch_a16w16_gfx1201<T>(M,N,K,batch,has_bias) → kernel pointer
//   opus_a16w16_tune_dispatch_gfx1201<T>(id) → tune dispatch (stub)
//
// Tensile-aligned MIWaveTile [4,4] pipeline: 128x128 tile, 4-wave, 128t.
#pragma once

#include "../opus_gemm_arch.cuh"
#include "../opus_gemm_common.cuh"
#include "aiter_tensor.h"
#include "opus_gemm_traits_a16w16_gfx1201.cuh"
#include "opus_gemm_pipeline_a16w16_gfx1201.cuh"

#include <cstddef>
#include <cstdint>

namespace opus_gfx1201_detail {

using OpusA16W16NoscaleKernel = void (*)(
    aiter_tensor_t &, aiter_tensor_t &,
    aiter_tensor_t &, std::optional<aiter_tensor_t>, int);

// ── Minimal kargs (packed for kernel launch) ─────────────────────────────
struct opus_gemm_noscale_kargs_gfx1201 {
    const void* ptr_a;
    const void* ptr_b;
    void*       ptr_c;
    int m;
    int n;
    int k;
    int stride_a;   // in elements
    int stride_b;
    int stride_c;
};

// ── Grid-walk + pipeline kernel ───────────────────────────────────────────
template <int BLOCK_M, int BLOCK_N>
__global__ void gemm_a16w16_gfx1201_kernel(opus_gemm_noscale_kargs_gfx1201 kargs)
{
    // Single-block launch for now: one workgroup handles one BLOCK_M x BLOCK_N tile.
    // Future: grid launch with column-major walk.
    if (blockIdx.x == 0 && blockIdx.y == 0) {
        using D_A = opus::bf16_t;
        gemm_a16w16_big_tile_gfx1201_impl<BLOCK_M, BLOCK_N>(
            static_cast<const D_A*>(kargs.ptr_a),
            static_cast<const D_A*>(kargs.ptr_b),
            static_cast<D_A*>(kargs.ptr_c),
            kargs.k);
    }
}

// ── Host-side launcher ────────────────────────────────────────────────────
template <int BLOCK_M, int BLOCK_N>
void launch_gemm_a16w16_gfx1201(
    aiter_tensor_t &a, aiter_tensor_t &b,
    aiter_tensor_t &c, std::optional<aiter_tensor_t> bias,
    int /*split_k*/)
{
    (void)bias;
    opus_gemm_noscale_kargs_gfx1201 kargs;
    kargs.ptr_a   = a.data;
    kargs.ptr_b   = b.data;
    kargs.ptr_c   = c.data;
    kargs.m       = static_cast<int>(a.size[0]);
    kargs.n       = static_cast<int>(b.size[0]);
    kargs.k       = static_cast<int>(a.size[1]);
    kargs.stride_a = static_cast<int>(a.stride[0] ? a.stride[0] : a.size[1]);
    kargs.stride_b = static_cast<int>(b.stride[0] ? b.stride[0] : b.size[1]);
    kargs.stride_c = static_cast<int>(c.stride[0] ? c.stride[0] : b.size[0]);

    // Determine grid: ceil_div(M/BLOCK_M) × ceil_div(N/BLOCK_N)
    int grid_m = (kargs.m + BLOCK_M - 1) / BLOCK_M;
    int grid_n = (kargs.n + BLOCK_N - 1) / BLOCK_N;

    gemm_a16w16_gfx1201_kernel<BLOCK_M, BLOCK_N>
        <<<dim3(grid_m, grid_n, 1), dim3(128, 1, 1)>>>(kargs);
}

// ── Kernel lookup (host-side function that returns launcher pointer) ──────
template <typename CDataType>
OpusA16W16NoscaleKernel
opus_dispatch_a16w16_gfx1201(int M, int N, int K, int batch, bool has_bias)
{
    (void)M; (void)N; (void)K; (void)batch; (void)has_bias;
    // For now: always use 128x128 big tile.  Heuristic dispatch (pick tile
    // size based on M/N) can be added later.
    return &launch_gemm_a16w16_gfx1201<128, 128>;
}

// ── Tune dispatch (stub) ──────────────────────────────────────────────────
struct OpusA16W16TuneKernelGfx1201 { int kid; };

template <typename CDataType>
OpusA16W16TuneKernelGfx1201
opus_a16w16_tune_dispatch_gfx1201(int id)
{
    (void)id;
    const auto &info = opus_get_arch_info();
    AITER_CHECK(false,
                "opus_gemm: a16w16 tune dispatch for gfx1201 is not yet "
                "implemented.");
}

}  // namespace opus_gfx1201_detail
