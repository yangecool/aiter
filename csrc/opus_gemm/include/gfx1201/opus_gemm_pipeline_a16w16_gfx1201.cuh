// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// opus_gemm_pipeline_a16w16_gfx1201.cuh — minimal bf16 a16w16 GEMM pipeline
// for gfx1201 (RDNA4 / WMMA-128b / wave32).
//
// Phase 2b "path B": does NOT use make_tiled_mma / wmma_adaptor (whose lane
// encoding matches gfx1250, not gfx12). Instead it calls opus::wmma<>::operator()
// directly with the gfx1201-native fragment encoding (A row-distributed,
// B/C column-distributed) proven correct by test_wmma_gfx1201.cu.
//
// This is a single-wave, single-tile reference pipeline: one workgroup of
// 32 threads computes one BLOCK_M x BLOCK_N output tile by iterating over K
// in BLOCK_K steps, issuing 16x16x16 WMMA per sub-tile. It validates that
// gfx1201 can run an opus_gemm bf16 GEMM end-to-end before investing in the
// tiled/multi-wave adaptor + codegen layer (Phase 2c).
//
// Geometry: BLOCK_M=16, BLOCK_N=16, BLOCK_K=K (full K in one pass, K-loop
// inside the kernel). One WMMA (16x16x16) per K=16 step.
#pragma once

#include <opus/opus.hpp>

#if defined(__gfx1201__) || defined(__gfx1200__)

namespace opus_gfx1201_pipeline {

using opus::operator""_I;

// Multi-tile bf16 a16w16 GEMM kernel body (device function). One workgroup
// of N_WAVES waves computes BLOCK_M x BLOCK_N output. Each wave handles one
// 16x16 sub-tile; waves tile both M and N (row-major over the tile grid).
// BLOCK_M=16*TILE_M, BLOCK_N=16*TILE_N, TILE_M*TILE_N=N_WAVES.
template <int BLOCK_M, int BLOCK_N>
__device__ void gemm_a16w16_mono_tile_gfx1201_impl(
    const opus::bf16_t* __restrict__ A,   // [BLOCK_M, K]
    const opus::bf16_t* __restrict__ B,   // [BLOCK_N, K]  (B is N x K, transposed)
    opus::bf16_t* __restrict__ C,         // [BLOCK_M, BLOCK_N]
    int K)
{
    constexpr int WM = 16, WN = 16, WK = 16;
    constexpr int TILE_M = BLOCK_M / 16;
    constexpr int TILE_N = BLOCK_N / 16;
    static_assert(BLOCK_M % 16 == 0 && BLOCK_N % 16 == 0, "BLOCK must be 16*n");
    constexpr int ELEM_A = WM * WK / 32;  // 8
    constexpr int ELEM_B = WN * WK / 32;  // 8
    constexpr int ELEM_C = WM * WN / 32;  // 8

    using vtype_a = opus::vector_t<opus::bf16_t, ELEM_A>;
    using vtype_b = opus::vector_t<opus::bf16_t, ELEM_B>;
    using vtype_c = opus::vector_t<opus::fp32_t, ELEM_C>;  // fp32 accumulator

    int lane = static_cast<int>(__builtin_amdgcn_workitem_id_x() % 32);
    int wave_id = static_cast<int>(__builtin_amdgcn_workitem_id_x() / 32);
    int tile_m = wave_id / TILE_N;  // M tile index
    int tile_n = wave_id % TILE_N;  // N tile index
    int m_offset = tile_m * 16;
    int n_offset = tile_n * 16;
    // gfx1201 native wmma_128b fragment encoding (from test_wmma_gfx1201.cu):
    int a_row    = m_offset + (lane % 16);
    int a_k_base = (lane / 16) * 8;
    int b_col    = n_offset + (lane % 16);
    int b_k_base = (lane / 16) * 8;
    int c_col    = n_offset + (lane % 16);
    int c_m_base = m_offset + (lane / 16) * 8;

    vtype_c acc{};
    opus::wmma<opus::bf16_t, opus::bf16_t, opus::fp32_t, WM, WN, WK> mma;

    // K-loop: step by WK=16
    for (int k = 0; k < K; k += WK)
    {
        vtype_a va{};
        vtype_b vb{};
        #pragma unroll
        for (int j = 0; j < ELEM_A; ++j)
            va[j] = A[a_row * K + (k + a_k_base + j)];
        #pragma unroll
        for (int j = 0; j < ELEM_B; ++j)
            vb[j] = B[b_col * K + (k + b_k_base + j)];
        // WMMA: C += A * B (fp32 accumulator)
        acc = mma(va, vb, acc);
    }

    // Store C fragment (fp32 -> bf16). C is column-distributed.
    #pragma unroll
    for (int j = 0; j < ELEM_C; ++j)
        C[(c_m_base + j) * BLOCK_N + c_col] =
            static_cast<opus::bf16_t>(acc[j]);
}

}  // namespace opus_gfx1201_pipeline

#endif  // __gfx1201__ / __gfx1200__
