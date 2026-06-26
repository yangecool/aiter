// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// opus_gemm_pipeline_a16w16_gfx1201.cuh — bf16 a16w16 GEMM pipeline
// for gfx1201 (RDNA4 / WMMA-128b / wave32).
//
// Tensile-aligned design (BBS_BH_Bias_MT128x128x32 / SB_MT32x64x16):
//   - Waves split the output tile spatially (T_M×T_N wave grid).
//   - Each wave internally iterates E_M×E_N WMMA tiles (MIWaveTile).
//   - Single WMMA tile = 16×16×16 (WMMA-128b).
//   - K-loop inside the wave, steppping by WMMA_K=16 each iteration.
//
// BIG mode:  128×128 output, 4 waves (2×2), 16 WMMA/wave, DepthU~32
// SMALL mode: 32×64 output,  1 wave,        8 WMMA/wave, DepthU~16
#pragma once

#include <opus/opus.hpp>


namespace opus_gfx1201_pipeline {

using opus::operator""_I;

// ── Per-wave WMMA tile iterator (Tensile MIWaveTile equivalent) ──────────
// For a wave covering SUB_M×SUB_N, iterate E_M×E_N WMMA tiles of 16×16.
// tile_m in [0, E_M), tile_n in [0, E_N).
// Returns the m/n offset of the current tile within the wave's sub-block.
template <int E_M, int E_N>
struct wmma_tile_iterator {
    static constexpr int TOTAL = E_M * E_N;
    int idx;
    __device__ wmma_tile_iterator(int start) : idx(start) {}
    __device__ int tile_m() const { return idx / E_N; }
    __device__ int tile_n() const { return idx % E_N; }
    __device__ bool valid() const { return idx < TOTAL; }
    __device__ void next() { ++idx; }
    // Offset of this WMMA tile within the wave's sub-block.
    __device__ int m_offset() const { return tile_m() * 16; }
    __device__ int n_offset() const { return tile_n() * 16; }
};

// ── Big-tile pipeline (128×128, 4-wave) ──────────────────────────────────
// Each wave covers 64×64.  WMMA tiles: 4×4=16 per wave.
// BLOCK_M=128, BLOCK_N=128, called with K full (DepthU pacing handled
// by the outer host loop or split-K).
template <int BLOCK_M, int BLOCK_N>
__device__ void gemm_a16w16_big_tile_gfx1201_impl(
    const opus::bf16_t* __restrict__ A,   // [BLOCK_M, K]
    const opus::bf16_t* __restrict__ B,   // [BLOCK_N, K]  (N×K, transposed)
    opus::bf16_t* __restrict__ C,         // [BLOCK_M, BLOCK_N]
    int K)
{
    constexpr int W_M = 16, W_N = 16, W_K = 16;
    constexpr int T_M = 2, T_N = 2;       // 4 waves, 2×2 grid
    constexpr int SUB_M = BLOCK_M / T_M;   // 64
    constexpr int SUB_N = BLOCK_N / T_N;   // 64
    constexpr int E_M = SUB_M / W_M;       // 4
    constexpr int E_N = SUB_N / W_N;       // 4
    constexpr int ELEM_A = W_M * W_K / 32; // 8
    constexpr int ELEM_B = W_N * W_K / 32; // 8
    constexpr int ELEM_C = W_M * W_N / 32; // 8

    using vtype_a = opus::vector_t<opus::bf16_t, ELEM_A>;
    using vtype_b = opus::vector_t<opus::bf16_t, ELEM_B>;
    using vtype_c = opus::vector_t<opus::fp32_t, ELEM_C>;

    int tid  = static_cast<int>(__builtin_amdgcn_workitem_id_x());
    int lane = tid % 32;
    int wave = tid / 32;
    int tile_m_wave = wave / T_N;   // wave row in 2×2 grid
    int tile_n_wave = wave % T_N;   // wave col
    int m_base = tile_m_wave * SUB_M;
    int n_base = tile_n_wave * SUB_N;

    // gfx1201 native WMMA fragment encoding (row-distributed A, col-distributed B/C)
    // A fragment: lane i in [0,31], j in [0,7]:
    //   regA[i][j] = A[m_offset + (i%16), k + (i/16)*8 + j]
    // B/C fragment: lane i, j:
    //   regB[i][j] = B[n_offset + (i%16), k + (i/16)*8 + j]
    //   regC[i][j] = C[m_offset + (i/16)*8 + j, n_offset + (i%16)]

    int a_row    = m_base + (lane % 16);
    int a_k_base = (lane / 16) * 8;
    int b_col    = n_base + (lane % 16);
    int b_k_base = (lane / 16) * 8;

    opus::wmma<opus::bf16_t, opus::bf16_t, opus::fp32_t, W_M, W_N, W_K> mma;

    // Initialize fp32 accumulators for all E_M×E_N tiles in this wave.
    constexpr int NTILES = E_M * E_N;
    vtype_c acc[NTILES];
    #pragma unroll
    for (int t = 0; t < NTILES; ++t)
        acc[t] = vtype_c{};

    // K-loop: iterate over full K by WMMA_K steps.
    for (int k = 0; k < K; k += W_K)
    {
        // Load A fragment once per K-step (A is shared across N tiles).
        vtype_a va;
        #pragma unroll
        for (int j = 0; j < ELEM_A; ++j)
            va[j] = A[a_row * K + (k + a_k_base + j)];

        // For each WMMA tile in this wave, load B fragment and issue WMMA.
        wmma_tile_iterator<E_M, E_N> it(0);
        #pragma unroll
        for (; it.valid(); it.next())
        {
            int m_off = m_base + it.m_offset();
            int n_off = n_base + it.n_offset();

            vtype_b vb;
            #pragma unroll
            for (int j = 0; j < ELEM_B; ++j)
                vb[j] = B[(n_off + (lane % 16)) * K + (k + b_k_base + j)];

            acc[it.idx] = mma(va, vb, acc[it.idx]);
        }
    }

    // Store fp32 accumulator → bf16 C (column-distributed).
    wmma_tile_iterator<E_M, E_N> it2(0);
    int c_col_base = n_base + (lane % 16);
    int c_m_base  = m_base + (lane / 16) * 8;
    #pragma unroll
    for (; it2.valid(); it2.next())
    {
        int m_off = it2.m_offset();
        int n_off = it2.n_offset();
        #pragma unroll
        for (int j = 0; j < ELEM_C; ++j)
            C[(c_m_base + m_off + j) * BLOCK_N + (c_col_base + n_off)] =
                static_cast<opus::bf16_t>(acc[it2.idx][j]);
    }
}

// ── Small-tile pipeline (32×64, 1-wave, Tensile SB_UserArgs) ────────────
template <int BLOCK_M, int BLOCK_N>
__device__ void gemm_a16w16_small_tile_gfx1201_impl(
    const opus::bf16_t* __restrict__ A,
    const opus::bf16_t* __restrict__ B,
    opus::bf16_t* __restrict__ C,
    int K)
{
    constexpr int W_M = 16, W_N = 16, W_K = 16;
    constexpr int E_M = BLOCK_M / W_M;     // 2
    constexpr int E_N = BLOCK_N / W_N;     // 4
    constexpr int ELEM_A = W_M * W_K / 32; // 8
    constexpr int ELEM_B = W_N * W_K / 32; // 8
    constexpr int ELEM_C = W_M * W_N / 32; // 8

    using vtype_a = opus::vector_t<opus::bf16_t, ELEM_A>;
    using vtype_b = opus::vector_t<opus::bf16_t, ELEM_B>;
    using vtype_c = opus::vector_t<opus::fp32_t, ELEM_C>;

    int lane = static_cast<int>(__builtin_amdgcn_workitem_id_x() % 32);

    int a_row    = lane % 16;
    int a_k_base = (lane / 16) * 8;
    int b_col    = lane % 16;
    int b_k_base = (lane / 16) * 8;

    opus::wmma<opus::bf16_t, opus::bf16_t, opus::fp32_t, W_M, W_N, W_K> mma;

    constexpr int NTILES = E_M * E_N;
    vtype_c acc[NTILES];
    #pragma unroll
    for (int t = 0; t < NTILES; ++t)
        acc[t] = vtype_c{};

    for (int k = 0; k < K; k += W_K)
    {
        for (int em = 0; em < E_M; ++em)
        {
            vtype_a va;
            #pragma unroll
            for (int j = 0; j < ELEM_A; ++j)
                va[j] = A[(em * W_M + (lane % 16)) * K + (k + a_k_base + j)];

            for (int en = 0; en < E_N; ++en)
            {
                vtype_b vb;
                #pragma unroll
                for (int j = 0; j < ELEM_B; ++j)
                    vb[j] = B[(en * W_N + (lane % 16)) * K + (k + b_k_base + j)];

                acc[em * E_N + en] = mma(va, vb, acc[em * E_N + en]);
            }
        }
    }

    // Store
    int c_col_base = lane % 16;
    int c_m_base  = (lane / 16) * 8;
    for (int em = 0; em < E_M; ++em)
    {
        for (int en = 0; en < E_N; ++en)
        {
            #pragma unroll
            for (int j = 0; j < ELEM_C; ++j)
                C[(c_m_base + em * W_M + j) * BLOCK_N + (c_col_base + en * W_N)] =
                    static_cast<opus::bf16_t>(acc[em * E_N + en][j]);
        }
    }
}

}  // namespace opus_gfx1201_pipeline

