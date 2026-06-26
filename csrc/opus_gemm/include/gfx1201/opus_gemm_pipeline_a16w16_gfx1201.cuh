// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// opus_gemm_pipeline_a16w16_gfx1201.cuh — bf16 a16w16 GEMM pipeline
// for gfx1201 (RDNA4 / WMMA-128b / wave32).
//
// Tensile-aligned design (BBS_BH_Bias_MT128x128x32 / SB_MT32x64x16):
//   - Waves split output tile spatially (T_M x T_N wave grid).
//   - Each wave internally iterates E_M x E_N WMMA tiles (MIWaveTile [4,4]).
//   - WMMA tile = 16x16x16.  K-loop inside wave body, step WMMA_K=16.
//   - For each K-step: iterate M-tiles (load A), for each N-tile (load B, WMMA).
//
// BIG:  128x128 output, 4 waves (2x2), 16 WMMA tiles/wave
// SMALL: 32x64 output,  1 wave,        8 WMMA tiles/wave
#pragma once

#include <opus/opus.hpp>

namespace opus_gfx1201_pipeline {

using opus::operator""_I;

// ── BIG tile: 128x128, 4-wave (2x2), per-wave MIWaveTile [4,4] ──────────
template <int BLOCK_M, int BLOCK_N>
__device__ void gemm_a16w16_big_tile_gfx1201_impl(
    const opus::bf16_t* __restrict__ A,
    const opus::bf16_t* __restrict__ B,
    opus::bf16_t* __restrict__ C,
    int K)
{
    constexpr int WM = 16, WN = 16, WK = 16;
    constexpr int TM = 2, TN = 2;          // 4-wave grid
    constexpr int SM = BLOCK_M / TM;        // 64
    constexpr int SN = BLOCK_N / TN;        // 64
    constexpr int EM = SM / WM;             // 4
    constexpr int EN = SN / WN;             // 4
    constexpr int EA = WM*WK/32;            // 8
    constexpr int EB = WN*WK/32;            // 8
    constexpr int EC = WM*WN/32;            // 8

    using mma_t = opus::wmma<opus::bf16_t, opus::bf16_t, opus::fp32_t, WM, WN, WK>;
    using va_t  = typename mma_t::vtype_a;
    using vb_t  = typename mma_t::vtype_b;
    using vc_t  = typename mma_t::vtype_c;

    int tid  = static_cast<int>(__builtin_amdgcn_workitem_id_x());
    int lane = tid % 32;
    int wave = tid / 32;
    int wm   = wave / TN;
    int wn   = wave % TN;
    int m0   = wm * SM;                    // wave M base
    int n0   = wn * SN;                    // wave N base

    // gfx1201 native WMMA fragment: A row-distributed, B/C col-distributed.
    //   lane i: A[m + i%16, k + (i/16)*8 + j]
    //   lane i: B[n + i%16, k + (i/16)*8 + j]
    int lane_m = lane % 16;                // row/col index
    int lane_k = (lane / 16) * 8;          // K sub-offset

    mma_t mma;

    // Accumulators: EM x EN = 16 tiles, each a vc_t (fp32[8])
    constexpr int NT = EM * EN;
    vc_t acc[NT];
    #pragma unroll
    for (int t = 0; t < NT; ++t) {
        #pragma unroll
        for (int j = 0; j < EC; ++j) acc[t][j] = 0.0f;
    }

    // K-loop
    for (int k = 0; k < K; k += WK)
    {
        // Iterate M-tiles (A varies per M-tile)
        #pragma unroll
        for (int em = 0; em < EM; ++em)
        {
            int m_off = m0 + em * WM;       // M offset of this tile row
            // Load A fragment for this M-tile row
            va_t va;
            #pragma unroll
            for (int j = 0; j < EA; ++j)
                va[j] = A[(m_off + lane_m) * K + (k + lane_k + j)];

            // Iterate N-tiles (B varies per N-tile)
            #pragma unroll
            for (int en = 0; en < EN; ++en)
            {
                int n_off = n0 + en * WN;    // N offset of this tile col
                vb_t vb;
                #pragma unroll
                for (int j = 0; j < EB; ++j)
                    vb[j] = B[(n_off + lane_m) * K + (k + lane_k + j)];

                acc[em * EN + en] = mma(va, vb, acc[em * EN + en]);
            }
        }
    }

    // Store fp32 accumulators -> bf16 C (col-distributed fragment)
    #pragma unroll
    for (int em = 0; em < EM; ++em)
    {
        int m_off = m0 + em * WM;
        #pragma unroll
        for (int en = 0; en < EN; ++en)
        {
            int n_off = n0 + en * WN;
            int tidx = em * EN + en;
            #pragma unroll
            for (int j = 0; j < EC; ++j)
                C[(m_off + lane_k + j) * BLOCK_N + (n_off + lane_m)] =
                    static_cast<opus::bf16_t>(acc[tidx][j]);
        }
    }
}

// ── SMALL tile: 32x64, 1-wave, MIWaveTile [2,4] ─────────────────────────
template <int BLOCK_M, int BLOCK_N>
__device__ void gemm_a16w16_small_tile_gfx1201_impl(
    const opus::bf16_t* __restrict__ A,
    const opus::bf16_t* __restrict__ B,
    opus::bf16_t* __restrict__ C,
    int K)
{
    constexpr int WM = 16, WN = 16, WK = 16;
    constexpr int EM = BLOCK_M / WM;       // 2
    constexpr int EN = BLOCK_N / WN;       // 4
    constexpr int EA = WM*WK/32;
    constexpr int EB = WN*WK/32;
    constexpr int EC = WM*WN/32;

    using mma_t = opus::wmma<opus::bf16_t, opus::bf16_t, opus::fp32_t, WM, WN, WK>;
    using va_t  = typename mma_t::vtype_a;
    using vb_t  = typename mma_t::vtype_b;
    using vc_t  = typename mma_t::vtype_c;

    int lane   = static_cast<int>(__builtin_amdgcn_workitem_id_x() % 32);
    int lane_m = lane % 16;
    int lane_k = (lane / 16) * 8;

    mma_t mma;

    constexpr int NT = EM * EN;
    vc_t acc[NT];
    #pragma unroll
    for (int t = 0; t < NT; ++t) {
        #pragma unroll
        for (int j = 0; j < EC; ++j) acc[t][j] = 0.0f;
    }

    for (int k = 0; k < K; k += WK)
    {
        #pragma unroll
        for (int em = 0; em < EM; ++em)
        {
            va_t va;
            #pragma unroll
            for (int j = 0; j < EA; ++j)
                va[j] = A[(em * WM + lane_m) * K + (k + lane_k + j)];

            #pragma unroll
            for (int en = 0; en < EN; ++en)
            {
                vb_t vb;
                #pragma unroll
                for (int j = 0; j < EB; ++j)
                    vb[j] = B[(en * WN + lane_m) * K + (k + lane_k + j)];

                acc[em * EN + en] = mma(va, vb, acc[em * EN + en]);
            }
        }
    }

    #pragma unroll
    for (int em = 0; em < EM; ++em)
        #pragma unroll
        for (int en = 0; en < EN; ++en)
            #pragma unroll
            for (int j = 0; j < EC; ++j)
                C[(em * WM + lane_k + j) * BLOCK_N + (en * WN + lane_m)] =
                    static_cast<opus::bf16_t>(acc[em * EN + en][j]);
}

}  // namespace opus_gfx1201_pipeline
