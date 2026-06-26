// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// opus_gemm_pipeline_a16w16_gfx1201.cuh — bf16 a16w16 GEMM pipeline
// for gfx1201 (RDNA4 / WMMA-128b / wave32).
//
// Tensile-aligned design (BBS_BH_Bias_MT128x128x32 / SB_MT32x64x16):
//   - Waves split output tile spatially (T_M×T_N wave grid).
//   - Each wave internally iterates E_M×E_N WMMA tiles (MIWaveTile [4,4]).
//   - WMMA tile = 16×16×16 (WMMA-128b, gfx1201 fixed).
//   - K-loop inside wave body, stepping by WMMA_K=16 per iteration.
//
// BIG mode:  128×128 output, 4 waves (2×2), 16 WMMA tiles/wave
// SMALL mode: 32×64 output,  1 wave,        8 WMMA tiles/wave
#pragma once

#include <opus/opus.hpp>

namespace opus_gfx1201_pipeline {

using opus::operator""_I;

// ── WMMA tile iterator ───────────────────────────────────────────────────
template <int E_M, int E_N>
struct wmma_tile_iter {
    static constexpr int TOTAL = E_M * E_N;
    int i;
    __device__ wmma_tile_iter(int start) : i(start) {}
    __device__ int tile_m() const { return i / E_N; }
    __device__ int tile_n() const { return i % E_N; }
    __device__ operator bool() const { return i < TOTAL; }
    __device__ void operator++() { ++i; }
    __device__ int m_off() const { return tile_m() * 16; }
    __device__ int n_off() const { return tile_n() * 16; }
};

// ── BIG tile: 128×128, 4-wave (2×2), 16 WMMA tiles/wave ──────────────────
template <int BLOCK_M, int BLOCK_N>
__device__ void gemm_a16w16_big_tile_gfx1201_impl(
    const opus::bf16_t* __restrict__ A,   // [BLOCK_M, K] row-major
    const opus::bf16_t* __restrict__ B,   // [BLOCK_N, K] N×K (transposed)
    opus::bf16_t* __restrict__ C,         // [BLOCK_M, BLOCK_N]
    int K)
{
    constexpr int WM = 16, WN = 16, WK = 16;
    constexpr int TM = 2, TN = 2;          // 4 waves, 2×2 grid
    constexpr int SUB_M = BLOCK_M / TM;     // 64
    constexpr int SUB_N = BLOCK_N / TN;     // 64
    constexpr int EM = SUB_M / WM;          // 4
    constexpr int EN = SUB_N / WN;          // 4
    constexpr int EA = WM*WK / 32;          // 8
    constexpr int EB = WN*WK / 32;          // 8
    constexpr int EC = WM*WN / 32;          // 8

    using mma_t = opus::wmma<opus::bf16_t, opus::bf16_t, opus::fp32_t, WM, WN, WK>;
    using va_t  = typename mma_t::vtype_a;
    using vb_t  = typename mma_t::vtype_b;
    using vc_t  = typename mma_t::vtype_c;

    int tid  = static_cast<int>(__builtin_amdgcn_workitem_id_x());
    int lane = tid % 32;
    int wave = tid / 32;
    int wm   = wave / TN;                  // wave row in 2×2 grid
    int wn   = wave % TN;                  // wave col
    int m_base = wm * SUB_M;
    int n_base = wn * SUB_N;

    // gfx1201 native fragment encoding: A row-distributed, B/C col-distributed
    int a_row   = m_base + (lane % 16);
    int a_kbase = (lane / 16) * 8;
    int b_col   = n_base + (lane % 16);
    int b_kbase = (lane / 16) * 8;

    mma_t mma;

    // Accumulators for all E_M×E_N tiles (16 fp32×8 per wave)
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
        // Load A fragment (same for all N-tiles in this K-step)
        va_t va;
        #pragma unroll
        for (int j = 0; j < EA; ++j)
            va[j] = A[a_row * K + (k + a_kbase + j)];

        // For each WMMA tile in this wave
        for (wmma_tile_iter<EM, EN> it(0); it; ++it)
        {
            int m_off = m_base + it.m_off();
            int n_off = n_base + it.n_off();

            vb_t vb;
            #pragma unroll
            for (int j = 0; j < EB; ++j)
                vb[j] = B[(n_off + (lane % 16)) * K + (k + b_kbase + j)];

            acc[it.i] = mma(va, vb, acc[it.i]);
        }
    }

    // Store fp32 → bf16 C (col-distributed fragment)
    int c_col = n_base + (lane % 16);
    int c_mbase = m_base + (lane / 16) * 8;

    for (wmma_tile_iter<EM, EN> it(0); it; ++it)
    {
        int m_off = it.m_off();
        int n_off = it.n_off();
        #pragma unroll
        for (int j = 0; j < EC; ++j)
            C[(c_mbase + m_off + j) * BLOCK_N + (c_col + n_off)] =
                static_cast<opus::bf16_t>(acc[it.i][j]);
    }
}

// ── SMALL tile: 32×64, 1-wave, MIWaveTile [2,4] ──────────────────────────
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
    constexpr int EA = WM*WK / 32;         // 8
    constexpr int EB = WN*WK / 32;         // 8
    constexpr int EC = WM*WN / 32;         // 8

    using mma_t = opus::wmma<opus::bf16_t, opus::bf16_t, opus::fp32_t, WM, WN, WK>;
    using va_t  = typename mma_t::vtype_a;
    using vb_t  = typename mma_t::vtype_b;
    using vc_t  = typename mma_t::vtype_c;

    int lane = static_cast<int>(__builtin_amdgcn_workitem_id_x() % 32);

    int a_row   = lane % 16;
    int a_kbase = (lane / 16) * 8;
    int b_col   = lane % 16;
    int b_kbase = (lane / 16) * 8;

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
        for (int em = 0; em < EM; ++em)
        {
            va_t va;
            #pragma unroll
            for (int j = 0; j < EA; ++j)
                va[j] = A[(em * WM + (lane % 16)) * K + (k + a_kbase + j)];

            for (int en = 0; en < EN; ++en)
            {
                vb_t vb;
                #pragma unroll
                for (int j = 0; j < EB; ++j)
                    vb[j] = B[(en * WN + (lane % 16)) * K + (k + b_kbase + j)];

                acc[em * EN + en] = mma(va, vb, acc[em * EN + en]);
            }
        }
    }

    int c_col = lane % 16;
    int c_mbase = (lane / 16) * 8;
    for (int em = 0; em < EM; ++em)
        for (int en = 0; en < EN; ++en)
            #pragma unroll
            for (int j = 0; j < EC; ++j)
                C[(c_mbase + em * WM + j) * BLOCK_N + (c_col + en * WN)] =
                    static_cast<opus::bf16_t>(acc[em * EN + en][j]);
}

}  // namespace opus_gfx1201_pipeline
