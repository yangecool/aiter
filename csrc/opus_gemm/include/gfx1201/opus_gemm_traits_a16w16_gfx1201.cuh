// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// opus_gemm_traits_a16w16_gfx1201.cuh — a16w16 (bf16) traits for gfx1201.
//
// Geometry derived from hipBLASLt Tensile tuned solutions for gfx1201:
//   Solution                    MacroTile  Waves  MIWaveTile  DepthU
//   SB_UserArgs (small BF16)    32×64      1      2×4=8       16
//   BBS_BH_Bias (big BF16)     128×128     4      4×4=16      32
//   HHS_BH_Bias (FP16)         128×128     4      4×4=16      32
//   F8F8S_BH (FP8)             128×128     4      4×4=16      64
//
// All share: WMMA 16×16×16, wave32, WorkGroup [32,4,1]=128 threads.
// Key insight: waves split output tile spatially (2×2 grid), each wave
// internally iterates MIWaveTile [4,4] WMMA tiles for its 64×64 sub-block.
#pragma once

#include "../opus_gemm_utils.cuh"

namespace opus_gfx1201_detail {

using opus::operator""_I;

// ── WMMA-128b constants (fixed for gfx1201) ──────────────────────────────
static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;
static constexpr int WMMA_WAVE = 32;  // wave32

// ── Small tile (SB_UserArgs_MT32x64x16, 1 wave) ─────────────────────────
static constexpr int SMALL_B_M = 32;
static constexpr int SMALL_B_N = 64;
static constexpr int SMALL_B_K = 64;
static constexpr int SMALL_T_M = 1;
static constexpr int SMALL_T_N = 1;
static constexpr int SMALL_T_K = 1;
static constexpr int SMALL_BLOCK_SIZE = 32;  // 1 wave
static constexpr int SMALL_E_M = SMALL_B_M / (WMMA_M * SMALL_T_M);  // 2
static constexpr int SMALL_E_N = SMALL_B_N / (WMMA_N * SMALL_T_N);  // 4
static constexpr int SMALL_E_K = SMALL_B_K / (WMMA_K * SMALL_T_K);  // 4

// ── Big tile (BBS/HHS, 4-wave, 128×128×DepthU) ──────────────────────────
static constexpr int BIG_B_M = 128;
static constexpr int BIG_B_N = 128;
static constexpr int BIG_B_K = 64;         // 4 K-steps per iteration
static constexpr int BIG_T_M = 2;
static constexpr int BIG_T_N = 2;
static constexpr int BIG_T_K = 1;
static constexpr int BIG_BLOCK_SIZE = 128; // 4 waves × 32

// Per-wave sub-block: 64×64, MIWaveTile [4,4]
static constexpr int BIG_WAVE_SUB_M = BIG_B_M / BIG_T_M;     // 64
static constexpr int BIG_WAVE_SUB_N = BIG_B_N / BIG_T_N;     // 64
static constexpr int BIG_E_M = BIG_WAVE_SUB_M / WMMA_M;      // 4
static constexpr int BIG_E_N = BIG_WAVE_SUB_N / WMMA_N;      // 4
static constexpr int BIG_E_K = BIG_B_K / (WMMA_K * BIG_T_K); // 4

// LDS budget (bf16, 2 bytes/elem, 2-stage):
//   BIG:  2*(128*64*2 + 128*64*2) = 65536 bytes = 64 KiB << 128 KiB
//   SMALL: 2*(32*64*2 + 64*64*2)  = 24576 bytes = 24 KiB << 128 KiB
static constexpr int BIG_LDS_2STG = 2 * (BIG_B_M * BIG_B_K * 2 + BIG_B_N * BIG_B_K * 2);
static_assert(BIG_LDS_2STG <= 131072, "gfx1201 big tile LDS must fit 128 KiB");
static constexpr int SMALL_LDS_2STG = 2 * (SMALL_B_M * SMALL_B_K * 2 + SMALL_B_N * SMALL_B_K * 2);
static_assert(SMALL_LDS_2STG <= 131072, "gfx1201 small tile LDS must fit 128 KiB");

// ── Traits struct ────────────────────────────────────────────────────────
template<int BLOCK_SIZE_,
         typename BLOCK_,
         typename DTYPE_,
         typename VEC_,
         typename TILE_,
         typename WAVE_,
         bool HAS_BIAS_ = false,
         typename D_BIAS_ = void,
         bool HAS_OOB_ = true>
struct opus_gemm_a16w16_traits_gfx1201 {
    using BLOCK = opus::remove_cvref_t<BLOCK_>;
    using DTYPE = opus::remove_cvref_t<DTYPE_>;
    using VEC   = opus::remove_cvref_t<VEC_>;
    using TILE  = opus::remove_cvref_t<TILE_>;
    using WAVE  = opus::remove_cvref_t<WAVE_>;

    static constexpr int BLOCK_SIZE = BLOCK_SIZE_;

    static constexpr int B_M = opus::get<0>(BLOCK{});
    static constexpr int B_N = opus::get<1>(BLOCK{});
    static constexpr int B_K = opus::get<2>(BLOCK{});

    using D_A   = typename DTYPE::a;
    using D_B   = typename DTYPE::b;
    using D_C   = typename DTYPE::c;
    using D_ACC = typename DTYPE::acc;

    static constexpr int T_M = opus::get<0>(TILE{});
    static constexpr int T_N = opus::get<1>(TILE{});
    static constexpr int T_K = opus::get<2>(TILE{});

    static_assert(BLOCK_SIZE / WMMA_WAVE == T_M * T_N * T_K,
                  "BLOCK_SIZE/wave must equal T_M*T_N*T_K");

    static constexpr int W_M = opus::get<0>(WAVE{});
    static constexpr int W_N = opus::get<1>(WAVE{});
    static constexpr int W_K = opus::get<2>(WAVE{});

    static_assert(W_M == WMMA_M && W_N == WMMA_N && W_K == WMMA_K,
                  "gfx1201 WMMA must be 16x16x16");

    // Spatial wave partition: each wave covers B_M/T_M × B_N/T_N sub-block.
    static constexpr int WAVE_SUB_M = B_M / T_M;
    static constexpr int WAVE_SUB_N = B_N / T_N;

    // MIWaveTile: WMMA tiles per wave (per sub-block).
    static constexpr int E_M = WAVE_SUB_M / W_M;
    static constexpr int E_N = WAVE_SUB_N / W_N;
    static constexpr int E_K = B_K / (W_K * T_K);

    static constexpr int VEC_A = opus::get<0>(VEC{});
    static constexpr int VEC_B = opus::get<1>(VEC{});

    static constexpr bool HAS_BIAS = HAS_BIAS_;
    using D_BIAS = D_BIAS_;
    static constexpr bool HAS_OOB = HAS_OOB_;

    static_assert(WMMA_WAVE == 32, "gfx1201 requires wave32");
};

}  // namespace opus_gfx1201_detail
