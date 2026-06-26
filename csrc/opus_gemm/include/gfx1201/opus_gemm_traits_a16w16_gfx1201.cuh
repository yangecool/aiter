// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// opus_gemm_traits_a16w16_gfx1201.cuh — a16w16 (bf16) traits for gfx1201.
//
// gfx1201 (RDNA4 / Navi 48) uses WMMA-128b (wave32) instructions:
//   wmma_f32_16x16x16_bf16_w32_gfx12  (MatrixInstruction [16, 16, 16, 1])
// This differs from gfx950 (MFMA, 16x16x32) and gfx1250 (WMMA-256b,
// 16x16x{64,128}). The pipeline geometry here is derived from the tuned
// hipBLASLt gfx1201 Tensile solutions
// (MatrixInstruction [16,16,16,1], MacroTile {64,16}, MIWaveGroup [4,1],
// DepthU 32, WorkGroup [64,2,1]) — i.e. the same WMMA width AMD's own
// library tunes for this silicon.
//
// Status: traits skeleton. Pipeline kernel bodies (the WMMA emission loop
// with gfx1201 register layout) are NOT yet ported — they require a
// WMMA-128b register layout distinct from gfx950's MFMA body. This header
// fixes the geometry so a future opus_gemm_pipeline_*_gfx1201.cuh can be
// filled in against stable constants.
#pragma once

#include "../opus_gemm_utils.cuh"

namespace opus_gfx1201_detail {

using opus::operator""_I;

// ── WMMA-128b geometry constants (gfx1201, wave32) ──────────────────────
// Single WMMA instruction: 16x16x16 (M x N x K). K-dim width = 16, i.e.
// one WMMA consumes 16 elements along the contraction axis. Compare:
//   gfx950  MFMA       16x16x32  (W_K=32)
//   gfx1250 WMMA-256b  16x16x64/128 (W_K=64/128)
//   gfx1201 WMMA-128b  16x16x16  (W_K=16)  <-- this file
static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;
static constexpr int WMMA_WAVE = 32;   // wave32 on gfx1201

// MacroTile from hipBLASLt tuned solution (F8B8HS): MacroTile0=64,
// MacroTile1=16, MIWaveGroup=[4,1]. For bf16 a16w16 we keep the same
// wave-per-M count (4 waves cover M=64 with W_M=16) and N=16.
// BLOCK = 64 x 16 x K, TILE = (4,1,1), WAVE = (16,16,16).
static constexpr int B_M_DEFAULT = 64;
static constexpr int B_N_DEFAULT = 16;
static constexpr int B_K_DEFAULT = 64;   // 4x WMMA_K steps per K iteration
static constexpr int T_M_DEFAULT = 4;
static constexpr int T_N_DEFAULT = 1;
static constexpr int T_K_DEFAULT = 1;
static constexpr int BLOCK_SIZE_DEFAULT = T_M_DEFAULT * T_N_DEFAULT * T_K_DEFAULT * WMMA_WAVE;
// = 4*1*1*32 = 128 threads

static_assert(BLOCK_SIZE_DEFAULT == 128, "gfx1201 mono-tile BLOCK_SIZE must be 128");
static_assert(B_M_DEFAULT % (WMMA_M * T_M_DEFAULT) == 0, "B_M must be WMMA_M * T_M * n");
static_assert(B_N_DEFAULT % (WMMA_N * T_N_DEFAULT) == 0, "B_N must be WMMA_N * T_N");
static_assert(B_K_DEFAULT % (WMMA_K * T_K_DEFAULT) == 0, "B_K must be WMMA_K * T_K");

// K-dim WMMA steps per B_K iteration (hipBLASLt DepthU=32 => 2 steps for
// bf16 at W_K=16; we use B_K=64 => 4 steps for fuller pipelining within
// LDS budget — 2*64*(64+16)*2 bytes = 20480 bytes << 128 KiB).
static constexpr int E_K_DEFAULT = B_K_DEFAULT / (WMMA_K * T_K_DEFAULT);

// LDS budget check (bf16, 2 bytes/element, async double-buffer):
//   LDS = num_stages * (B_M*BK*2 + B_N*BK*2) bytes
//   B_M=64 B_N=16 B_K=64 stages=2 => 2*(64*64*2 + 16*64*2) = 2*(8192+2048) = 20480
//   << 128 KiB (131072). Comfortable; room for 3+ stages if beneficial.
static constexpr int LDS_BYTES_MONO_TILE_2STG =
    2 * (B_M_DEFAULT * B_K_DEFAULT * 2 + B_N_DEFAULT * B_K_DEFAULT * 2);
static_assert(LDS_BYTES_MONO_TILE_2STG <= 131072,
              "gfx1201 mono-tile a16w16 LDS must fit 128 KiB");

// ── Traits struct (mirrors gfx950 shape; usable by future pipeline) ──────
// Parameterised so the eventual pipeline kernel can specialise on dtype /
// vec width. WAVE is locked to (16,16,16) — the only WMMA width gfx1201
// has for bf16.
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
                  "BLOCK_SIZE / wave == T_M * T_N * T_K must hold");

    static constexpr int W_M = opus::get<0>(WAVE{});
    static constexpr int W_N = opus::get<1>(WAVE{});
    static constexpr int W_K = opus::get<2>(WAVE{});

    // gfx1201 WMMA-128b: the ONLY legal instruction width for bf16/fp8 is
    // 16x16x16. Reject anything wider — gfx1250's 16x16x{64,128} builtins
    // do not exist on gfx1201.
    static_assert(W_M == WMMA_M && W_N == WMMA_N && W_K == WMMA_K,
                  "gfx1201 only has WMMA 16x16x16 (WMMA-128b, wave32); "
                  "16x16x{32,64,128} are gfx1250-only.");

    static constexpr int E_M = B_M / (W_M * T_M);
    static constexpr int E_N = B_N / (W_N * T_N);
    static constexpr int E_K = B_K / (W_K * T_K);

    static constexpr int VEC_A = opus::get<0>(VEC{});
    static constexpr int VEC_B = opus::get<1>(VEC{});

    static constexpr bool HAS_BIAS = HAS_BIAS_;
    using D_BIAS = D_BIAS_;
    static constexpr bool HAS_OOB = HAS_OOB_;

    // Sanity: wave must be 32 on gfx1201 (WMMA-128b is _w32_gfx12).
    static_assert(WMMA_WAVE == 32, "gfx1201 WMMA-128b requires wave32");
};

}  // namespace opus_gfx1201_detail
