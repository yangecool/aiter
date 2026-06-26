// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// opus_gemm_arch_gfx1201.cuh — gfx1201 (RDNA4 / Navi 48) dispatch skeleton.
//
// gfx1201 uses WMMA-128b (wave32, 16x16x16) — a different instruction
// family from gfx950 (MFMA 16x16x32) and gfx1250 (WMMA-256b). The opus
// framework already has the gfx12 w32 WMMA builtins wired (opus.hpp
// L2448-2455, verified PASS bit-exact on RX 9070 XT), and this header
// gives opus_gemm a per-arch entry point for gfx1201 so the top-level
// router (opus_gemm.cu) recognises the device instead of hitting the
// generic "only gfx950" fallback.
//
// The traits geometry is fixed in opus_gemm_traits_a16w16_gfx1201.cuh
// (WMMA 16x16x16, MacroTile 64x16, derived from hipBLASLt gfx1201 tuned
// solutions MatrixInstruction [16,16,16,1]). The pipeline kernel body —
// the WMMA-128b emission loop with gfx1201 register layout — is NOT yet
// ported; it needs a layout distinct from gfx950's MFMA body. Until it
// lands, dispatch raises a clear "pipeline not yet implemented" error
// rather than a misleading "unsupported arch".
//
// To complete: implement opus_gemm_pipeline_a16w16_*_gfx1201.cuh kernels
// (mirror gfx950 pipeline structure, swap MFMA -> WMMA-128b via
// DISPATCH_WMMA_GFX12_F32_ macros), then replace the stub bodies below
// with real heuristic dispatch.
#pragma once

#include "../opus_gemm_arch.cuh"
#include "../opus_gemm_common.cuh"
#include "aiter_tensor.h"  // aiter_tensor_t (used in OpusA16W16NoscaleKernel signature)
#include "opus_gemm_traits_a16w16_gfx1201.cuh"

#include <cstddef>

namespace opus_gfx1201_detail {

// Function-pointer type — identical shape to gfx950's so the top-level
// router can store the result uniformly. Until a real kernel is linked,
// the pointers stay nullptr and dispatch raises.
using OpusA16W16NoscaleKernel = void (*)(
    aiter_tensor_t &, aiter_tensor_t &,
    aiter_tensor_t &, std::optional<aiter_tensor_t>, int);

}  // namespace opus_gfx1201_detail

// ── Dispatch entry points (stub: pipeline kernels not yet ported) ─────────
// These match the gfx950 signatures so opus_gemm.cu's `case OpusGfxArch::Gfx1201`
// branch compiles and links. They raise a precise, actionable error so the
// failure mode is distinguishable from "arch unknown".

template <typename CDataType>
opus_gfx1201_detail::OpusA16W16NoscaleKernel
opus_dispatch_a16w16_gfx1201(int M, int N, int K, int batch, bool has_bias = false)
{
    (void)M; (void)N; (void)K; (void)batch; (void)has_bias;
    const auto &info = opus_get_arch_info();
    AITER_CHECK(false,
                "opus_gemm: a16w16 dispatch for gfx1201 (WMMA-128b) is not yet "
                "implemented. The opus WMMA builtins are verified working "
                "(test_wmma_gfx1201 PASS), and the traits geometry is fixed "
                "(WMMA 16x16x16, MacroTile 64x16, hipBLASLt-derived). "
                "Pipeline kernel bodies remain to be ported. "
                "Current device ", info.dev, " gcnArchName='", info.name, "'.");
}

// Tune dispatch (id-based) — also a stub.
struct OpusA16W16TuneKernelGfx1201 {
    int kid;
};

template <typename CDataType>
OpusA16W16TuneKernelGfx1201
opus_a16w16_tune_dispatch_gfx1201(int id)
{
    (void)id;
    const auto &info = opus_get_arch_info();
    AITER_CHECK(false,
                "opus_gemm: a16w16 tune dispatch for gfx1201 is not yet "
                "implemented (pipeline kernels pending). "
                "Current device ", info.dev, " gcnArchName='", info.name, "'.");
}
