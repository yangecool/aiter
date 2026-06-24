// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "../opus_moe_common.cuh"
#include "opus_moe_stage2_utils_gfx950.cuh"

#include "aiter_hip_common.h"

#include <cstdint>
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

constexpr int kOpusMoeStage2RouteOutputReduceAutoBlockN = -1;
constexpr int kOpusMoeStage2RouteOutputReduceBf16BlockN = 2048;
constexpr int kOpusMoeStage2RouteOutputReduceDefaultBlockN = 4096;
constexpr int kOpusMoeStage2RouteOutputReduceDefaultThreads = 256;
constexpr int kOpusMoeStage2RouteOutputReduceDsv4BlockN =
    opus_moe::kStage2A8W4DecodeModelDim;
constexpr int kOpusMoeStage2RouteOutputReduceDsv4Threads = 448;

inline int opus_moe_stage2_reduce_token_slot_route_output_select_block_n(
    int model_dim,
    int requested_block_n)
{
    if(requested_block_n > 0)
        return requested_block_n;
    return model_dim == opus_moe::kStage2A8W4DecodeModelDim
               ? opus_moe::kStage2A8W4DecodeModelDim
               : kOpusMoeStage2RouteOutputReduceDefaultBlockN;
}

// TOPK > 0  -> slot loop bound is a compile-time constant so the whole loop
//              unrolls; all TOPK * (elems/4) global loads are issued up front
//              before any s_waitcnt, hiding memory latency (route_reduce is
//              ~86% stalled on load waitcnt when topk is a runtime value).
// TOPK == 0 -> fallback to the runtime kargs.topk loop.
template<int BLOCK_N, int BLOCK_THREADS, int TOPK = 0>
__global__ __launch_bounds__(BLOCK_THREADS, 4) void
opus_moe_stage2_reduce_token_slot_route_output_kernel_gfx950(opus_moe_stage2_kargs kargs)
{
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx950__)
    static_assert(BLOCK_N % BLOCK_THREADS == 0);
    constexpr int elems_per_thread = BLOCK_N / BLOCK_THREADS;
    const int token = static_cast<int>(blockIdx.x);
    const int col_base =
        static_cast<int>(blockIdx.y) * BLOCK_N + static_cast<int>(threadIdx.x) * elems_per_thread;
    // Compile-time bound when TOPK>0 (launch guarantees TOPK == kargs.topk).
    const int topk_loop = (TOPK > 0) ? TOPK : kargs.topk;

    if constexpr(elems_per_thread % 4 == 0)
    {
        if(col_base + elems_per_thread - 1 < kargs.model_dim)
        {
            float acc[elems_per_thread];
#pragma unroll
            for(int j = 0; j < elems_per_thread; ++j)
            {
                acc[j] = 0.0f;
            }

#pragma unroll
            for(int slot = 0; slot < topk_loop; ++slot)
            {
                const int route_row = token * kargs.topk + slot;
#pragma unroll
                for(int group = 0; group < elems_per_thread / 4; ++group)
                {
                    const int col = col_base + group * 4;
                    const uint64_t packed =
                        *reinterpret_cast<const uint64_t*>(
                            kargs.route_out_bf16 +
                            static_cast<int64_t>(route_row) * kargs.stride_route_o_t + col);
                    hip_bfloat16 v0;
                    hip_bfloat16 v1;
                    hip_bfloat16 v2;
                    hip_bfloat16 v3;
                    v0.data = static_cast<uint16_t>(packed);
                    v1.data = static_cast<uint16_t>(packed >> 16);
                    v2.data = static_cast<uint16_t>(packed >> 32);
                    v3.data = static_cast<uint16_t>(packed >> 48);
                    acc[group * 4 + 0] += static_cast<float>(v0);
                    acc[group * 4 + 1] += static_cast<float>(v1);
                    acc[group * 4 + 2] += static_cast<float>(v2);
                    acc[group * 4 + 3] += static_cast<float>(v3);
                }
            }

#pragma unroll
            for(int group = 0; group < elems_per_thread / 4; ++group)
            {
                const int col = col_base + group * 4;
                const uint32_t packed01 =
                    opus_moe_gfx950_cvt_pk_bf16_f32(acc[group * 4 + 0],
                                                     acc[group * 4 + 1]);
                const uint32_t packed23 =
                    opus_moe_gfx950_cvt_pk_bf16_f32(acc[group * 4 + 2],
                                                     acc[group * 4 + 3]);
                const uint64_t packed_out =
                    static_cast<uint64_t>(packed01) |
                    (static_cast<uint64_t>(packed23) << 32);
                *reinterpret_cast<uint64_t*>(kargs.out_bf16 +
                                             static_cast<int64_t>(token) * kargs.stride_o_t +
                                             col) = packed_out;
            }
            return;
        }
    }

    float acc[elems_per_thread];
#pragma unroll
    for(int j = 0; j < elems_per_thread; ++j)
    {
        acc[j] = 0.0f;
    }

#pragma unroll
    for(int slot = 0; slot < topk_loop; ++slot)
    {
        const int route_row = token * kargs.topk + slot;
#pragma unroll
        for(int j = 0; j < elems_per_thread; ++j)
        {
            const int col = col_base + j;
            if(col < kargs.model_dim)
            {
                const hip_bfloat16 value =
                    kargs.route_out_bf16[static_cast<int64_t>(route_row) *
                                             kargs.stride_route_o_t +
                                         col];
                acc[j] += static_cast<float>(value);
            }
        }
    }

#pragma unroll
    for(int j = 0; j < elems_per_thread; ++j)
    {
        const int col = col_base + j;
        if(col < kargs.model_dim)
        {
            kargs.out_bf16[static_cast<int64_t>(token) * kargs.stride_o_t + col] =
                hip_bfloat16(acc[j]);
        }
    }
#endif
#endif
}

template<int BLOCK_N, int BLOCK_THREADS, int TOPK>
inline void opus_moe_stage2_reduce_token_slot_route_output_launch_variant_gfx950(
    const opus_moe_stage2_kargs& kargs,
    dim3 grid,
    hipStream_t stream)
{
    opus_moe_stage2_reduce_token_slot_route_output_kernel_gfx950<
        BLOCK_N,
        BLOCK_THREADS,
        TOPK><<<grid, dim3(BLOCK_THREADS), 0, stream>>>(kargs);
}

// Dispatch on block_n with the topk known at compile time (TOPK).
template<int TOPK>
inline void opus_moe_stage2_reduce_token_slot_route_output_dispatch_block_n_gfx950(
    const opus_moe_stage2_kargs& kargs,
    dim3 grid,
    hipStream_t stream,
    int block_n)
{
    switch(block_n)
    {
    case kOpusMoeStage2RouteOutputReduceBf16BlockN:
        opus_moe_stage2_reduce_token_slot_route_output_launch_variant_gfx950<
            kOpusMoeStage2RouteOutputReduceBf16BlockN,
            kOpusMoeStage2RouteOutputReduceDefaultThreads,
            TOPK>(kargs, grid, stream);
        break;
    case kOpusMoeStage2RouteOutputReduceDefaultBlockN:
        opus_moe_stage2_reduce_token_slot_route_output_launch_variant_gfx950<
            kOpusMoeStage2RouteOutputReduceDefaultBlockN,
            kOpusMoeStage2RouteOutputReduceDefaultThreads,
            TOPK>(kargs, grid, stream);
        break;
    case kOpusMoeStage2RouteOutputReduceDsv4BlockN:
        opus_moe_stage2_reduce_token_slot_route_output_launch_variant_gfx950<
            kOpusMoeStage2RouteOutputReduceDsv4BlockN,
            kOpusMoeStage2RouteOutputReduceDsv4Threads,
            TOPK>(kargs, grid, stream);
        break;
    default:
        AITER_CHECK(false,
                    "unsupported Opus MoE route-output reduce block_n=",
                    block_n);
    }
}

inline void opus_moe_stage2_reduce_token_slot_route_output_launch_gfx950(
    const opus_moe_stage2_kargs& kargs,
    hipStream_t stream,
    int requested_block_n)
{
    const int block_n = opus_moe_stage2_reduce_token_slot_route_output_select_block_n(
        kargs.model_dim, requested_block_n);
    dim3 grid(kargs.token_num, (kargs.model_dim + block_n - 1) / block_n, 1);
    // Specialize for common topk values (unrolls the slot loop -> all loads
    // issued up front, hiding latency); otherwise fall back to the runtime-topk
    // kernel (TOPK=0), which is correct for any topk.
    switch(kargs.topk)
    {
    case 4:
        opus_moe_stage2_reduce_token_slot_route_output_dispatch_block_n_gfx950<4>(
            kargs, grid, stream, block_n);
        break;
    case 6:
        opus_moe_stage2_reduce_token_slot_route_output_dispatch_block_n_gfx950<6>(
            kargs, grid, stream, block_n);
        break;
    case 8:
        opus_moe_stage2_reduce_token_slot_route_output_dispatch_block_n_gfx950<8>(
            kargs, grid, stream, block_n);
        break;
    default:
        opus_moe_stage2_reduce_token_slot_route_output_dispatch_block_n_gfx950<0>(
            kargs, grid, stream, block_n);
        break;
    }
}
