// test_pipeline_gfx1201.cu — verify minimal opus_gemm bf16 GEMM on gfx1201
// Dual-pass pattern matching test_wmma_gfx1201.cu (proven to run).

#ifdef __HIP_DEVICE_COMPILE__
// ── Device pass ─────────────────────────────────────────────────────────────
#include "opus/opus.hpp"
#if defined(__gfx1201__) || defined(__gfx1200__)

#include "gfx1201/opus_gemm_pipeline_a16w16_gfx1201.cuh"

// Re-export the 16x16 instantiation under the plain symbol the host stub names.
// (host sees the same wmma_gfx12_pipe_kernel<bf16_t> declaration below)
__global__ void gemm_pipe_kernel_bf16_16x16(
    const opus::bf16_t* __restrict__ A,
    const opus::bf16_t* __restrict__ B,
    opus::bf16_t* __restrict__ C,
    int K)
{
    opus_gfx1201_pipeline::gemm_a16w16_mono_tile_gfx1201_impl<16, 16>(A, B, C, K);
}

#endif

#else
// ── Host pass: empty kernel stub + extern "C" launcher + driver ─────────────
#include "opus/opus.hpp"
#include "opus/hip_minimal.hpp"
#include <cstdio>

// Host-side stub: same signature, empty body. The device pass defines the real one.
__global__ void gemm_pipe_kernel_bf16_16x16(
    const opus::bf16_t* A, const opus::bf16_t* B, opus::bf16_t* C, int K) {}

extern "C" void run_gemm_pipe_bf16_16x16(
    const void* dA, const void* dB, void* dC, int K)
{
    hipLaunchKernelGGL((gemm_pipe_kernel_bf16_16x16),
                       dim3(1), 32, 0, 0,
                       static_cast<const opus::bf16_t*>(dA),
                       static_cast<const opus::bf16_t*>(dB),
                       static_cast<opus::bf16_t*>(dC), K);
    hipDeviceSynchronize();
}

int main() {
    constexpr int M = 16, N = 16, K = 64;
    opus::bf16_t hA[M * K], hB[N * K];
    for (int i = 0; i < M * K; ++i) hA[i] = opus::bf16_t(float(i % 7) * 0.1f);
    for (int i = 0; i < N * K; ++i) hB[i] = opus::bf16_t(float(i % 5) * 0.1f);
    float hC_ref[M * N];
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float s = 0;
            for (int k = 0; k < K; ++k) s += float(hA[m * K + k]) * float(hB[n * K + k]);
            hC_ref[m * N + n] = s;
        }
    opus::bf16_t *dA, *dB, *dC;
    hipMalloc(&dA, M * K * sizeof(opus::bf16_t));
    hipMalloc(&dB, N * K * sizeof(opus::bf16_t));
    hipMalloc(&dC, M * N * sizeof(opus::bf16_t));
    hipMemcpy(dA, hA, M * K * sizeof(opus::bf16_t), hipMemcpyHostToDevice);
    hipMemcpy(dB, hB, N * K * sizeof(opus::bf16_t), hipMemcpyHostToDevice);
    run_gemm_pipe_bf16_16x16(dA, dB, dC, K);
    opus::bf16_t hC[M * N];
    hipMemcpy(hC, dC, M * N * sizeof(opus::bf16_t), hipMemcpyDeviceToHost);
    float max_diff = 0;
    for (int i = 0; i < M * N; ++i) {
        float d = float(hC[i]) - hC_ref[i];
        if (d < 0) d = -d;
        if (d > max_diff) max_diff = d;
    }
    printf("gfx1201 opus_gemm bf16 GEMM (16x16x%d): max_diff = %f %s\n",
           K, max_diff, max_diff < 1.0f ? "PASS" : "FAIL");
    hipFree(dA); hipFree(dB); hipFree(dC);
    return max_diff < 1.0f ? 0 : 1;
}
#endif
