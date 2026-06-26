// test_pipeline_gfx1201.cu — verify minimal opus_gemm bf16 GEMM on gfx1201
// Follows test_wmma_gfx1201.cu's host/device dual-pass pattern.
#ifdef __HIP_DEVICE_COMPILE__
// ── Device pass ─────────────────────────────────────────────────────────────
#include "opus/opus.hpp"
#include "gfx1201/opus_gemm_pipeline_a16w16_gfx1201.cuh"

#if defined(__gfx1201__) || defined(__gfx1200__)

// Instantiate the 16x16 template so the host launcher can name it.
template __global__ void opus_gfx1201_pipeline::gemm_a16w16_mono_tile_gfx1201_kernel<16, 16>(
    const opus::bf16_t*, const opus::bf16_t*, opus::bf16_t*, int);

#endif

#else
// ── Host pass: extern "C" launcher + test driver ────────────────────────────
#include "opus/opus.hpp"
#include <cstdio>
#include <hip/hip_runtime.h>

#define HIP_CHECK(x) do { auto e = (x); if (e != hipSuccess) { fprintf(stderr, "HIP err %d at %d\n", (int)e, __LINE__); return 1; } } while(0)

template<typename T>
__global__ void gemm_a16w16_mono_tile_gfx1201_kernel(
    const T*, const T*, T*, int) {}

extern "C" void run_gemm_pipe_gfx1201(
    const void* dA, const void* dB, void* dC, int K)
{
    hipLaunchKernelGGL(
        (gemm_a16w16_mono_tile_gfx1201_kernel<opus::bf16_t>),
        dim3(1), dim3(32), 0, 0,
        static_cast<const opus::bf16_t*>(dA),
        static_cast<const opus::bf16_t*>(dB),
        static_cast<opus::bf16_t*>(dC), K);
    hipGetLastError();
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
    HIP_CHECK(hipMalloc(&dA, M * K * sizeof(opus::bf16_t)));
    HIP_CHECK(hipMalloc(&dB, N * K * sizeof(opus::bf16_t)));
    HIP_CHECK(hipMalloc(&dC, M * N * sizeof(opus::bf16_t)));
    HIP_CHECK(hipMemcpy(dA, hA, M * K * sizeof(opus::bf16_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB, N * K * sizeof(opus::bf16_t), hipMemcpyHostToDevice));
    run_gemm_pipe_gfx1201(dA, dB, dC, K);
    opus::bf16_t hC[M * N];
    HIP_CHECK(hipMemcpy(hC, dC, M * N * sizeof(opus::bf16_t), hipMemcpyDeviceToHost));
    float max_diff = 0;
    for (int i = 0; i < M * N; ++i) {
        float d = float(hC[i]) - hC_ref[i];
        if (d < 0) d = -d;
        if (d > max_diff) max_diff = d;
    }
    printf("gfx1201 opus_gemm bf16 GEMM (16x16x%d): max_diff = %f %s\n",
           K, max_diff, max_diff < 1.0f ? "PASS" : "FAIL");
    HIP_CHECK(hipFree(dA)); HIP_CHECK(hipFree(dB)); HIP_CHECK(hipFree(dC));
    return max_diff < 1.0f ? 0 : 1;
}
#endif
