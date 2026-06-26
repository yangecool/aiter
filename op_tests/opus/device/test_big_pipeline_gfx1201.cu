#include <hip/hip_runtime.h>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include "gfx1201/opus_gemm_pipeline_a16w16_gfx1201.cuh"

__global__ void pipeline_big_kernel(
    const opus::bf16_t* A, const opus::bf16_t* B,
    opus::bf16_t* C, int M, int N, int K)
{
    if (blockIdx.x == 0 && blockIdx.y == 0)
        opus_gfx1201_pipeline::gemm_a16w16_big_tile_gfx1201_impl<128, 128>(A, B, C, K);
}

void cpu_ref(const opus::bf16_t* A, const opus::bf16_t* B, float* C, int M, int N, int K) {
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float sum = 0;
            for (int k = 0; k < K; ++k)
                sum += (float)A[m*K + k] * (float)B[n*K + k];
            C[m*N + n] = sum;
        }
}

int main() {
    constexpr int M = 128, N = 128, K = 4096;
    hipSetDevice(0);

    opus::bf16_t *hA = new opus::bf16_t[M*K];
    opus::bf16_t *hB = new opus::bf16_t[N*K];
    float *hRef = new float[M*N];

    for (int i = 0; i < M*K; ++i) hA[i] = opus::bf16_t((float)(rand()%1000 - 500) / 1000.0f);
    for (int i = 0; i < N*K; ++i) hB[i] = opus::bf16_t((float)(rand()%1000 - 500) / 1000.0f);

    opus::bf16_t *dA, *dB, *dC;
    hipMalloc(&dA, M*K*sizeof(opus::bf16_t));
    hipMalloc(&dB, N*K*sizeof(opus::bf16_t));
    hipMalloc(&dC, M*N*sizeof(opus::bf16_t));
    hipMemcpy(dA, hA, M*K*sizeof(opus::bf16_t), hipMemcpyHostToDevice);
    hipMemcpy(dB, hB, N*K*sizeof(opus::bf16_t), hipMemcpyHostToDevice);

    pipeline_big_kernel<<<dim3(1,1,1), dim3(128,1,1)>>>(dA, dB, dC, M, N, K);
    hipDeviceSynchronize();

    hipEvent_t start, stop;
    hipEventCreate(&start); hipEventCreate(&stop);
    hipEventRecord(start);
    pipeline_big_kernel<<<dim3(1,1,1), dim3(128,1,1)>>>(dA, dB, dC, M, N, K);
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    float ms = 0;
    hipEventElapsedTime(&ms, start, stop);

    opus::bf16_t *hC = new opus::bf16_t[M*N];
    hipMemcpy(hC, dC, M*N*sizeof(opus::bf16_t), hipMemcpyDeviceToHost);

    cpu_ref(hA, hB, hRef, M, N, K);

    float max_diff = 0;
    for (int i = 0; i < M*N; ++i) {
        float diff = fabsf((float)hC[i] - hRef[i]);
        if (diff > max_diff) max_diff = diff;
    }

    float gflops = (2.0f * M * N * K) / (ms * 1e6);
    printf("gfx1201 opus bf16 %dx%dx%d (big_tile,4-wave,128t): max_diff=%.6f %s  %.4fms  %.1f GFlops\n",
           M, N, K, max_diff, max_diff < 1.0f ? "PASS" : "FAIL", ms, gflops);

    delete[] hA; delete[] hB; delete[] hC; delete[] hRef;
    hipFree(dA); hipFree(dB); hipFree(dC);
    return 0;
}
