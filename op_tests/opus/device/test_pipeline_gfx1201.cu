// test_pipeline_gfx1201.cu — multi-wave 64x64 bf16 GEMM on gfx1201
#ifdef __HIP_DEVICE_COMPILE__
#include "opus/opus.hpp"
#if defined(__gfx1201__) || defined(__gfx1200__)
#include "gfx1201/opus_gemm_pipeline_a16w16_gfx1201.cuh"
__global__ void gemm_pipe_bf16_64x64(const opus::bf16_t* A, const opus::bf16_t* B, opus::bf16_t* C, int K) {
    opus_gfx1201_pipeline::gemm_a16w16_mono_tile_gfx1201_impl<64, 64>(A, B, C, K);
}
#endif
#else
#include "opus/opus.hpp"
#include <cstdio>
#include <hip/hip_runtime.h>
#include <chrono>
__global__ void gemm_pipe_bf16_64x64(const opus::bf16_t*, const opus::bf16_t*, opus::bf16_t*, int) {}
int main() {
    constexpr int M=64,N=64,K=4096;
    opus::bf16_t hA[M*K],hB[N*K];
    for(int i=0;i<M*K;i++)hA[i]=opus::bf16_t(float(i%7)*0.1f);
    for(int i=0;i<N*K;i++)hB[i]=opus::bf16_t(float(i%5)*0.1f);
    float ref[M*N];
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){float s=0;for(int k=0;k<K;k++)s+=float(hA[m*K+k])*float(hB[n*K+k]);ref[m*N+n]=s;}
    opus::bf16_t *dA,*dB,*dC;hipMalloc(&dA,M*K*2);hipMalloc(&dB,N*K*2);hipMalloc(&dC,M*N*2);
    hipMemcpy(dA,hA,M*K*2,hipMemcpyHostToDevice);hipMemcpy(dB,hB,N*K*2,hipMemcpyHostToDevice);
    // 4 waves = 128 threads, 1 grid block
    auto t0=std::chrono::high_resolution_clock::now();
    for(int i=0;i<100;i++) gemm_pipe_bf16_64x64<<<dim3(1),512,0,0>>>(dA,dB,dC,K);
    hipDeviceSynchronize();
    auto t1=std::chrono::high_resolution_clock::now();
    double ms=std::chrono::duration<double,std::milli>(t1-t0).count()/100;
    opus::bf16_t hC[M*N];hipMemcpy(hC,dC,M*N*2,hipMemcpyDeviceToHost);
    float md=0;for(int i=0;i<M*N;i++){float d=float(hC[i])-ref[i];if(d<0)d=-d;if(d>md)md=d;}
    double gflops=2.0*M*N*K/ms/1e6;
    printf("gfx1201 opus bf16 64x64x%d: max_diff=%f %s  %.4fms  %.1f GFlops\n",K,md,md<1.0f?"PASS":"FAIL",ms,gflops);
    hipFree(dA);hipFree(dB);hipFree(dC);
    return md<1.0f?0:1;
}
#endif
