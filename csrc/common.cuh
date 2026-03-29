#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warp_reduce_sum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    return val;
}

__device__ __forceinline__ float to_float(half v) { return __half2float(v); }
__device__ __forceinline__ float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float to_float(float v) { return v; }

__device__ __forceinline__ int binary_search(const float* boundaries, int n, float val) {
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (val > boundaries[mid]) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}
