#include <torch/extension.h>
#include "common.cuh"

template <typename scalar_t>
__global__ void turbo_dequantize_kernel(
    const uint8_t* __restrict__ indices,
    const float* __restrict__ centroids,
    const float* __restrict__ rotation_t,
    const float* __restrict__ norms,
    scalar_t* __restrict__ output,
    int batch_size,
    int head_dim
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    extern __shared__ float smem[];
    float* centroid_vals = smem;

    const uint8_t* idx = indices + batch_idx * head_dim;
    scalar_t* out = output + batch_idx * head_dim;
    float norm = norms[batch_idx];

    for (int j = threadIdx.x; j < head_dim; j += blockDim.x) {
        centroid_vals[j] = centroids[idx[j]];
    }
    __syncthreads();

    for (int j = threadIdx.x; j < head_dim; j += blockDim.x) {
        float val = 0.0f;
        for (int k = 0; k < head_dim; k++) {
            val += centroid_vals[k] * rotation_t[k * head_dim + j];
        }
        out[j] = (scalar_t)(val * norm);
    }
}


torch::Tensor turbo_dequantize_cuda(
    torch::Tensor indices,
    torch::Tensor centroids,
    torch::Tensor rotation_t,
    torch::Tensor norms,
    torch::ScalarType output_dtype
) {
    auto batch_size = indices.size(0);
    auto head_dim = indices.size(1);

    auto options = torch::TensorOptions().dtype(output_dtype).device(indices.device());
    auto output = torch::zeros({batch_size, head_dim}, options);

    int threads = 256;
    int smem_size = head_dim * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        output_dtype, "turbo_dequantize", ([&] {
            turbo_dequantize_kernel<scalar_t><<<batch_size, threads, smem_size>>>(
                indices.data_ptr<uint8_t>(),
                centroids.data_ptr<float>(),
                rotation_t.data_ptr<float>(),
                norms.data_ptr<float>(),
                output.data_ptr<scalar_t>(),
                batch_size, head_dim
            );
        })
    );

    return output;
}
