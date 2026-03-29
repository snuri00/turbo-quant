#include <torch/extension.h>
#include "common.cuh"

template <typename scalar_t>
__global__ void turbo_quantize_kernel(
    const scalar_t* __restrict__ input,
    const float* __restrict__ rotation,
    const float* __restrict__ boundaries,
    uint8_t* __restrict__ output,
    float* __restrict__ norms,
    int batch_size,
    int head_dim,
    int num_boundaries
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    extern __shared__ float smem[];
    float* rotated = smem;

    const scalar_t* x = input + batch_idx * head_dim;
    uint8_t* out = output + batch_idx * head_dim;

    float norm_val = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float v = to_float(x[i]);
        norm_val += v * v;
    }
    norm_val = block_reduce_sum(norm_val);

    __shared__ float shared_norm;
    if (threadIdx.x == 0) {
        shared_norm = sqrtf(norm_val + 1e-10f);
        norms[batch_idx] = shared_norm;
    }
    __syncthreads();

    float inv_norm = 1.0f / shared_norm;

    for (int j = threadIdx.x; j < head_dim; j += blockDim.x) {
        float dot = 0.0f;
        for (int k = 0; k < head_dim; k++) {
            dot += to_float(x[k]) * inv_norm * rotation[j * head_dim + k];
        }
        rotated[j] = dot;
    }
    __syncthreads();

    for (int j = threadIdx.x; j < head_dim; j += blockDim.x) {
        int idx = binary_search(boundaries, num_boundaries, rotated[j]);
        out[j] = (uint8_t)idx;
    }
}


torch::Tensor turbo_quantize_cuda(
    torch::Tensor input,
    torch::Tensor rotation,
    torch::Tensor boundaries
) {
    auto batch_size = input.size(0);
    auto head_dim = input.size(1);
    auto num_boundaries = boundaries.size(0);

    auto output = torch::zeros({batch_size, head_dim}, torch::dtype(torch::kUInt8).device(input.device()));
    auto norms = torch::zeros({batch_size}, torch::dtype(torch::kFloat32).device(input.device()));

    int threads = 256;
    int smem_size = head_dim * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "turbo_quantize", ([&] {
            turbo_quantize_kernel<scalar_t><<<batch_size, threads, smem_size>>>(
                input.data_ptr<scalar_t>(),
                rotation.data_ptr<float>(),
                boundaries.data_ptr<float>(),
                output.data_ptr<uint8_t>(),
                norms.data_ptr<float>(),
                batch_size, head_dim, num_boundaries
            );
        })
    );

    return output;
}
