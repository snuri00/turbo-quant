#include <torch/extension.h>
#include "common.cuh"

template <typename scalar_t>
__global__ void qjl_quantize_kernel(
    const scalar_t* __restrict__ input,
    const float* __restrict__ jl_matrix,
    uint8_t* __restrict__ sign_bits,
    float* __restrict__ norms,
    int batch_size,
    int head_dim,
    int proj_dim,
    int packed_dim
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    const scalar_t* x = input + batch_idx * head_dim;

    float norm_val = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float v = to_float(x[i]);
        norm_val += v * v;
    }
    norm_val = block_reduce_sum(norm_val);
    if (threadIdx.x == 0) {
        norms[batch_idx] = sqrtf(norm_val);
    }

    uint8_t* out = sign_bits + batch_idx * packed_dim;

    for (int byte_idx = threadIdx.x; byte_idx < packed_dim; byte_idx += blockDim.x) {
        uint8_t packed = 0;
        for (int bit = 0; bit < 8; bit++) {
            int j = byte_idx * 8 + bit;
            if (j >= proj_dim) break;

            float dot = 0.0f;
            for (int k = 0; k < head_dim; k++) {
                dot += to_float(x[k]) * jl_matrix[j * head_dim + k];
            }

            if (dot >= 0.0f) packed |= (1 << bit);
        }
        out[byte_idx] = packed;
    }
}


torch::Tensor qjl_quantize_cuda(
    torch::Tensor input,
    torch::Tensor jl_matrix
) {
    auto batch_size = input.size(0);
    auto head_dim = input.size(1);
    auto proj_dim = jl_matrix.size(0);
    auto packed_dim = (proj_dim + 7) / 8;

    auto sign_bits = torch::zeros({batch_size, packed_dim}, torch::dtype(torch::kUInt8).device(input.device()));
    auto norms = torch::zeros({batch_size}, torch::dtype(torch::kFloat32).device(input.device()));

    int threads = 256;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "qjl_quantize", ([&] {
            qjl_quantize_kernel<scalar_t><<<batch_size, threads>>>(
                input.data_ptr<scalar_t>(),
                jl_matrix.data_ptr<float>(),
                sign_bits.data_ptr<uint8_t>(),
                norms.data_ptr<float>(),
                batch_size, head_dim, proj_dim, packed_dim
            );
        })
    );

    return sign_bits;
}
