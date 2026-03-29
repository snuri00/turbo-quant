#include <torch/extension.h>
#include "common.cuh"

#define SQRT_PI_OVER_2 1.2533141373f

template <typename scalar_t>
__global__ void qjl_score_kernel(
    const scalar_t* __restrict__ query,
    const float* __restrict__ jl_matrix,
    const uint8_t* __restrict__ sign_bits,
    const float* __restrict__ key_norms,
    float* __restrict__ scores,
    int num_queries,
    int num_keys,
    int head_dim,
    int proj_dim,
    int packed_dim
) {
    int q_idx = blockIdx.x;
    int k_idx = blockIdx.y;

    if (q_idx >= num_queries || k_idx >= num_keys) return;

    const scalar_t* q = query + q_idx * head_dim;
    const uint8_t* k_bits = sign_bits + k_idx * packed_dim;
    float k_norm = key_norms[k_idx];
    float scale = SQRT_PI_OVER_2 / (float)proj_dim;

    float dot = 0.0f;

    for (int j = threadIdx.x; j < proj_dim; j += blockDim.x) {
        float sq_j = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            sq_j += to_float(q[d]) * jl_matrix[j * head_dim + d];
        }

        int byte_idx = j / 8;
        int bit_idx = j % 8;
        float sign = ((k_bits[byte_idx] >> bit_idx) & 1) ? 1.0f : -1.0f;

        dot += sq_j * sign;
    }

    dot = block_reduce_sum(dot);

    if (threadIdx.x == 0) {
        scores[q_idx * num_keys + k_idx] = scale * k_norm * dot;
    }
}


torch::Tensor qjl_score_cuda(
    torch::Tensor query,
    torch::Tensor jl_matrix,
    torch::Tensor sign_bits,
    torch::Tensor key_norms
) {
    auto num_queries = query.size(0);
    auto num_keys = sign_bits.size(0);
    auto head_dim = query.size(1);
    auto proj_dim = jl_matrix.size(0);
    auto packed_dim = sign_bits.size(1);

    auto scores = torch::zeros({num_queries, num_keys}, torch::dtype(torch::kFloat32).device(query.device()));

    dim3 grid(num_queries, num_keys);
    int threads = 128;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        query.scalar_type(), "qjl_score", ([&] {
            qjl_score_kernel<scalar_t><<<grid, threads>>>(
                query.data_ptr<scalar_t>(),
                jl_matrix.data_ptr<float>(),
                sign_bits.data_ptr<uint8_t>(),
                key_norms.data_ptr<float>(),
                scores.data_ptr<float>(),
                num_queries, num_keys, head_dim, proj_dim, packed_dim
            );
        })
    );

    return scores;
}
