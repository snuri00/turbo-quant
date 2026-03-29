#include <torch/extension.h>
#include "common.cuh"

template <typename scalar_t>
__global__ void turbo_score_kernel(
    const scalar_t* __restrict__ query,
    const float* __restrict__ rotation,
    const uint8_t* __restrict__ key_indices,
    const float* __restrict__ centroids,
    const float* __restrict__ key_norms,
    float* __restrict__ scores,
    int num_queries,
    int num_keys,
    int head_dim
) {
    int q_idx = blockIdx.x;
    int k_idx = blockIdx.y;

    if (q_idx >= num_queries || k_idx >= num_keys) return;

    extern __shared__ float smem[];
    float* q_rotated = smem;

    const scalar_t* q = query + q_idx * head_dim;

    if (k_idx == 0) {
        float q_norm = 0.0f;
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
            float v = to_float(q[i]);
            q_norm += v * v;
        }
        q_norm = block_reduce_sum(q_norm);

        __shared__ float shared_q_norm;
        if (threadIdx.x == 0) shared_q_norm = sqrtf(q_norm + 1e-10f);
        __syncthreads();

        float inv_q_norm = 1.0f / shared_q_norm;
        for (int j = threadIdx.x; j < head_dim; j += blockDim.x) {
            float dot = 0.0f;
            for (int k = 0; k < head_dim; k++) {
                dot += to_float(q[k]) * inv_q_norm * rotation[j * head_dim + k];
            }
            q_rotated[j] = dot * shared_q_norm;
        }
    }
    __syncthreads();

    const uint8_t* k_idx_ptr = key_indices + k_idx * head_dim;
    float k_norm = key_norms[k_idx];

    float score = 0.0f;
    for (int j = threadIdx.x; j < head_dim; j += blockDim.x) {
        score += q_rotated[j] * centroids[k_idx_ptr[j]] * k_norm;
    }

    score = block_reduce_sum(score);
    if (threadIdx.x == 0) {
        scores[q_idx * num_keys + k_idx] = score;
    }
}


torch::Tensor turbo_score_cuda(
    torch::Tensor query,
    torch::Tensor rotation,
    torch::Tensor key_indices,
    torch::Tensor centroids,
    torch::Tensor key_norms
) {
    auto num_queries = query.size(0);
    auto num_keys = key_indices.size(0);
    auto head_dim = query.size(1);

    auto scores = torch::zeros({num_queries, num_keys}, torch::dtype(torch::kFloat32).device(query.device()));

    dim3 grid(num_queries, num_keys);
    int threads = 128;
    int smem_size = head_dim * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        query.scalar_type(), "turbo_score", ([&] {
            turbo_score_kernel<scalar_t><<<grid, threads, smem_size>>>(
                query.data_ptr<scalar_t>(),
                rotation.data_ptr<float>(),
                key_indices.data_ptr<uint8_t>(),
                centroids.data_ptr<float>(),
                key_norms.data_ptr<float>(),
                scores.data_ptr<float>(),
                num_queries, num_keys, head_dim
            );
        })
    );

    return scores;
}
