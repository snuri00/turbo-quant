#include <torch/extension.h>
#include "common.cuh"

template <typename scalar_t>
__global__ void turbo_gqa_score_kernel(
    const scalar_t* __restrict__ query,
    const float* __restrict__ rotation,
    const uint8_t* __restrict__ key_indices,
    const float* __restrict__ centroids,
    const float* __restrict__ key_norms,
    float* __restrict__ scores,
    int batch_size,
    int num_q_heads,
    int num_kv_heads,
    int num_keys,
    int head_dim
) {
    int batch_q_head = blockIdx.x;
    int k_idx = blockIdx.y;

    int b = batch_q_head / num_q_heads;
    int q_head = batch_q_head % num_q_heads;
    int kv_head = q_head / (num_q_heads / num_kv_heads);

    if (b >= batch_size || k_idx >= num_keys) return;

    extern __shared__ float smem[];
    float* q_rotated = smem;

    const scalar_t* q = query + (b * num_q_heads + q_head) * head_dim;
    const uint8_t* k_idx_ptr = key_indices + (b * num_kv_heads + kv_head) * num_keys * head_dim + k_idx * head_dim;
    float k_norm = key_norms[(b * num_kv_heads + kv_head) * num_keys + k_idx];

    for (int j = threadIdx.x; j < head_dim; j += blockDim.x) {
        float dot = 0.0f;
        for (int k = 0; k < head_dim; k++) {
            dot += to_float(q[k]) * rotation[j * head_dim + k];
        }
        q_rotated[j] = dot;
    }
    __syncthreads();

    float score = 0.0f;
    for (int j = threadIdx.x; j < head_dim; j += blockDim.x) {
        score += q_rotated[j] * centroids[k_idx_ptr[j]] * k_norm;
    }

    score = block_reduce_sum(score);
    if (threadIdx.x == 0) {
        int out_idx = (b * num_q_heads + q_head) * num_keys + k_idx;
        scores[out_idx] = score;
    }
}


torch::Tensor turbo_gqa_score_cuda(
    torch::Tensor query,
    torch::Tensor rotation,
    torch::Tensor key_indices,
    torch::Tensor centroids,
    torch::Tensor key_norms,
    int num_q_heads,
    int num_kv_heads
) {
    auto batch_size = query.size(0) / num_q_heads;
    auto num_keys = key_indices.size(1);
    auto head_dim = query.size(1);

    auto scores = torch::zeros({batch_size * num_q_heads, num_keys},
                               torch::dtype(torch::kFloat32).device(query.device()));

    dim3 grid(batch_size * num_q_heads, num_keys);
    int threads = 128;
    int smem_size = head_dim * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        query.scalar_type(), "turbo_gqa_score", ([&] {
            turbo_gqa_score_kernel<scalar_t><<<grid, threads, smem_size>>>(
                query.data_ptr<scalar_t>(),
                rotation.data_ptr<float>(),
                key_indices.data_ptr<uint8_t>(),
                centroids.data_ptr<float>(),
                key_norms.data_ptr<float>(),
                scores.data_ptr<float>(),
                batch_size, num_q_heads, num_kv_heads, num_keys, head_dim
            );
        })
    );

    return scores;
}
