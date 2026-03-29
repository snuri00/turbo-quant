#include <torch/extension.h>
#include "common.cuh"

__global__ void polar_forward_level1_kernel(
    const float* __restrict__ input,
    float* __restrict__ angles,
    float* __restrict__ radii,
    int batch_size,
    int dim
) {
    int batch_idx = blockIdx.x;
    int pair_idx = blockIdx.y * blockDim.x + threadIdx.x;
    int num_pairs = dim / 2;

    if (batch_idx >= batch_size || pair_idx >= num_pairs) return;

    const float* x = input + batch_idx * dim;
    float x1 = x[2 * pair_idx];
    float x2 = x[2 * pair_idx + 1];

    angles[batch_idx * num_pairs + pair_idx] = atan2f(x2, x1);
    radii[batch_idx * num_pairs + pair_idx] = sqrtf(x1 * x1 + x2 * x2);
}

__global__ void polar_forward_level_kernel(
    const float* __restrict__ radii_in,
    float* __restrict__ angles_out,
    float* __restrict__ radii_out,
    int batch_size,
    int num_pairs
) {
    int batch_idx = blockIdx.x;
    int pair_idx = blockIdx.y * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size || pair_idx >= num_pairs) return;

    float r1 = radii_in[batch_idx * num_pairs * 2 + 2 * pair_idx];
    float r2 = radii_in[batch_idx * num_pairs * 2 + 2 * pair_idx + 1];

    angles_out[batch_idx * num_pairs + pair_idx] = atan2f(r2, r1);
    radii_out[batch_idx * num_pairs + pair_idx] = sqrtf(r1 * r1 + r2 * r2);
}

__global__ void polar_inverse_level1_kernel(
    const float* __restrict__ angles,
    const float* __restrict__ radii,
    float* __restrict__ output,
    int batch_size,
    int num_pairs
) {
    int batch_idx = blockIdx.x;
    int pair_idx = blockIdx.y * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size || pair_idx >= num_pairs) return;

    float angle = angles[batch_idx * num_pairs + pair_idx];
    float radius = radii[batch_idx * num_pairs + pair_idx];

    output[batch_idx * num_pairs * 2 + 2 * pair_idx] = radius * cosf(angle);
    output[batch_idx * num_pairs * 2 + 2 * pair_idx + 1] = radius * sinf(angle);
}

__global__ void polar_inverse_level_kernel(
    const float* __restrict__ angles,
    const float* __restrict__ radii_in,
    float* __restrict__ radii_out,
    int batch_size,
    int num_pairs
) {
    int batch_idx = blockIdx.x;
    int pair_idx = blockIdx.y * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size || pair_idx >= num_pairs) return;

    float angle = angles[batch_idx * num_pairs + pair_idx];
    float radius = radii_in[batch_idx * num_pairs + pair_idx];

    radii_out[batch_idx * num_pairs * 2 + 2 * pair_idx] = radius * cosf(angle);
    radii_out[batch_idx * num_pairs * 2 + 2 * pair_idx + 1] = radius * sinf(angle);
}
