#include <torch/extension.h>
#include "common.cuh"

template <typename scalar_t>
__global__ void value_quantize_kernel(
    const scalar_t* __restrict__ input,
    uint8_t* __restrict__ output,
    float* __restrict__ scales,
    float* __restrict__ zero_points,
    int batch_size,
    int dim,
    int max_val
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    const scalar_t* x = input + batch_idx * dim;
    uint8_t* out = output + batch_idx * dim;

    float vmin = 1e10f, vmax = -1e10f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = to_float(x[i]);
        vmin = fminf(vmin, v);
        vmax = fmaxf(vmax, v);
    }

    __shared__ float shared_min[32];
    __shared__ float shared_max[32];

    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        vmin = fminf(vmin, __shfl_down_sync(FULL_MASK, vmin, offset));
        vmax = fmaxf(vmax, __shfl_down_sync(FULL_MASK, vmax, offset));
    }

    if (lane == 0) {
        shared_min[wid] = vmin;
        shared_max[wid] = vmax;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int nwarps = blockDim.x / WARP_SIZE;
        for (int i = 1; i < nwarps; i++) {
            shared_min[0] = fminf(shared_min[0], shared_min[i]);
            shared_max[0] = fmaxf(shared_max[0], shared_max[i]);
        }
        float range = shared_max[0] - shared_min[0];
        if (range < 1e-10f) range = 1.0f;
        scales[batch_idx] = range / (float)max_val;
        zero_points[batch_idx] = shared_min[0];
    }
    __syncthreads();

    float scale = scales[batch_idx];
    float zp = zero_points[batch_idx];
    float inv_scale = 1.0f / scale;

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = to_float(x[i]);
        float q = roundf((v - zp) * inv_scale);
        q = fminf(fmaxf(q, 0.0f), (float)max_val);
        out[i] = (uint8_t)q;
    }
}

template <typename scalar_t>
__global__ void value_dequantize_kernel(
    const uint8_t* __restrict__ input,
    const float* __restrict__ scales,
    const float* __restrict__ zero_points,
    scalar_t* __restrict__ output,
    int batch_size,
    int dim
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    float scale = scales[batch_idx];
    float zp = zero_points[batch_idx];

    const uint8_t* in = input + batch_idx * dim;
    scalar_t* out = output + batch_idx * dim;

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        out[i] = (scalar_t)((float)in[i] * scale + zp);
    }
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> value_quantize_cuda(
    torch::Tensor input,
    int bits
) {
    auto batch_size = input.size(0);
    auto dim = input.size(1);
    int max_val = (1 << bits) - 1;

    auto output = torch::zeros({batch_size, dim}, torch::dtype(torch::kUInt8).device(input.device()));
    auto scales = torch::zeros({batch_size}, torch::dtype(torch::kFloat32).device(input.device()));
    auto zero_points = torch::zeros({batch_size}, torch::dtype(torch::kFloat32).device(input.device()));

    int threads = 256;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "value_quantize", ([&] {
            value_quantize_kernel<scalar_t><<<batch_size, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<uint8_t>(),
                scales.data_ptr<float>(),
                zero_points.data_ptr<float>(),
                batch_size, dim, max_val
            );
        })
    );

    return std::make_tuple(output, scales, zero_points);
}


torch::Tensor value_dequantize_cuda(
    torch::Tensor input,
    torch::Tensor scales,
    torch::Tensor zero_points,
    torch::ScalarType output_dtype
) {
    auto batch_size = input.size(0);
    auto dim = input.size(1);

    auto output = torch::zeros({batch_size, dim}, torch::TensorOptions().dtype(output_dtype).device(input.device()));

    int threads = 256;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        output_dtype, "value_dequantize", ([&] {
            value_dequantize_kernel<scalar_t><<<batch_size, threads>>>(
                input.data_ptr<uint8_t>(),
                scales.data_ptr<float>(),
                zero_points.data_ptr<float>(),
                output.data_ptr<scalar_t>(),
                batch_size, dim
            );
        })
    );

    return output;
}
