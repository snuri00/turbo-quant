import torch
from torch import Tensor


def turbo_quantize_cuda(input: Tensor, rotation: Tensor, boundaries: Tensor):
    import turbo_quant._C as _C
    flat = input.reshape(-1, input.shape[-1]).contiguous()
    return _C.turbo_quantize_cuda(flat, rotation.float(), boundaries.float())


def turbo_dequantize_cuda(indices: Tensor, centroids: Tensor, rotation_t: Tensor, norms: Tensor, output_dtype=torch.float32):
    import turbo_quant._C as _C
    flat = indices.reshape(-1, indices.shape[-1]).contiguous()
    return _C.turbo_dequantize_cuda(flat, centroids.float(), rotation_t.float(), norms.float(), output_dtype)


def qjl_quantize_cuda(input: Tensor, jl_matrix: Tensor):
    import turbo_quant._C as _C
    flat = input.reshape(-1, input.shape[-1]).contiguous()
    return _C.qjl_quantize_cuda(flat, jl_matrix.float())


def value_quantize_cuda(input: Tensor, bits: int):
    import turbo_quant._C as _C
    flat = input.reshape(-1, input.shape[-1]).contiguous()
    return _C.value_quantize_cuda(flat, bits)


def value_dequantize_cuda(input: Tensor, scales: Tensor, zero_points: Tensor, output_dtype=torch.float32):
    import turbo_quant._C as _C
    flat = input.reshape(-1, input.shape[-1]).contiguous()
    return _C.value_dequantize_cuda(flat, scales.float(), zero_points.float(), output_dtype)
