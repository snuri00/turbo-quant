import torch
from torch import Tensor


def turbo_score_cuda(query: Tensor, rotation: Tensor, key_indices: Tensor, centroids: Tensor, key_norms: Tensor):
    import turbo_quant._C as _C
    q_flat = query.reshape(-1, query.shape[-1]).contiguous()
    k_flat = key_indices.reshape(-1, key_indices.shape[-1]).contiguous()
    n_flat = key_norms.reshape(-1).contiguous()
    return _C.turbo_score_cuda(q_flat, rotation.float(), k_flat, centroids.float(), n_flat)


def qjl_score_cuda(query: Tensor, jl_matrix: Tensor, sign_bits: Tensor, key_norms: Tensor):
    import turbo_quant._C as _C
    q_flat = query.reshape(-1, query.shape[-1]).contiguous()
    s_flat = sign_bits.reshape(-1, sign_bits.shape[-1]).contiguous()
    n_flat = key_norms.reshape(-1).contiguous()
    return _C.qjl_score_cuda(q_flat, jl_matrix.float(), s_flat, n_flat)


def turbo_gqa_score_cuda(query: Tensor, rotation: Tensor, key_indices: Tensor, centroids: Tensor, key_norms: Tensor, num_q_heads: int, num_kv_heads: int):
    import turbo_quant._C as _C
    q_flat = query.reshape(-1, query.shape[-1]).contiguous()
    return _C.turbo_gqa_score_cuda(q_flat, rotation.float(), key_indices, centroids.float(), key_norms, num_q_heads, num_kv_heads)
