from typing import Dict, Tuple

import torch
from torch import Tensor


_ROTATION_CACHE: Dict[Tuple[int, int, str], Tensor] = {}
_JL_CACHE: Dict[Tuple[int, int, int, str, bool], Tensor] = {}


def generate_rotation_matrix(
    d: int,
    seed: int = 42,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    key = (d, seed, device)
    if key in _ROTATION_CACHE:
        cached = _ROTATION_CACHE[key]
        if cached.dtype == dtype:
            return cached
        return cached.to(dtype)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    gaussian = torch.randn(d, d, generator=gen, dtype=torch.float32)
    q, r = torch.linalg.qr(gaussian)
    diag_sign = torch.sign(torch.diag(r))
    q = q * diag_sign.unsqueeze(0)

    _ROTATION_CACHE[key] = q.to(device)
    return q.to(device=device, dtype=dtype)


def generate_jl_matrix(
    m: int,
    d: int,
    seed: int = 42,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    orthogonalize: bool = True,
) -> Tensor:
    key = (m, d, seed, device, orthogonalize)
    if key in _JL_CACHE:
        cached = _JL_CACHE[key]
        if cached.dtype == dtype:
            return cached
        return cached.to(dtype)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + 1000)
    gaussian = torch.randn(m, d, generator=gen, dtype=torch.float32)

    if orthogonalize and m == d:
        q, r = torch.linalg.qr(gaussian)
        diag_sign = torch.sign(torch.diag(r))
        result = q * diag_sign.unsqueeze(0)
    elif orthogonalize and m < d:
        q, r = torch.linalg.qr(gaussian.T)
        diag_sign = torch.sign(torch.diag(r))
        result = (q * diag_sign.unsqueeze(0)).T[:m]
    else:
        result = gaussian

    _JL_CACHE[key] = result.to(device)
    return result.to(device=device, dtype=dtype)


def clear_caches():
    _ROTATION_CACHE.clear()
    _JL_CACHE.clear()
