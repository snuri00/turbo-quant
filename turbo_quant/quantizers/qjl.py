import math
from typing import Tuple

import torch
from torch import Tensor

from turbo_quant.quantizers.rotation import generate_jl_matrix


class QJLQuantizer:
    def __init__(
        self,
        head_dim: int,
        proj_dim: int = None,
        seed: int = 42,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        orthogonalize: bool = True,
    ):
        self.head_dim = head_dim
        self.proj_dim = proj_dim or head_dim
        self.seed = seed
        self.device = device
        self.dtype = dtype

        self.S = generate_jl_matrix(
            self.proj_dim, head_dim, seed, device, dtype, orthogonalize
        )
        self.S_t = self.S.T.contiguous()
        self.scale = math.sqrt(math.pi / 2) / self.proj_dim

    def quantize(self, k: Tensor) -> Tuple[Tensor, Tensor]:
        original_shape = k.shape
        k_flat = k.reshape(-1, self.head_dim)

        norms = torch.norm(k_flat, dim=-1)
        projected = k_flat @ self.S.T
        sign_bits = (projected >= 0).to(torch.uint8)

        packed = self._pack_sign_bits(sign_bits)

        return (
            packed.reshape(original_shape[:-1] + (packed.shape[-1],)),
            norms.reshape(original_shape[:-1]),
        )

    def estimate_inner_product(
        self, query: Tensor, sign_bits_packed: Tensor, norms: Tensor
    ) -> Tensor:
        q_shape = query.shape
        q_flat = query.reshape(-1, self.head_dim)

        Sq = q_flat @ self.S.T

        sign_bits = self._unpack_sign_bits(sign_bits_packed)
        s_flat = sign_bits.reshape(-1, sign_bits.shape[-2], self.proj_dim)
        signs = 2.0 * s_flat.to(self.dtype) - 1.0

        norms_flat = norms.reshape(-1, norms.shape[-1])

        dots = torch.bmm(Sq.unsqueeze(1), signs.transpose(-1, -2)).squeeze(1)
        scores = self.scale * norms_flat * dots

        return scores.reshape(q_shape[:-1] + (sign_bits.shape[-2],))

    def dequantize(self, sign_bits_packed: Tensor, norms: Tensor) -> Tensor:
        sign_bits = self._unpack_sign_bits(sign_bits_packed)
        signs = 2.0 * sign_bits.to(self.dtype) - 1.0

        original_shape = norms.shape
        signs_flat = signs.reshape(-1, self.proj_dim)
        norms_flat = norms.reshape(-1)

        reconstructed = self.scale * norms_flat.unsqueeze(-1) * (signs_flat @ self.S)

        return reconstructed.reshape(original_shape + (self.head_dim,))

    def _pack_sign_bits(self, bits: Tensor) -> Tensor:
        shape = bits.shape
        flat = bits.reshape(-1, self.proj_dim)
        batch = flat.shape[0]
        packed_dim = math.ceil(self.proj_dim / 8)

        packed = torch.zeros(batch, packed_dim, dtype=torch.uint8, device=bits.device)
        for i in range(self.proj_dim):
            byte_idx = i // 8
            bit_idx = i % 8
            packed[:, byte_idx] |= flat[:, i] << bit_idx

        return packed.reshape(shape[:-1] + (packed_dim,))

    def _unpack_sign_bits(self, packed: Tensor) -> Tensor:
        shape = packed.shape
        flat = packed.reshape(-1, shape[-1])
        batch = flat.shape[0]

        bits = torch.zeros(batch, self.proj_dim, dtype=torch.uint8, device=packed.device)
        for i in range(self.proj_dim):
            byte_idx = i // 8
            bit_idx = i % 8
            bits[:, i] = (flat[:, byte_idx] >> bit_idx) & 1

        return bits.reshape(shape[:-1] + (self.proj_dim,))
