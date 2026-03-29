import math
from typing import Tuple

import torch
from torch import Tensor

from turbo_quant.quantizers.codebook import get_optimal_codebook
from turbo_quant.quantizers.rotation import generate_rotation_matrix


class TurboQuantMSE:
    def __init__(
        self,
        head_dim: int,
        bits: int,
        seed: int = 42,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.head_dim = head_dim
        self.bits = bits
        self.seed = seed
        self.device = device
        self.dtype = dtype

        self.rotation = generate_rotation_matrix(head_dim, seed, device, dtype)
        self.rotation_t = self.rotation.T.contiguous()

        centroids, boundaries = get_optimal_codebook(head_dim, bits, device)
        self.centroids = centroids.to(dtype)
        self.boundaries = boundaries.to(dtype)
        self.num_centroids = len(centroids)

    def quantize(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        original_shape = x.shape
        x_flat = x.reshape(-1, self.head_dim)
        norms = torch.norm(x_flat, dim=-1, keepdim=True)
        x_normalized = x_flat / (norms + 1e-10)

        y = x_normalized @ self.rotation.T

        indices = torch.bucketize(y, self.boundaries[1:-1])
        indices = torch.clamp(indices, 0, self.num_centroids - 1)

        return indices.reshape(original_shape).to(torch.uint8), norms.reshape(original_shape[:-1] + (1,))

    def dequantize(self, indices: Tensor, norms: Tensor) -> Tensor:
        original_shape = indices.shape
        indices_flat = indices.reshape(-1, self.head_dim).long()

        y_hat = self.centroids[indices_flat]
        x_hat = y_hat @ self.rotation

        norms_flat = norms.reshape(-1, 1)
        x_hat = x_hat * norms_flat

        return x_hat.reshape(original_shape[:-1] + (self.head_dim,)).to(self.dtype)

    def quantize_and_pack(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        indices, norms = self.quantize(x)
        packed = self._pack_bits(indices)
        return packed, norms

    def unpack_and_dequantize(self, packed: Tensor, norms: Tensor, seq_len: int) -> Tensor:
        indices = self._unpack_bits(packed, seq_len)
        return self.dequantize(indices, norms)

    def _pack_bits(self, indices: Tensor) -> Tensor:
        shape = indices.shape
        flat = indices.reshape(-1, self.head_dim).to(torch.uint8)
        batch = flat.shape[0]

        if self.bits == 1:
            packed_dim = math.ceil(self.head_dim / 8)
            packed = torch.zeros(batch, packed_dim, dtype=torch.uint8, device=indices.device)
            for i in range(8):
                if i < self.head_dim:
                    col_start = i
                    cols = torch.arange(col_start, self.head_dim, 8, device=indices.device)
                    for j, c in enumerate(cols):
                        packed[:, j] |= (flat[:, c] & 1) << i
            return packed.reshape(shape[:-1] + (packed_dim,))

        elif self.bits == 2:
            packed_dim = math.ceil(self.head_dim / 4)
            packed = torch.zeros(batch, packed_dim, dtype=torch.uint8, device=indices.device)
            for i in range(min(4, self.head_dim)):
                cols = torch.arange(i, self.head_dim, 4, device=indices.device)
                for j, c in enumerate(cols):
                    packed[:, j] |= (flat[:, c] & 0x3) << (i * 2)
            return packed.reshape(shape[:-1] + (packed_dim,))

        elif self.bits == 3:
            packed_dim = math.ceil(self.head_dim * 3 / 8)
            packed = torch.zeros(batch, packed_dim, dtype=torch.uint8, device=indices.device)
            for idx in range(self.head_dim):
                bit_offset = idx * 3
                byte_idx = bit_offset // 8
                bit_idx = bit_offset % 8
                val = flat[:, idx].to(torch.int32) & 0x7
                if byte_idx < packed_dim:
                    packed[:, byte_idx] |= (val << bit_idx).to(torch.uint8)
                if bit_idx > 5 and byte_idx + 1 < packed_dim:
                    packed[:, byte_idx + 1] |= (val >> (8 - bit_idx)).to(torch.uint8)
            return packed.reshape(shape[:-1] + (packed_dim,))

        elif self.bits == 4:
            packed_dim = math.ceil(self.head_dim / 2)
            packed = torch.zeros(batch, packed_dim, dtype=torch.uint8, device=indices.device)
            for i in range(0, self.head_dim, 2):
                pack_idx = i // 2
                packed[:, pack_idx] = flat[:, i] & 0xF
                if i + 1 < self.head_dim:
                    packed[:, pack_idx] |= (flat[:, i + 1] & 0xF) << 4
            return packed.reshape(shape[:-1] + (packed_dim,))

        return indices

    def _unpack_bits(self, packed: Tensor, original_dim_0: int) -> Tensor:
        shape = packed.shape
        packed_flat = packed.reshape(-1, shape[-1])
        batch = packed_flat.shape[0]

        indices = torch.zeros(batch, self.head_dim, dtype=torch.uint8, device=packed.device)

        if self.bits == 1:
            for i in range(self.head_dim):
                byte_idx = i // 8
                bit_idx = i % 8
                indices[:, i] = (packed_flat[:, byte_idx] >> bit_idx) & 1

        elif self.bits == 2:
            for i in range(self.head_dim):
                byte_idx = i // 4
                bit_idx = (i % 4) * 2
                indices[:, i] = (packed_flat[:, byte_idx] >> bit_idx) & 0x3

        elif self.bits == 3:
            for i in range(self.head_dim):
                bit_offset = i * 3
                byte_idx = bit_offset // 8
                bit_idx = bit_offset % 8
                val = packed_flat[:, byte_idx].to(torch.int32) >> bit_idx
                if bit_idx > 5 and byte_idx + 1 < shape[-1]:
                    val |= packed_flat[:, byte_idx + 1].to(torch.int32) << (8 - bit_idx)
                indices[:, i] = (val & 0x7).to(torch.uint8)

        elif self.bits == 4:
            for i in range(self.head_dim):
                byte_idx = i // 2
                if i % 2 == 0:
                    indices[:, i] = packed_flat[:, byte_idx] & 0xF
                else:
                    indices[:, i] = (packed_flat[:, byte_idx] >> 4) & 0xF

        return indices.reshape(shape[:-1] + (self.head_dim,))

    def compute_rotated_score(self, query: Tensor, indices: Tensor, norms: Tensor) -> Tensor:
        q_flat = query.reshape(-1, self.head_dim)
        q_norms = torch.norm(q_flat, dim=-1, keepdim=True)
        q_normalized = q_flat / (q_norms + 1e-10)
        q_rotated = q_normalized @ self.rotation.T

        idx_flat = indices.reshape(-1, indices.shape[-2], self.head_dim).long()
        k_rotated = self.centroids[idx_flat]

        norms_flat = norms.reshape(-1, indices.shape[-2], 1)
        k_rotated = k_rotated * norms_flat

        scores = torch.bmm(
            q_rotated.unsqueeze(1) * q_norms.unsqueeze(1),
            k_rotated.transpose(-1, -2)
        ).squeeze(1)

        return scores.reshape(query.shape[:-1] + (indices.shape[-2],))
