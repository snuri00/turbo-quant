import math
from typing import Tuple

import torch
from torch import Tensor


class ValueQuantizer:
    def __init__(self, bits: int = 4, device: str = "cpu", dtype: torch.dtype = torch.float32):
        self.bits = bits
        self.device = device
        self.dtype = dtype
        self.max_val = (1 << bits) - 1

    def quantize(self, v: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        v_min = v.min(dim=-1, keepdim=True).values
        v_max = v.max(dim=-1, keepdim=True).values
        scale = (v_max - v_min) / self.max_val
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        zero_point = v_min

        quantized = torch.round((v - zero_point) / scale).clamp(0, self.max_val).to(torch.uint8)

        return quantized, scale.to(self.dtype), zero_point.to(self.dtype)

    def dequantize(self, quantized: Tensor, scale: Tensor, zero_point: Tensor) -> Tensor:
        return quantized.to(self.dtype) * scale + zero_point

    def quantize_and_pack(self, v: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        quantized, scale, zero_point = self.quantize(v)
        packed = self._pack(quantized)
        return packed, scale, zero_point

    def unpack_and_dequantize(self, packed: Tensor, scale: Tensor, zero_point: Tensor) -> Tensor:
        quantized = self._unpack(packed)
        return self.dequantize(quantized, scale, zero_point)

    def _pack(self, quantized: Tensor) -> Tensor:
        if self.bits == 4:
            shape = quantized.shape
            flat = quantized.reshape(-1, shape[-1])
            packed_dim = math.ceil(shape[-1] / 2)
            packed = torch.zeros(flat.shape[0], packed_dim, dtype=torch.uint8, device=quantized.device)
            for i in range(0, shape[-1], 2):
                idx = i // 2
                packed[:, idx] = flat[:, i]
                if i + 1 < shape[-1]:
                    packed[:, idx] |= flat[:, i + 1] << 4
            return packed.reshape(shape[:-1] + (packed_dim,))

        elif self.bits == 2:
            shape = quantized.shape
            flat = quantized.reshape(-1, shape[-1])
            packed_dim = math.ceil(shape[-1] / 4)
            packed = torch.zeros(flat.shape[0], packed_dim, dtype=torch.uint8, device=quantized.device)
            for i in range(shape[-1]):
                byte_idx = i // 4
                shift = (i % 4) * 2
                packed[:, byte_idx] |= flat[:, i] << shift
            return packed.reshape(shape[:-1] + (packed_dim,))

        return quantized

    def _unpack(self, packed: Tensor) -> Tensor:
        if self.bits == 4:
            shape = packed.shape
            flat = packed.reshape(-1, shape[-1])
            head_dim = shape[-1] * 2
            unpacked = torch.zeros(flat.shape[0], head_dim, dtype=torch.uint8, device=packed.device)
            for i in range(shape[-1]):
                unpacked[:, 2 * i] = flat[:, i] & 0xF
                unpacked[:, 2 * i + 1] = (flat[:, i] >> 4) & 0xF
            return unpacked.reshape(shape[:-1] + (head_dim,))

        elif self.bits == 2:
            shape = packed.shape
            flat = packed.reshape(-1, shape[-1])
            head_dim = shape[-1] * 4
            unpacked = torch.zeros(flat.shape[0], head_dim, dtype=torch.uint8, device=packed.device)
            for i in range(head_dim):
                byte_idx = i // 4
                shift = (i % 4) * 2
                unpacked[:, i] = (flat[:, byte_idx] >> shift) & 0x3
            return unpacked.reshape(shape[:-1] + (head_dim,))

        return packed
