import math
from typing import Tuple, NamedTuple

import torch
from torch import Tensor

from turbo_quant.quantizers.turbo_mse import TurboQuantMSE
from turbo_quant.quantizers.qjl import QJLQuantizer


class QuantizedKeyProd(NamedTuple):
    mse_indices: Tensor
    mse_norms: Tensor
    qjl_bits: Tensor
    residual_norms: Tensor


class TurboQuantProd:
    def __init__(
        self,
        head_dim: int,
        bits: int,
        seed: int = 42,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        orthogonalize_jl: bool = True,
    ):
        self.head_dim = head_dim
        self.bits = bits
        self.seed = seed
        self.device = device
        self.dtype = dtype

        self.mse_quantizer = TurboQuantMSE(
            head_dim=head_dim,
            bits=max(bits - 1, 1),
            seed=seed,
            device=device,
            dtype=dtype,
        )

        self.qjl_quantizer = QJLQuantizer(
            head_dim=head_dim,
            proj_dim=head_dim,
            seed=seed + 500,
            device=device,
            dtype=dtype,
            orthogonalize=orthogonalize_jl,
        )

    def quantize(self, x: Tensor) -> QuantizedKeyProd:
        mse_indices, mse_norms = self.mse_quantizer.quantize(x)
        x_hat = self.mse_quantizer.dequantize(mse_indices, mse_norms)
        residual = x - x_hat

        qjl_bits, residual_norms = self.qjl_quantizer.quantize(residual)

        return QuantizedKeyProd(
            mse_indices=mse_indices,
            mse_norms=mse_norms,
            qjl_bits=qjl_bits,
            residual_norms=residual_norms,
        )

    def dequantize(self, quant: QuantizedKeyProd) -> Tensor:
        x_mse = self.mse_quantizer.dequantize(quant.mse_indices, quant.mse_norms)
        x_qjl = self.qjl_quantizer.dequantize(quant.qjl_bits, quant.residual_norms)
        return x_mse + x_qjl

    def estimate_inner_product(self, query: Tensor, quant: QuantizedKeyProd) -> Tensor:
        mse_scores = self.mse_quantizer.compute_rotated_score(
            query, quant.mse_indices, quant.mse_norms
        )

        qjl_scores = self.qjl_quantizer.estimate_inner_product(
            query, quant.qjl_bits, quant.residual_norms
        )

        return mse_scores + qjl_scores

    def quantize_and_pack(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        mse_packed, mse_norms = self.mse_quantizer.quantize_and_pack(x)
        x_hat = self.mse_quantizer.unpack_and_dequantize(
            mse_packed, mse_norms, x.shape[-2] if x.dim() > 1 else 1
        )
        residual = x - x_hat

        qjl_bits, residual_norms = self.qjl_quantizer.quantize(residual)

        return mse_packed, mse_norms, qjl_bits, residual_norms

    def memory_bytes_per_token(self) -> int:
        mse_bits = (self.bits - 1) * self.head_dim
        qjl_bits = self.head_dim
        norm_bits = 16 + 16
        total_bits = mse_bits + qjl_bits + norm_bits
        return math.ceil(total_bits / 8)

    def compression_ratio(self) -> float:
        original_bytes = self.head_dim * 2
        return original_bytes / self.memory_bytes_per_token()
