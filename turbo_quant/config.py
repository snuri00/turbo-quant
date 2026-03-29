from dataclasses import dataclass, field
from typing import Literal, Optional

import torch


@dataclass
class TurboQuantConfig:
    mode: Literal["turbo_mse", "turbo_prod", "qjl", "polar"] = "turbo_prod"
    key_bits: int = 3
    value_bits: int = 4
    outlier_bits: int = 4
    num_outlier_channels: int = 32
    outlier_threshold_percentile: float = 0.95
    buffer_size: int = 128
    rotation_seed: int = 42
    jl_dim: Optional[int] = None
    orthogonalize_jl: bool = True
    device: str = "cuda"
    dtype: torch.dtype = field(default=torch.bfloat16)

    @property
    def effective_bits(self) -> float:
        head_dim = 128
        regular = head_dim - self.num_outlier_channels
        return (self.num_outlier_channels * self.outlier_bits + regular * self.key_bits) / head_dim
