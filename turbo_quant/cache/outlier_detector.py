import torch
from torch import Tensor
from typing import Tuple


class OutlierDetector:
    def __init__(self, num_outlier_channels: int = 32, threshold_percentile: float = 0.95):
        self.num_outlier_channels = num_outlier_channels
        self.threshold_percentile = threshold_percentile

    def detect(self, key_states: Tensor) -> Tensor:
        if key_states.dim() == 4:
            magnitudes = key_states.abs().mean(dim=(0, 2))
        elif key_states.dim() == 3:
            magnitudes = key_states.abs().mean(dim=(0, 1))
        else:
            magnitudes = key_states.abs().mean(dim=0)

        if magnitudes.dim() == 2:
            avg_magnitudes = magnitudes.mean(dim=0)
        else:
            avg_magnitudes = magnitudes

        _, top_indices = torch.topk(avg_magnitudes, self.num_outlier_channels)
        return top_indices.sort().values

    def split_channels(
        self, x: Tensor, outlier_indices: Tensor
    ) -> Tuple[Tensor, Tensor]:
        head_dim = x.shape[-1]
        all_indices = torch.arange(head_dim, device=x.device)
        mask = torch.ones(head_dim, dtype=torch.bool, device=x.device)
        mask[outlier_indices] = False
        regular_indices = all_indices[mask]

        outlier_data = x[..., outlier_indices]
        regular_data = x[..., regular_indices]

        return regular_data, outlier_data

    def merge_channels(
        self, regular: Tensor, outlier: Tensor, outlier_indices: Tensor, head_dim: int
    ) -> Tensor:
        shape = regular.shape[:-1] + (head_dim,)
        result = torch.zeros(shape, dtype=regular.dtype, device=regular.device)

        all_indices = torch.arange(head_dim, device=regular.device)
        mask = torch.ones(head_dim, dtype=torch.bool, device=regular.device)
        mask[outlier_indices] = False
        regular_indices = all_indices[mask]

        result[..., regular_indices] = regular
        result[..., outlier_indices] = outlier

        return result
