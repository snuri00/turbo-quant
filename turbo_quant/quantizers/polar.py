import math
from typing import List, Tuple

import torch
from torch import Tensor
import numpy as np
from scipy.special import gamma as gamma_fn
from scipy.integrate import quad

from turbo_quant.quantizers.rotation import generate_rotation_matrix


_POLAR_CODEBOOK_CACHE = {}


def polar_angle_pdf(theta: np.ndarray, d_level: int) -> np.ndarray:
    k = d_level
    coeff = gamma_fn(k) / (2 ** (k - 2) * gamma_fn(k / 2) ** 2)
    return coeff * np.power(np.sin(2 * theta), k - 1)


def solve_polar_codebook(d_level: int, bits: int) -> Tuple[np.ndarray, np.ndarray]:
    key = (d_level, bits)
    if key in _POLAR_CODEBOOK_CACHE:
        return _POLAR_CODEBOOK_CACHE[key]

    n_centroids = 2 ** bits
    centroids = np.linspace(0.01, math.pi / 2 - 0.01, n_centroids)

    for _ in range(2000):
        boundaries = np.zeros(n_centroids + 1)
        boundaries[0] = 0.0
        boundaries[-1] = math.pi / 2
        for i in range(n_centroids - 1):
            boundaries[i + 1] = (centroids[i] + centroids[i + 1]) / 2

        new_centroids = np.zeros(n_centroids)
        for i in range(n_centroids):
            num, _ = quad(lambda t: t * polar_angle_pdf(np.array([t]), d_level)[0],
                          boundaries[i], boundaries[i + 1])
            den, _ = quad(lambda t: polar_angle_pdf(np.array([t]), d_level)[0],
                          boundaries[i], boundaries[i + 1])
            new_centroids[i] = num / den if abs(den) > 1e-15 else (boundaries[i] + boundaries[i + 1]) / 2

        if np.max(np.abs(new_centroids - centroids)) < 1e-14:
            centroids = new_centroids
            break
        centroids = new_centroids

    boundaries = np.zeros(n_centroids + 1)
    boundaries[0] = 0.0
    boundaries[-1] = math.pi / 2
    for i in range(n_centroids - 1):
        boundaries[i + 1] = (centroids[i] + centroids[i + 1]) / 2

    _POLAR_CODEBOOK_CACHE[key] = (centroids, boundaries)
    return centroids, boundaries


class PolarQuantizer:
    def __init__(
        self,
        head_dim: int,
        bits: int = 3,
        seed: int = 42,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.head_dim = head_dim
        self.bits = bits
        self.seed = seed
        self.device = device
        self.dtype = dtype
        self.num_levels = int(math.log2(head_dim))

        self.precondition_matrix = generate_rotation_matrix(head_dim, seed, device, dtype)

        self.level_codebooks = {}
        self.level_boundaries = {}

        for level in range(2, self.num_levels + 1):
            d_level = 2 ** (level - 1)
            c, b = solve_polar_codebook(d_level, bits)
            self.level_codebooks[level] = torch.from_numpy(c).float().to(device)
            self.level_boundaries[level] = torch.from_numpy(b).float().to(device)

    def _cartesian_to_polar(self, y: Tensor) -> Tuple[List[Tensor], Tensor]:
        batch_shape = y.shape[:-1]
        y_flat = y.reshape(-1, self.head_dim)
        batch = y_flat.shape[0]

        angles_per_level = []

        radii = torch.zeros(batch, self.head_dim // 2, device=y.device, dtype=torch.float32)
        level1_angles = torch.zeros(batch, self.head_dim // 2, device=y.device, dtype=torch.float32)

        for j in range(self.head_dim // 2):
            x1 = y_flat[:, 2 * j]
            x2 = y_flat[:, 2 * j + 1]
            level1_angles[:, j] = torch.atan2(x2, x1)
            radii[:, j] = torch.sqrt(x1 ** 2 + x2 ** 2)

        angles_per_level.append(level1_angles)

        current_radii = radii
        for level in range(2, self.num_levels + 1):
            num_pairs = current_radii.shape[-1] // 2
            new_angles = torch.zeros(batch, num_pairs, device=y.device, dtype=torch.float32)
            new_radii = torch.zeros(batch, num_pairs, device=y.device, dtype=torch.float32)

            for j in range(num_pairs):
                r1 = current_radii[:, 2 * j]
                r2 = current_radii[:, 2 * j + 1]
                new_angles[:, j] = torch.atan2(r2, r1)
                new_radii[:, j] = torch.sqrt(r1 ** 2 + r2 ** 2)

            angles_per_level.append(new_angles)
            current_radii = new_radii

        final_radius = current_radii.squeeze(-1)
        return angles_per_level, final_radius

    def _polar_to_cartesian(
        self, angles_per_level: List[Tensor], final_radius: Tensor
    ) -> Tensor:
        batch = final_radius.shape[0]

        current_radii = final_radius.unsqueeze(-1)

        for level in range(self.num_levels, 1, -1):
            angles = angles_per_level[level - 1]
            num_pairs = angles.shape[-1]
            new_radii = torch.zeros(batch, num_pairs * 2, device=final_radius.device, dtype=torch.float32)

            for j in range(num_pairs):
                r = current_radii[:, j]
                a = angles[:, j]
                new_radii[:, 2 * j] = r * torch.cos(a)
                new_radii[:, 2 * j + 1] = r * torch.sin(a)

            current_radii = new_radii

        level1_angles = angles_per_level[0]
        output = torch.zeros(batch, self.head_dim, device=final_radius.device, dtype=torch.float32)

        for j in range(self.head_dim // 2):
            r = current_radii[:, j]
            a = level1_angles[:, j]
            output[:, 2 * j] = r * torch.cos(a)
            output[:, 2 * j + 1] = r * torch.sin(a)

        return output

    def quantize(self, x: Tensor) -> Tuple[List[Tensor], Tensor]:
        original_shape = x.shape
        x_flat = x.reshape(-1, self.head_dim).to(torch.float32)

        y = x_flat @ self.precondition_matrix.T.float()

        angles_per_level, final_radius = self._cartesian_to_polar(y)

        quantized_angles = [angles_per_level[0]]

        for level in range(2, self.num_levels + 1):
            angles = angles_per_level[level - 1]
            boundaries = self.level_boundaries[level]
            centroids = self.level_codebooks[level]

            inner_boundaries = boundaries[1:-1]
            indices = torch.bucketize(angles, inner_boundaries)
            indices = torch.clamp(indices, 0, len(centroids) - 1)
            quantized_angles.append(indices.to(torch.uint8))

        return quantized_angles, final_radius

    def dequantize(self, quantized_angles: List[Tensor], final_radius: Tensor) -> Tensor:
        reconstructed_angles = [quantized_angles[0]]

        for level in range(2, self.num_levels + 1):
            indices = quantized_angles[level - 1].long()
            centroids = self.level_codebooks[level]
            reconstructed_angles.append(centroids[indices])

        y_hat = self._polar_to_cartesian(reconstructed_angles, final_radius)

        x_hat = y_hat @ self.precondition_matrix.float()
        return x_hat.to(self.dtype)
