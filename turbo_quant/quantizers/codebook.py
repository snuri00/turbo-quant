import math
from typing import Dict, Tuple

import torch
from torch import Tensor
from scipy.special import gamma as gamma_fn
from scipy.integrate import quad
import numpy as np


_CODEBOOK_CACHE: Dict[Tuple[int, int], Tuple[Tensor, Tensor]] = {}


def beta_pdf(x: np.ndarray, d: int) -> np.ndarray:
    coeff = gamma_fn(d / 2) / (math.sqrt(math.pi) * gamma_fn((d - 1) / 2))
    return coeff * np.power(np.maximum(1 - x ** 2, 0), (d - 3) / 2)


def compute_boundary(c_left: float, c_right: float) -> float:
    return (c_left + c_right) / 2


def compute_centroid(left: float, right: float, d: int) -> float:
    numerator, _ = quad(lambda x: x * beta_pdf(np.array([x]), d)[0], left, right)
    denominator, _ = quad(lambda x: beta_pdf(np.array([x]), d)[0], left, right)
    if abs(denominator) < 1e-15:
        return (left + right) / 2
    return numerator / denominator


def compute_mse_cost(centroids: np.ndarray, d: int) -> float:
    k = len(centroids)
    boundaries = np.zeros(k + 1)
    boundaries[0] = -1.0
    boundaries[-1] = 1.0
    for i in range(k - 1):
        boundaries[i + 1] = compute_boundary(centroids[i], centroids[i + 1])

    cost = 0.0
    for i in range(k):
        c = centroids[i]
        val, _ = quad(
            lambda x: (x - c) ** 2 * beta_pdf(np.array([x]), d)[0],
            boundaries[i], boundaries[i + 1]
        )
        cost += val
    return cost


def lloyd_max_solve(d: int, b: int, max_iter: int = 2000, tol: float = 1e-14) -> Tuple[np.ndarray, np.ndarray]:
    k = 2 ** b
    centroids = np.linspace(-0.95, 0.95, k)

    for _ in range(max_iter):
        boundaries = np.zeros(k + 1)
        boundaries[0] = -1.0
        boundaries[-1] = 1.0
        for i in range(k - 1):
            boundaries[i + 1] = compute_boundary(centroids[i], centroids[i + 1])

        new_centroids = np.zeros(k)
        for i in range(k):
            new_centroids[i] = compute_centroid(boundaries[i], boundaries[i + 1], d)

        if np.max(np.abs(new_centroids - centroids)) < tol:
            centroids = new_centroids
            break
        centroids = new_centroids

    boundaries = np.zeros(k + 1)
    boundaries[0] = -1.0
    boundaries[-1] = 1.0
    for i in range(k - 1):
        boundaries[i + 1] = compute_boundary(centroids[i], centroids[i + 1])

    return centroids, boundaries


def get_codebook(d: int, b: int, device: str = "cpu") -> Tuple[Tensor, Tensor]:
    key = (d, b)
    if key in _CODEBOOK_CACHE:
        cached_c, cached_b = _CODEBOOK_CACHE[key]
        return cached_c.to(device), cached_b.to(device)

    centroids_np, boundaries_np = lloyd_max_solve(d, b)
    centroids = torch.from_numpy(centroids_np).float()
    boundaries = torch.from_numpy(boundaries_np).float()

    _CODEBOOK_CACHE[key] = (centroids.cpu(), boundaries.cpu())
    return centroids.to(device), boundaries.to(device)


def get_gaussian_codebook(d: int, b: int, device: str = "cpu") -> Tuple[Tensor, Tensor]:
    scale = 1.0 / math.sqrt(d)

    if b == 1:
        c = math.sqrt(2.0 / math.pi) * scale
        centroids = torch.tensor([-c, c], device=device)
    elif b == 2:
        centroids = torch.tensor([-1.51 * scale, -0.453 * scale, 0.453 * scale, 1.51 * scale], device=device)
    elif b == 3:
        centroids_np, _ = lloyd_max_solve(max(d, 64), b)
        centroids = torch.from_numpy(centroids_np).float().to(device)
    elif b == 4:
        centroids_np, _ = lloyd_max_solve(max(d, 64), b)
        centroids = torch.from_numpy(centroids_np).float().to(device)
    else:
        centroids_np, _ = lloyd_max_solve(max(d, 64), b)
        centroids = torch.from_numpy(centroids_np).float().to(device)

    k = len(centroids)
    boundaries = torch.zeros(k + 1, device=device)
    boundaries[0] = -float('inf')
    boundaries[-1] = float('inf')
    for i in range(k - 1):
        boundaries[i + 1] = (centroids[i] + centroids[i + 1]) / 2

    return centroids, boundaries


def get_optimal_codebook(d: int, b: int, device: str = "cpu") -> Tuple[Tensor, Tensor]:
    if d >= 64:
        return get_gaussian_codebook(d, b, device)
    return get_codebook(d, b, device)
