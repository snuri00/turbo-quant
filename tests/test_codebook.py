import math
import torch
import numpy as np
from turbo_quant.quantizers.codebook import (
    lloyd_max_solve, get_optimal_codebook, compute_mse_cost, beta_pdf
)


def test_beta_pdf_normalizes():
    from scipy.integrate import quad
    for d in [16, 64, 128, 256]:
        integral, _ = quad(lambda x: beta_pdf(np.array([x]), d)[0], -1, 1)
        assert abs(integral - 1.0) < 1e-6, f"d={d}: integral={integral}"


def test_codebook_symmetry():
    for d in [64, 128]:
        for b in [1, 2, 3]:
            centroids, boundaries = lloyd_max_solve(d, b)
            for i in range(len(centroids) // 2):
                assert abs(centroids[i] + centroids[-(i + 1)]) < 1e-6


def test_mse_bounds():
    expected = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}
    d = 128
    for b, expected_mse in expected.items():
        centroids, _ = lloyd_max_solve(d, b)
        cost = compute_mse_cost(centroids, d)
        actual_mse = d * cost
        assert abs(actual_mse - expected_mse) < 0.05, (
            f"b={b}: expected ~{expected_mse}, got {actual_mse}"
        )


def test_get_optimal_codebook_shape():
    for b in [1, 2, 3, 4]:
        centroids, boundaries = get_optimal_codebook(128, b)
        assert centroids.shape[0] == 2 ** b
        assert boundaries.shape[0] == 2 ** b + 1


def test_codebook_sorted():
    for b in [1, 2, 3, 4]:
        centroids, boundaries = get_optimal_codebook(128, b)
        for i in range(len(centroids) - 1):
            assert centroids[i] < centroids[i + 1]


if __name__ == "__main__":
    test_beta_pdf_normalizes()
    print("PASS: beta_pdf_normalizes")
    test_codebook_symmetry()
    print("PASS: codebook_symmetry")
    test_mse_bounds()
    print("PASS: mse_bounds")
    test_get_optimal_codebook_shape()
    print("PASS: codebook_shape")
    test_codebook_sorted()
    print("PASS: codebook_sorted")
    print("ALL CODEBOOK TESTS PASSED")
