import math
import torch
from turbo_quant.quantizers.qjl import QJLQuantizer


def test_quantize_shape():
    d = 128
    q = QJLQuantizer(d, d, device="cpu")
    x = torch.randn(4, 10, d)
    bits, norms = q.quantize(x)
    assert norms.shape == (4, 10)
    assert bits.shape[-1] == math.ceil(d / 8)


def test_unbiasedness():
    d = 128
    n_trials = 50
    n_samples = 500

    total_bias = 0.0
    for _ in range(n_trials):
        seed = torch.randint(0, 100000, (1,)).item()
        q = QJLQuantizer(d, d, seed=seed, device="cpu")

        keys = torch.randn(n_samples, d)
        keys = keys / keys.norm(dim=-1, keepdim=True)
        query = torch.randn(1, d)

        exact_ip = (query @ keys.T).squeeze(0)

        bits, norms = q.quantize(keys)
        estimated_ip = q.estimate_inner_product(query, bits, norms).squeeze(0)

        bias = (estimated_ip - exact_ip).mean().item()
        total_bias += abs(bias)

    avg_bias = total_bias / n_trials
    assert avg_bias < 0.05, f"Average bias too high: {avg_bias}"


def test_dequantize():
    d = 128
    q = QJLQuantizer(d, d, device="cpu")
    x = torch.randn(10, d)
    bits, norms = q.quantize(x)
    x_hat = q.dequantize(bits, norms)
    assert x_hat.shape == (10, d)


def test_variance_bound():
    d = 128
    q = QJLQuantizer(d, d, device="cpu")

    keys = torch.randn(1000, d)
    keys = keys / keys.norm(dim=-1, keepdim=True)
    query = torch.randn(1, d)
    query_norm_sq = (query ** 2).sum().item()

    exact_ip = (query @ keys.T).squeeze(0)
    bits, norms = q.quantize(keys)
    estimated_ip = q.estimate_inner_product(query, bits, norms).squeeze(0)

    variance = ((estimated_ip - exact_ip) ** 2).mean().item()
    theoretical_bound = math.pi / (2 * d) * query_norm_sq

    assert variance < theoretical_bound * 3.0, (
        f"Variance {variance} exceeds 3x theoretical bound {theoretical_bound}"
    )


if __name__ == "__main__":
    test_quantize_shape()
    print("PASS: shape")
    test_unbiasedness()
    print("PASS: unbiasedness")
    test_dequantize()
    print("PASS: dequantize")
    test_variance_bound()
    print("PASS: variance_bound")
    print("ALL QJL TESTS PASSED")
