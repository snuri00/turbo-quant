import math
import torch
from turbo_quant.quantizers.turbo_prod import TurboQuantProd


def test_quantize_dequantize():
    d = 128
    b = 3
    q = TurboQuantProd(d, b, device="cpu")
    x = torch.randn(4, 10, d)
    result = q.quantize(x)
    x_hat = q.dequantize(result)
    assert x_hat.shape == (4, 10, d)


def test_unbiasedness():
    d = 128
    n_trials = 30

    total_bias = 0.0
    for trial in range(n_trials):
        q = TurboQuantProd(d, 3, seed=trial * 7, device="cpu")
        keys = torch.randn(200, d)
        keys = keys / keys.norm(dim=-1, keepdim=True)
        query = torch.randn(1, d)

        exact_ip = (query @ keys.T).squeeze(0)

        result = q.quantize(keys)
        estimated_ip = q.estimate_inner_product(query, result).squeeze(0)

        bias = (estimated_ip - exact_ip).mean().item()
        total_bias += abs(bias)

    avg_bias = total_bias / n_trials
    assert avg_bias < 0.1, f"TurboQuant_prod average bias: {avg_bias}"


def test_compression_ratio():
    d = 128
    for b in [2, 3, 4]:
        q = TurboQuantProd(d, b, device="cpu")
        ratio = q.compression_ratio()
        assert ratio > 1.0, f"Compression ratio should be > 1, got {ratio}"


def test_inner_product_distortion():
    d = 128
    b = 3
    q = TurboQuantProd(d, b, device="cpu")

    keys = torch.randn(500, d)
    keys = keys / keys.norm(dim=-1, keepdim=True)
    query = torch.randn(1, d)

    exact_ip = (query @ keys.T).squeeze(0)
    result = q.quantize(keys)
    estimated_ip = q.estimate_inner_product(query, result).squeeze(0)

    distortion = ((estimated_ip - exact_ip) ** 2).mean().item()
    query_norm_sq = (query ** 2).sum().item()
    theoretical = math.sqrt(3) * math.pi ** 2 * query_norm_sq / (d * 4 ** b)

    assert distortion < theoretical * 5.0, (
        f"Distortion {distortion} exceeds 5x theoretical {theoretical}"
    )


if __name__ == "__main__":
    test_quantize_dequantize()
    print("PASS: quantize_dequantize")
    test_unbiasedness()
    print("PASS: unbiasedness")
    test_compression_ratio()
    print("PASS: compression_ratio")
    test_inner_product_distortion()
    print("PASS: inner_product_distortion")
    print("ALL TURBO_PROD TESTS PASSED")
