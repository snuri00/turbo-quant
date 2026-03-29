import torch
from turbo_quant.quantizers.turbo_mse import TurboQuantMSE


def test_quantize_dequantize_shape():
    d = 128
    for b in [1, 2, 3, 4]:
        q = TurboQuantMSE(d, b, device="cpu", dtype=torch.float32)
        x = torch.randn(4, 10, d)
        x = x / x.norm(dim=-1, keepdim=True)
        indices, norms = q.quantize(x)
        assert indices.shape == (4, 10, d)
        x_hat = q.dequantize(indices, norms)
        assert x_hat.shape == (4, 10, d)


def test_mse_distortion():
    d = 128
    expected = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}

    for b, expected_mse in expected.items():
        q = TurboQuantMSE(d, b, device="cpu", dtype=torch.float32)
        n_samples = 2000
        x = torch.randn(n_samples, d)
        x = x / x.norm(dim=-1, keepdim=True)

        indices, norms = q.quantize(x)
        x_hat = q.dequantize(indices, norms)

        mse = ((x - x_hat) ** 2).sum(dim=-1).mean().item()
        assert mse < expected_mse * 2.0, f"b={b}: MSE={mse}, expected < {expected_mse * 2.0}"


def test_pack_unpack_roundtrip():
    d = 128
    for b in [1, 2, 3, 4]:
        q = TurboQuantMSE(d, b, device="cpu", dtype=torch.float32)
        x = torch.randn(8, d)
        x = x / x.norm(dim=-1, keepdim=True)

        packed, norms = q.quantize_and_pack(x)
        x_hat = q.unpack_and_dequantize(packed, norms, 8)

        indices, norms2 = q.quantize(x)
        x_hat2 = q.dequantize(indices, norms2)

        assert torch.allclose(x_hat, x_hat2, atol=1e-5), f"b={b}: pack/unpack mismatch"


def test_rotated_score():
    d = 128
    b = 3
    q = TurboQuantMSE(d, b, device="cpu", dtype=torch.float32)

    query = torch.randn(2, 1, d)
    keys = torch.randn(2, 10, d)

    indices, norms = q.quantize(keys)
    keys_hat = q.dequantize(indices, norms)

    exact_scores = torch.bmm(query, keys_hat.transpose(-1, -2)).squeeze(1)
    estimated_scores = q.compute_rotated_score(query.squeeze(1), indices, norms)

    assert torch.allclose(exact_scores, estimated_scores, atol=1e-3)


if __name__ == "__main__":
    test_quantize_dequantize_shape()
    print("PASS: shape")
    test_mse_distortion()
    print("PASS: mse_distortion")
    test_pack_unpack_roundtrip()
    print("PASS: pack_unpack")
    test_rotated_score()
    print("PASS: rotated_score")
    print("ALL TURBO_MSE TESTS PASSED")
