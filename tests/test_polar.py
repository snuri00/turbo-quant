import torch
from turbo_quant.quantizers.polar import PolarQuantizer


def test_polar_roundtrip():
    d = 128
    p = PolarQuantizer(d, bits=3, device="cpu")
    x = torch.randn(10, d)
    y = x @ p.precondition_matrix.T.float()
    angles, radius = p._cartesian_to_polar(y)
    y_hat = p._polar_to_cartesian(angles, radius)
    assert torch.allclose(y, y_hat, atol=1e-4), "Polar roundtrip failed"


def test_quantize_dequantize():
    d = 128
    p = PolarQuantizer(d, bits=3, device="cpu")
    x = torch.randn(5, d)
    quant_angles, radius = p.quantize(x)
    x_hat = p.dequantize(quant_angles, radius)
    assert x_hat.shape == (5, d)


def test_reconstruction_error():
    d = 128
    p = PolarQuantizer(d, bits=3, device="cpu")
    x = torch.randn(100, d)
    x = x / x.norm(dim=-1, keepdim=True)
    quant_angles, radius = p.quantize(x)
    x_hat = p.dequantize(quant_angles, radius)
    mse = ((x - x_hat) ** 2).sum(dim=-1).mean().item()
    assert mse < 1.0, f"PolarQuant MSE too high: {mse}"


if __name__ == "__main__":
    test_polar_roundtrip()
    print("PASS: polar_roundtrip")
    test_quantize_dequantize()
    print("PASS: quantize_dequantize")
    test_reconstruction_error()
    print("PASS: reconstruction_error")
    print("ALL POLAR TESTS PASSED")
