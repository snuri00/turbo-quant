import torch
from turbo_quant.quantizers.turbo_mse import TurboQuantMSE
from turbo_quant.quantizers.turbo_prod import TurboQuantProd
from turbo_quant.quantizers.qjl import QJLQuantizer
from turbo_quant.quantizers.value_quant import ValueQuantizer


def test_full_pipeline_turbo_prod():
    d = 128
    seq_len = 64
    b = 3

    quantizer = TurboQuantProd(d, b, device="cpu")
    value_quant = ValueQuantizer(bits=4, device="cpu")

    keys = torch.randn(seq_len, d)
    values = torch.randn(seq_len, d)
    query = torch.randn(1, d)

    key_result = quantizer.quantize(keys)
    v_quant, v_scale, v_zp = value_quant.quantize(values)

    scores = quantizer.estimate_inner_product(query, key_result)
    scores = scores / (d ** 0.5)
    attn_weights = torch.softmax(scores, dim=-1)

    values_hat = value_quant.dequantize(v_quant, v_scale, v_zp)
    output = attn_weights @ values_hat

    assert output.shape == (1, d)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_full_pipeline_exact_comparison():
    d = 128
    seq_len = 32
    b = 4

    keys = torch.randn(seq_len, d)
    keys = keys / keys.norm(dim=-1, keepdim=True)
    query = torch.randn(1, d)

    exact_scores = (query @ keys.T) / (d ** 0.5)
    exact_attn = torch.softmax(exact_scores, dim=-1)

    quantizer = TurboQuantProd(d, b, device="cpu")
    key_result = quantizer.quantize(keys)
    quant_scores = quantizer.estimate_inner_product(query, key_result) / (d ** 0.5)
    quant_attn = torch.softmax(quant_scores, dim=-1)

    score_error = (exact_scores - quant_scores).abs().mean().item()
    attn_error = (exact_attn - quant_attn).abs().mean().item()

    assert score_error < 0.5, f"Score error too high: {score_error}"
    assert attn_error < 0.1, f"Attention error too high: {attn_error}"


def test_memory_savings():
    d = 128
    seq_len = 1000

    original_bytes = seq_len * d * 2

    for b in [2, 3, 4]:
        q = TurboQuantProd(d, b, device="cpu")
        ratio = q.compression_ratio()
        compressed_bytes = original_bytes / ratio
        savings = 1.0 - compressed_bytes / original_bytes

        assert savings > 0.5, f"b={b}: savings={savings:.1%}, expected > 50%"


if __name__ == "__main__":
    test_full_pipeline_turbo_prod()
    print("PASS: full_pipeline_turbo_prod")
    test_full_pipeline_exact_comparison()
    print("PASS: exact_comparison")
    test_memory_savings()
    print("PASS: memory_savings")
    print("ALL E2E TESTS PASSED")
