import torch
from turbo_quant.config import TurboQuantConfig
from turbo_quant.cache.turbo_cache import TurboQuantCache
from turbo_quant.cache.outlier_detector import OutlierDetector


def test_outlier_detector():
    d = 128
    det = OutlierDetector(num_outlier_channels=32)
    keys = torch.randn(1, 8, 100, d)
    keys[:, :, :, 10] *= 100
    keys[:, :, :, 50] *= 100

    indices = det.detect(keys)
    assert 10 in indices
    assert 50 in indices
    assert len(indices) == 32


def test_split_merge():
    d = 128
    det = OutlierDetector(num_outlier_channels=32)
    x = torch.randn(2, 8, 10, d)
    outlier_indices = torch.arange(0, 32)

    regular, outlier = det.split_channels(x, outlier_indices)
    assert regular.shape[-1] == 96
    assert outlier.shape[-1] == 32

    merged = det.merge_channels(regular, outlier, outlier_indices, d)
    assert merged.shape == x.shape
    assert torch.allclose(merged, x, atol=1e-6)


def test_cache_update():
    config = TurboQuantConfig(
        mode="turbo_mse", key_bits=2, value_bits=4,
        buffer_size=8, num_outlier_channels=16,
        device="cpu", dtype=torch.float32
    )
    cache = TurboQuantCache(config, num_layers=2, num_heads=4, head_dim=64)

    keys = torch.randn(1, 4, 20, 64)
    values = torch.randn(1, 4, 20, 64)
    cache.update(0, keys, values)

    assert cache.get_seq_len(0) == 20


def test_cache_scores():
    config = TurboQuantConfig(
        mode="turbo_mse", key_bits=2, value_bits=4,
        buffer_size=4, num_outlier_channels=8,
        device="cpu", dtype=torch.float32
    )
    cache = TurboQuantCache(config, num_layers=1, num_heads=2, head_dim=32)

    keys = torch.randn(1, 2, 10, 32)
    values = torch.randn(1, 2, 10, 32)
    cache.update(0, keys, values)

    query = torch.randn(1, 2, 1, 32)
    scores = cache.get_attention_scores(0, query)
    assert scores.shape[-1] == 10


if __name__ == "__main__":
    test_outlier_detector()
    print("PASS: outlier_detector")
    test_split_merge()
    print("PASS: split_merge")
    test_cache_update()
    print("PASS: cache_update")
    test_cache_scores()
    print("PASS: cache_scores")
    print("ALL CACHE TESTS PASSED")
