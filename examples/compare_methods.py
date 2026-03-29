import time
import torch
from turbo_quant.quantizers.turbo_mse import TurboQuantMSE
from turbo_quant.quantizers.turbo_prod import TurboQuantProd
from turbo_quant.quantizers.qjl import QJLQuantizer


def benchmark_method(name, quantize_fn, score_fn, keys, query, n_runs=100):
    quantize_fn()
    torch.cuda.synchronize() if keys.is_cuda else None

    start = time.perf_counter()
    for _ in range(n_runs):
        result = quantize_fn()
    torch.cuda.synchronize() if keys.is_cuda else None
    quant_time = (time.perf_counter() - start) / n_runs

    start = time.perf_counter()
    for _ in range(n_runs):
        scores = score_fn(result)
    torch.cuda.synchronize() if keys.is_cuda else None
    score_time = (time.perf_counter() - start) / n_runs

    exact_ip = (query @ keys.T).squeeze(0)
    estimated = score_fn(result).squeeze(0)
    mse = ((exact_ip - estimated) ** 2).mean().item()
    bias = (estimated - exact_ip).mean().item()

    print(f"{name:20s} | quant: {quant_time*1000:8.3f}ms | score: {score_time*1000:8.3f}ms | MSE: {mse:.6f} | bias: {bias:+.6f}")


def main():
    d = 128
    seq_len = 1024
    device = "cpu"

    keys = torch.randn(seq_len, d, device=device)
    keys = keys / keys.norm(dim=-1, keepdim=True)
    query = torch.randn(1, d, device=device)

    print(f"head_dim={d}, seq_len={seq_len}, device={device}")
    print("-" * 90)

    for b in [2, 3, 4]:
        print(f"\n--- bit-width = {b} ---")

        mse_q = TurboQuantMSE(d, b, device=device)
        benchmark_method(
            f"TurboQuant_mse(b={b})",
            lambda: mse_q.quantize(keys),
            lambda r: mse_q.compute_rotated_score(query, r[0], r[1]),
            keys, query, n_runs=20,
        )

        prod_q = TurboQuantProd(d, b, device=device)
        benchmark_method(
            f"TurboQuant_prod(b={b})",
            lambda: prod_q.quantize(keys),
            lambda r: prod_q.estimate_inner_product(query, r),
            keys, query, n_runs=20,
        )

    qjl = QJLQuantizer(d, d, device=device)
    benchmark_method(
        "QJL (1-bit)",
        lambda: qjl.quantize(keys),
        lambda r: qjl.estimate_inner_product(query, r[0], r[1]),
        keys, query, n_runs=20,
    )


if __name__ == "__main__":
    main()
