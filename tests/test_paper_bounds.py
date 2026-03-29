import math
import torch
from turbo_quant.quantizers.turbo_mse import TurboQuantMSE
from turbo_quant.quantizers.turbo_prod import TurboQuantProd
from turbo_quant.quantizers.qjl import QJLQuantizer


def test_mse_distortion_vs_paper():
    d = 128
    n_samples = 5000

    paper_mse = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}
    paper_upper = {b: math.sqrt(3 * math.pi) / 2 * (1 / 4**b) for b in range(1, 5)}
    paper_lower = {b: 1 / 4**b for b in range(1, 5)}

    print("=" * 70)
    print("PAPER Theorem 1: Dmse = E[||x - x_hat||^2]  (unit norm vectors)")
    print(f"{'bit':>4} | {'Lower Bound':>12} | {'Paper Exact':>12} | {'Upper Bound':>12} | {'OURS':>12} | {'Status':>10}")
    print("-" * 70)

    for b in [1, 2, 3, 4]:
        q = TurboQuantMSE(d, b, device="cpu", dtype=torch.float32)
        x = torch.randn(n_samples, d)
        x = x / x.norm(dim=-1, keepdim=True)

        indices, norms = q.quantize(x)
        x_hat = q.dequantize(indices, norms)

        our_mse = ((x - x_hat) ** 2).sum(dim=-1).mean().item()

        lb = paper_lower[b]
        ub = paper_upper[b]
        exact = paper_mse[b]

        within = "OK" if our_mse <= ub * 1.1 else "FAIL"

        print(f"{b:>4} | {lb:>12.6f} | {exact:>12.6f} | {ub:>12.6f} | {our_mse:>12.6f} | {within:>10}")

    print()


def test_inner_product_distortion_vs_paper():
    d = 128
    n_keys = 3000

    paper_dprod_exact = {1: 1.57, 2: 0.56, 3: 0.18, 4: 0.047}
    paper_dprod_upper = {b: math.sqrt(3) * math.pi**2 / (d * 4**b) for b in range(1, 5)}
    paper_dprod_lower = {b: 1.0 / (d * 4**b) for b in range(1, 5)}

    print("=" * 70)
    print("PAPER Theorem 2: Dprod = E[(⟨y,x⟩ - ⟨y,x_hat⟩)^2]")
    print("                 (x unit norm, y unit norm)")
    print(f"{'bit':>4} | {'Lower':>12} | {'Paper /d':>12} | {'Upper':>12} | {'OURS':>12} | {'Ratio':>8}")
    print("-" * 70)

    for b in [1, 2, 3, 4]:
        q = TurboQuantProd(d, b, device="cpu", dtype=torch.float32)

        keys = torch.randn(n_keys, d)
        keys = keys / keys.norm(dim=-1, keepdim=True)
        query = torch.randn(1, d)
        query = query / query.norm(dim=-1, keepdim=True)

        result = q.quantize(keys)
        estimated_ip = q.estimate_inner_product(query, result).squeeze(0)
        exact_ip = (query @ keys.T).squeeze(0)

        our_dprod = ((exact_ip - estimated_ip) ** 2).mean().item()

        lb = paper_dprod_lower[b]
        ub = paper_dprod_upper[b]
        paper_val = paper_dprod_exact[b] / d

        ratio = our_dprod / paper_val if paper_val > 0 else float('inf')

        print(f"{b:>4} | {lb:>12.8f} | {paper_val:>12.8f} | {ub:>12.8f} | {our_dprod:>12.8f} | {ratio:>8.2f}x")

    print()


def test_qjl_variance_vs_paper():
    d = 128
    n_keys = 5000

    print("=" * 70)
    print("PAPER Lemma 4 (QJL): Var ≤ pi/(2d) * ||y||^2")
    print(f"{'||y||^2':>10} | {'Theoretical':>12} | {'Measured':>12} | {'Ratio':>8}")
    print("-" * 70)

    for query_scale in [0.5, 1.0, 2.0, 5.0]:
        qjl = QJLQuantizer(d, d, device="cpu")

        keys = torch.randn(n_keys, d)
        keys = keys / keys.norm(dim=-1, keepdim=True)
        query = torch.randn(1, d) * query_scale

        y_norm_sq = (query ** 2).sum().item()

        bits, norms = qjl.quantize(keys)
        estimated = qjl.estimate_inner_product(query, bits, norms).squeeze(0)
        exact = (query @ keys.T).squeeze(0)

        measured_var = ((estimated - exact) ** 2).mean().item()
        theoretical = math.pi / (2 * d) * y_norm_sq

        ratio = measured_var / theoretical

        print(f"{y_norm_sq:>10.4f} | {theoretical:>12.8f} | {measured_var:>12.8f} | {ratio:>8.3f}x")

    print()


def test_unbiasedness_prod():
    d = 128
    n_keys = 5000

    print("=" * 70)
    print("PAPER Theorem 2: E[⟨y, x_hat⟩] = ⟨y, x⟩  (unbiased)")
    print(f"{'bit':>4} | {'Mean Exact IP':>14} | {'Mean Est IP':>14} | {'Bias':>12} | {'Rel Bias %':>12}")
    print("-" * 70)

    for b in [1, 2, 3, 4]:
        q = TurboQuantProd(d, b, device="cpu")

        keys = torch.randn(n_keys, d)
        keys = keys / keys.norm(dim=-1, keepdim=True)
        query = torch.randn(1, d)
        query = query / query.norm(dim=-1, keepdim=True)

        exact = (query @ keys.T).squeeze(0)
        result = q.quantize(keys)
        estimated = q.estimate_inner_product(query, result).squeeze(0)

        mean_exact = exact.mean().item()
        mean_est = estimated.mean().item()
        bias = mean_est - mean_exact
        rel_bias = abs(bias) / (abs(mean_exact) + 1e-10) * 100

        print(f"{b:>4} | {mean_exact:>14.8f} | {mean_est:>14.8f} | {bias:>+12.8f} | {rel_bias:>11.4f}%")

    print()


def test_mse_bias_at_1bit():
    d = 128
    n_keys = 5000

    print("=" * 70)
    print("PAPER Section 3.2: MSE quantizer bias = 2/pi ≈ 0.6366 at b=1")

    q = TurboQuantMSE(d, 1, device="cpu")
    keys = torch.randn(n_keys, d)
    keys = keys / keys.norm(dim=-1, keepdim=True)
    query = torch.randn(1, d)
    query = query / query.norm(dim=-1, keepdim=True)

    indices, norms = q.quantize(keys)
    keys_hat = q.dequantize(indices, norms)

    exact = (query @ keys.T).squeeze(0)
    estimated = (query @ keys_hat.T).squeeze(0)

    mask = exact.abs() > 0.01
    if mask.sum() > 100:
        ratio = (estimated[mask] / exact[mask]).mean().item()
        theoretical = 2.0 / math.pi
        print(f"  Measured E[est/exact]:  {ratio:.6f}")
        print(f"  Paper theoretical:     {theoretical:.6f} (2/pi)")
        print(f"  Difference:            {abs(ratio - theoretical):.6f}")
    print()


def main():
    print("\n" + "=" * 70)
    print("  TURBOQUANT - PAPER BOUNDS VALIDATION")
    print("  Comparing our implementation against arXiv:2504.19874")
    print("=" * 70 + "\n")

    test_mse_distortion_vs_paper()
    test_inner_product_distortion_vs_paper()
    test_qjl_variance_vs_paper()
    test_unbiasedness_prod()
    test_mse_bias_at_1bit()

    print("=" * 70)
    print("  VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
