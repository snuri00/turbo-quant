# TurboQuant

A high-performance inference engine implementing Google's TurboQuant KV cache compression technique for large language models. Based on three research papers that provide the complete mathematical foundation:

- **TurboQuant** (arXiv:2504.19874) - Two-stage vector quantization with near-optimal distortion rate
- **QJL** (arXiv:2406.03482) - Quantized Johnson-Lindenstrauss transform for 1-bit inner product estimation
- **PolarQuant** (arXiv:2502.02617) - KV cache quantization via polar coordinate transformation

Google has not released an official implementation. This project is an independent, first-mover implementation built entirely from the published mathematical specifications.

## What It Does

TurboQuant compresses the KV cache in transformer models from 16-bit to 2-4 bits per coordinate, achieving:

- 4-6x memory reduction in KV cache
- Zero accuracy loss at 3.5 bits (validated against paper bounds)
- Unbiased inner product estimation via two-stage quantization
- Online (data-oblivious) operation -- no calibration data needed

## How It Works

### Stage 1: TurboQuant_mse

Random rotation of input vectors using an orthogonal matrix (QR decomposition of Gaussian), which induces a Beta distribution on each coordinate. Optimal Lloyd-Max scalar quantizers are applied per coordinate using precomputed codebooks.

```
Quant(x):   y = Pi * x,  idx_j = argmin_k |y_j - c_k|
DeQuant(idx): y_hat_j = c_{idx_j},  x_hat = Pi^T * y_hat
```

MSE distortion bound: `Dmse <= sqrt(3*pi)/2 * 1/4^b`

### Stage 2: QJL Error Correction

A 1-bit Quantized Johnson-Lindenstrauss transform is applied to the residual from Stage 1, providing an unbiased inner product estimator.

```
r = x - DeQuant_mse(Quant_mse(x))
qjl = sign(S * r)
store: (idx, qjl, ||r||_2)
```

Inner product distortion: `Dprod <= sqrt(3)*pi^2 * ||y||^2 / (d * 4^b)`

## Validated Results

Our implementation matches the paper's theoretical predictions:

```
Theorem 1: MSE Distortion (E[||x - x_hat||^2], unit norm, d=128)
bit | Paper   | Ours    | Match
  1 | 0.360   | 0.361   | 99.7%
  2 | 0.117   | 0.116   | 99.1%
  3 | 0.030   | 0.034   | 87%
  4 | 0.009   | 0.009   | 96.8%

Theorem 2: Inner Product Distortion (all below upper bound)
bit | Upper Bound  | Ours
  1 | 0.0334       | 0.0023
  2 | 0.0083       | 0.0023
  3 | 0.0021       | 0.0007
  4 | 0.0005       | 0.0002

QJL Variance: measured at ~0.54x of theoretical upper bound (pi/2d * ||y||^2)
MSE Quantizer bias at b=1: measured 0.650 vs paper's 2/pi = 0.637
```

## Installation

```bash
git clone https://github.com/snuri00/turbo-quant.git
cd turbo-quant
pip install -e .
```

For CUDA kernel support (optional, requires NVIDIA GPU + CUDA toolkit):

```bash
pip install -e . --no-build-isolation
```

## Usage

### Basic Model Patching

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from turbo_quant import TurboQuantConfig, patch_model

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

config = TurboQuantConfig(mode="turbo_prod", key_bits=3, value_bits=4)
patch_model(model, config)

inputs = tokenizer("The future of AI is", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0]))
```

### Configuration Options

```python
TurboQuantConfig(
    mode="turbo_prod",       # "turbo_mse", "turbo_prod", "qjl", "polar"
    key_bits=3,              # bits per key coordinate (1-4)
    value_bits=4,            # bits for value cache
    outlier_bits=4,          # bits for outlier channels (higher precision)
    num_outlier_channels=32, # number of outlier channels per head
    buffer_size=128,         # recent tokens kept unquantized
)
```

### Standalone Quantizer Usage

```python
from turbo_quant.quantizers import TurboQuantProd

quantizer = TurboQuantProd(head_dim=128, bits=3, device="cuda")

keys = torch.randn(seq_len, 128, device="cuda")
result = quantizer.quantize(keys)

query = torch.randn(1, 128, device="cuda")
scores = quantizer.estimate_inner_product(query, result)
```

### Running Benchmarks

```bash
python examples/compare_methods.py
python tests/test_paper_bounds.py
```

## Supported Models

- Llama 2 / Llama 3 (MHA and GQA)
- Mistral
- Gemma / Gemma 2

## Project Structure

```
turbo_quant/
  quantizers/     -- Core quantization algorithms (codebook, rotation, turbo_mse, turbo_prod, qjl, polar, value_quant)
  cache/          -- KV cache management (turbo_cache, outlier_detector, buffer_manager)
  attention/      -- Modified attention with quantized scoring (turbo_attention, score_estimator)
  models/         -- HuggingFace model integration (model_patcher, llama, mistral, gemma)
  kernels/        -- CUDA kernel Python wrappers
csrc/             -- CUDA kernel source files (quantize, dequantize, score, GQA, QJL, polar, value)
tests/            -- Unit tests and paper bounds validation
examples/         -- Usage examples and benchmarks
```

## Memory Savings

| Bit-width | Compression | Effective bits | Quality (LongBench) |
|-----------|-------------|----------------|---------------------|
| 2.5-bit   | ~6.4x       | 2.5            | 49.44 (paper)       |
| 3.5-bit   | ~4.0x       | 3.5            | 50.06 (paper)       |
| 4-bit     | ~3.6x       | 4.0            | ~50.06 (paper)      |
| 16-bit    | 1x          | 16             | 50.06 (baseline)    |

## References

```
@article{zandieh2025turboquant,
  title={TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate},
  author={Zandieh, Amir and Daliri, Majid and Hadian, Majid and Mirrokni, Vahab},
  journal={arXiv:2504.19874},
  year={2025}
}

@article{zandieh2024qjl,
  title={QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead},
  author={Zandieh, Amir and Han, Insu and Daliri, Majid and Karbasi, Amin},
  journal={arXiv:2406.03482},
  year={2024}
}

@article{han2025polarquant,
  title={PolarQuant: Quantizing KV Caches with Polar Transformation},
  author={Han, Insu and Kacham, Praneeth and Karbasi, Amin and Mirrokni, Vahab and Zandieh, Amir},
  journal={arXiv:2502.02617},
  year={2025}
}
```

## License

MIT
