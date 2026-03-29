import sys
import os
import time

sys.path.insert(0, ".")

from huggingface_hub import hf_hub_download
from turbo_quant.config import TurboQuantConfig
from turbo_quant.models.llamacpp import TurboQuantLlamaCpp


def download_model():
    model_id = "leafspark/Meta-Llama-3.1-8B-Instruct-Q2_K-GGUF"
    filename = "meta-llama-3.1-8b-instruct-q2_k.gguf"

    print(f"Downloading {model_id}...")
    path = hf_hub_download(repo_id=model_id, filename=filename)
    size_gb = os.path.getsize(path) / 1e9
    print(f"Downloaded: {path} ({size_gb:.2f} GB)")
    return path


def main():
    model_path = download_model()

    config = TurboQuantConfig(
        mode="turbo_prod",
        key_bits=3,
        value_bits=4,
        num_outlier_channels=8,
        buffer_size=32,
        device="cpu",
    )

    print("\nLoading model...")
    t0 = time.time()
    engine = TurboQuantLlamaCpp(
        model_path=model_path,
        config=config,
        n_ctx=2048,
        n_gpu_layers=-1,
        verbose=False,
    )
    print(f"Loaded in {time.time() - t0:.1f}s")
    engine.info()

    print("\n" + "=" * 60)
    print("GENERATION TEST")
    print("=" * 60)

    prompt = "Explain quantum computing in simple terms:"
    print(f"\nPrompt: {prompt}\n")

    t0 = time.time()
    response = engine.generate_text(
        prompt,
        max_tokens=200,
        temperature=0.7,
        stream=True,
    )
    gen_time = time.time() - t0
    tokens = len(engine.llm.tokenize(response.encode()))
    print(f"\n[{tokens} tokens in {gen_time:.1f}s = {tokens/gen_time:.1f} tok/s]")

    print("\n" + "=" * 60)
    print("KV CACHE MEMORY ANALYSIS")
    print("=" * 60)
    engine.benchmark_kv_cache()

    print("\n" + "=" * 60)
    print("TURBOQUANT QUANTIZATION QUALITY")
    print("=" * 60)
    engine.quantize_benchmark()


if __name__ == "__main__":
    main()
