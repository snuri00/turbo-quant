import sys
import torch
import time

sys.path.insert(0, ".")
from turbo_quant import TurboQuantConfig, patch_model
from turbo_quant.config import _detect_device


def get_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return allocated, reserved, total
    return 0, 0, 0


def main():
    device = _detect_device()
    print(f"Device: {device}")

    if device == "cuda":
        alloc, res, total = get_gpu_memory()
        print(f"GPU VRAM: {total:.2f} GB total, {alloc:.2f} GB allocated")

    model_id = "mistralai/Mistral-7B-v0.1"
    print(f"\nLoading model: {model_id} (bitsandbytes NF4 quantized)")

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")

    if device == "cuda":
        alloc, res, total = get_gpu_memory()
        print(f"GPU after model load: {alloc:.2f} GB allocated, {total - alloc:.2f} GB free")

    config = TurboQuantConfig(
        mode="turbo_prod",
        key_bits=3,
        value_bits=4,
        num_outlier_channels=8,
        buffer_size=32,
    )
    print(f"\nApplying TurboQuant: mode={config.mode}, key_bits={config.key_bits}")
    print(f"Effective bits: {config.effective_bits:.2f}")
    patch_model(model, config)

    prompt = "Explain quantum computing in simple terms:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    print(f"\nPrompt: {prompt}")
    print(f"Input tokens: {inputs['input_ids'].shape[1]}")

    print("\nGenerating...")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
    gen_time = time.time() - t0

    new_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
    tokens_per_sec = new_tokens / gen_time

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n{'='*60}")
    print(response)
    print(f"{'='*60}")
    print(f"\nGenerated {new_tokens} tokens in {gen_time:.1f}s ({tokens_per_sec:.1f} tok/s)")

    if device == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1e9
        alloc, res, total = get_gpu_memory()
        print(f"Peak GPU memory: {peak:.2f} GB")
        print(f"Final GPU memory: {alloc:.2f} GB / {total:.2f} GB")


if __name__ == "__main__":
    main()
