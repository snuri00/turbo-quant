import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from turbo_quant import TurboQuantConfig, patch_model
from turbo_quant.config import _detect_device, _detect_dtype


def main():
    model_name = "meta-llama/Llama-2-7b-hf"
    device = _detect_device()
    model_dtype = _detect_dtype(device)

    print(f"Loading model: {model_name}")
    print(f"Device: {device}, dtype: {model_dtype}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=model_dtype,
        device_map="auto" if device == "cuda" else None,
    )
    if device != "cuda":
        model = model.to(device)

    config = TurboQuantConfig(
        mode="turbo_prod",
        key_bits=3,
        value_bits=4,
        num_outlier_channels=32,
        buffer_size=128,
    )

    print(f"Patching model with TurboQuant (mode={config.mode}, key_bits={config.key_bits})")
    print(f"Effective bits per key coordinate: {config.effective_bits:.2f}")
    patch_model(model, config)

    prompt = "The future of artificial intelligence is"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print(f"\nPrompt: {prompt}")
    print("Generating...")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nGenerated:\n{response}")


if __name__ == "__main__":
    main()
