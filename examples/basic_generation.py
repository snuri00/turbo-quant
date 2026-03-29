import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from turbo_quant import TurboQuantConfig, patch_model


def main():
    model_name = "meta-llama/Llama-2-7b-hf"

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

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
