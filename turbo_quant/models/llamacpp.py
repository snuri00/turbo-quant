import math
import time
from typing import Optional, List, Generator

import torch
import numpy as np
from llama_cpp import Llama

from turbo_quant.config import TurboQuantConfig
from turbo_quant.quantizers.turbo_mse import TurboQuantMSE
from turbo_quant.quantizers.turbo_prod import TurboQuantProd
from turbo_quant.quantizers.qjl import QJLQuantizer
from turbo_quant.quantizers.value_quant import ValueQuantizer


class TurboQuantLlamaCpp:
    def __init__(
        self,
        model_path: str,
        config: Optional[TurboQuantConfig] = None,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        n_batch: int = 512,
        verbose: bool = False,
    ):
        if config is None:
            config = TurboQuantConfig(device="cpu")
        self.config = config

        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            verbose=verbose,
        )

        self.n_ctx = n_ctx
        self.n_embd = self.llm.n_embd()
        self.n_vocab = self.llm.n_vocab()
        self._verbose = verbose

    def generate_text(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        stop: Optional[List[str]] = None,
        stream: bool = False,
    ) -> str:
        if stream:
            return self._generate_stream(prompt, max_tokens, temperature, top_p, top_k, stop)

        result = self.llm.create_completion(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
        )
        return result["choices"][0]["text"]

    def _generate_stream(self, prompt, max_tokens, temperature, top_p, top_k, stop):
        output = ""
        for chunk in self.llm.create_completion(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
            stream=True,
        ):
            text = chunk["choices"][0]["text"]
            output += text
            print(text, end="", flush=True)
        print()
        return output

    def chat(
        self,
        messages: List[dict],
        max_tokens: int = 256,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> str:
        if stream:
            output = ""
            for chunk in self.llm.create_chat_completion(
                messages, max_tokens=max_tokens, temperature=temperature, stream=True
            ):
                delta = chunk["choices"][0].get("delta", {})
                text = delta.get("content", "")
                output += text
                print(text, end="", flush=True)
            print()
            return output

        result = self.llm.create_chat_completion(
            messages, max_tokens=max_tokens, temperature=temperature
        )
        return result["choices"][0]["message"]["content"]

    def benchmark_kv_cache(self, context_lengths: List[int] = None):
        if context_lengths is None:
            context_lengths = [128, 256, 512, 1024, 2048]

        head_dim = 128
        num_layers = 32
        num_kv_heads = 8

        print(f"{'Context':>8} | {'FP16 KV':>10} | {'TQ 3-bit':>10} | {'Savings':>8} | {'Max w/ 0.5GB':>12}")
        print("-" * 65)

        for ctx in context_lengths:
            fp16_bytes = 2 * num_layers * num_kv_heads * head_dim * 2 * ctx
            turbo_bytes = 2 * num_layers * num_kv_heads * (
                self.config.key_bits * head_dim / 8 + head_dim / 8 + 4
            ) * ctx

            fp16_mb = fp16_bytes / 1e6
            turbo_mb = turbo_bytes / 1e6
            savings = (1 - turbo_mb / fp16_mb) * 100

            budget_mb = 512
            max_fp16 = int(budget_mb / (fp16_bytes / ctx / 1e6))
            max_turbo = int(budget_mb / (turbo_bytes / ctx / 1e6))

            print(f"{ctx:>8} | {fp16_mb:>8.1f} MB | {turbo_mb:>8.1f} MB | {savings:>6.1f}% | FP16:{max_fp16:>5} TQ:{max_turbo:>5}")

    def quantize_benchmark(self, n_tokens: int = 512):
        head_dim = 128

        print(f"\nTurboQuant quantization quality on {n_tokens} simulated KV vectors (d={head_dim}):")
        print(f"{'Method':>20} | {'Bits':>4} | {'MSE':>10} | {'IP Error':>10} | {'Compression':>11}")
        print("-" * 70)

        keys = torch.randn(n_tokens, head_dim)
        keys = keys / keys.norm(dim=-1, keepdim=True)
        query = torch.randn(1, head_dim)
        exact_ip = (query @ keys.T).squeeze(0)

        for b in [2, 3, 4]:
            mse_q = TurboQuantMSE(head_dim, b, device="cpu")
            idx, norms = mse_q.quantize(keys)
            keys_hat = mse_q.dequantize(idx, norms)
            mse = ((keys - keys_hat) ** 2).sum(dim=-1).mean().item()
            est_ip = (query @ keys_hat.T).squeeze(0)
            ip_err = ((exact_ip - est_ip) ** 2).mean().item()
            comp = 16.0 / b
            print(f"{'TurboQuant_mse':>20} | {b:>4} | {mse:>10.6f} | {ip_err:>10.6f} | {comp:>9.1f}x")

        for b in [2, 3, 4]:
            prod_q = TurboQuantProd(head_dim, b, device="cpu")
            result = prod_q.quantize(keys)
            est_ip = prod_q.estimate_inner_product(query, result).squeeze(0)
            ip_err = ((exact_ip - est_ip) ** 2).mean().item()
            keys_hat = prod_q.dequantize(result)
            mse = ((keys - keys_hat) ** 2).sum(dim=-1).mean().item()
            comp = 16.0 / b
            print(f"{'TurboQuant_prod':>20} | {b:>4} | {mse:>10.6f} | {ip_err:>10.6f} | {comp:>9.1f}x")

    def info(self):
        print(f"Model: {self.n_embd}d, {self.n_vocab} vocab, {self.n_ctx} ctx")
        print(f"TurboQuant: mode={self.config.mode}, key_bits={self.config.key_bits}, value_bits={self.config.value_bits}")
        print(f"Effective bits: {self.config.effective_bits:.2f}")
