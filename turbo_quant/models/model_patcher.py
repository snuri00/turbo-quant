from typing import Optional
import types

import torch
from torch import nn

from turbo_quant.config import TurboQuantConfig
from turbo_quant.cache.turbo_cache import TurboQuantCache


_PATCHED_MODELS = {}


def patch_model(model: nn.Module, config: Optional[TurboQuantConfig] = None) -> nn.Module:
    if config is None:
        config = TurboQuantConfig()

    model_type = getattr(model.config, "model_type", "unknown")
    num_layers = getattr(model.config, "num_hidden_layers", 32)
    num_heads = getattr(model.config, "num_attention_heads", 32)
    head_dim = getattr(model.config, "hidden_size", 4096) // num_heads
    num_kv_heads = getattr(model.config, "num_key_value_heads", num_heads)

    cache = TurboQuantCache(config, num_layers, num_kv_heads, head_dim)

    model._turbo_quant_config = config
    model._turbo_quant_cache = cache

    if model_type == "llama":
        from turbo_quant.models.llama import patch_llama
        patch_llama(model, config, cache)
    elif model_type == "mistral":
        from turbo_quant.models.mistral import patch_mistral
        patch_mistral(model, config, cache)
    elif model_type == "gemma" or model_type == "gemma2":
        from turbo_quant.models.gemma import patch_gemma
        patch_gemma(model, config, cache)
    else:
        from turbo_quant.models.llama import patch_llama
        patch_llama(model, config, cache)

    _PATCHED_MODELS[id(model)] = cache
    return model


def unpatch_model(model: nn.Module) -> nn.Module:
    model_id = id(model)
    if model_id in _PATCHED_MODELS:
        del _PATCHED_MODELS[model_id]

    if hasattr(model, "_turbo_quant_original_forwards"):
        for layer_idx, original_fn in model._turbo_quant_original_forwards.items():
            attention_layers = _get_attention_layers(model)
            if layer_idx < len(attention_layers):
                attention_layers[layer_idx].forward = original_fn

    if hasattr(model, "_turbo_quant_config"):
        delattr(model, "_turbo_quant_config")
    if hasattr(model, "_turbo_quant_cache"):
        delattr(model, "_turbo_quant_cache")
    if hasattr(model, "_turbo_quant_original_forwards"):
        delattr(model, "_turbo_quant_original_forwards")

    return model


def _get_attention_layers(model: nn.Module):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return [layer.self_attn for layer in model.model.layers]
    elif hasattr(model, "layers"):
        return [layer.self_attn for layer in model.layers]
    return []
