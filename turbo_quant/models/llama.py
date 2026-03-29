import math
import types
from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from turbo_quant.config import TurboQuantConfig
from turbo_quant.cache.turbo_cache import TurboQuantCache
from turbo_quant.attention.turbo_attention import turbo_attention_forward


def patch_llama(model: nn.Module, config: TurboQuantConfig, cache: TurboQuantCache):
    model._turbo_quant_original_forwards = {}

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    else:
        return

    for layer_idx, layer in enumerate(layers):
        attn = layer.self_attn
        model._turbo_quant_original_forwards[layer_idx] = attn.forward

        attn._turbo_layer_idx = layer_idx
        attn._turbo_cache = cache
        attn._turbo_config = config
        attn._turbo_is_prefill_done = False

        attn.forward = types.MethodType(_llama_attention_forward, attn)


def _llama_attention_forward(
    self,
    hidden_states: Tensor,
    attention_mask: Optional[Tensor] = None,
    position_ids: Optional[Tensor] = None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[Tensor] = None,
    position_embeddings: Optional[Tuple[Tensor, Tensor]] = None,
    **kwargs,
) -> Tuple[Tensor, Optional[Tensor], Optional[Tuple[Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    num_heads = self.num_heads
    num_kv_heads = getattr(self, "num_key_value_heads", num_heads)
    head_dim = self.head_dim

    query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

    if position_embeddings is not None:
        cos, sin = position_embeddings
        query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin)
    elif hasattr(self, "rotary_emb"):
        if position_ids is not None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            seq_len = key_states.shape[-2]
            if past_key_value is not None and hasattr(past_key_value, "get_seq_len"):
                seq_len += past_key_value.get_seq_len(self._turbo_layer_idx)
            position_ids_auto = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
            cos, sin = self.rotary_emb(value_states, position_ids_auto)
        query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin)

    num_kv_groups = num_heads // num_kv_heads
    if num_kv_groups > 1:
        key_states_expanded = key_states.unsqueeze(2).expand(-1, -1, num_kv_groups, -1, -1)
        key_states_expanded = key_states_expanded.reshape(bsz, num_heads, q_len, head_dim)
        value_states_expanded = value_states.unsqueeze(2).expand(-1, -1, num_kv_groups, -1, -1)
        value_states_expanded = value_states_expanded.reshape(bsz, num_heads, q_len, head_dim)
    else:
        key_states_expanded = key_states
        value_states_expanded = value_states

    is_prefill = q_len > 1 or not self._turbo_is_prefill_done

    attn_output, attn_weights = turbo_attention_forward(
        query_states=query_states,
        key_states=key_states,
        value_states=value_states,
        layer_idx=self._turbo_layer_idx,
        cache=self._turbo_cache,
        config=self._turbo_config,
        attention_mask=attention_mask,
        is_prefill=is_prefill,
    )

    if is_prefill:
        self._turbo_is_prefill_done = True

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, num_heads * head_dim)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def _apply_rotary_pos_emb(q, k, cos, sin):
    if cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
