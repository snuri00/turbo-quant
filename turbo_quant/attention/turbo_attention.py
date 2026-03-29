import math
from typing import Optional, Tuple

import torch
from torch import Tensor

from turbo_quant.config import TurboQuantConfig
from turbo_quant.cache.turbo_cache import TurboQuantCache
from turbo_quant.attention.score_estimator import ScoreEstimator


_SCORE_ESTIMATORS = {}


def get_or_create_estimator(config: TurboQuantConfig, cache: TurboQuantCache) -> ScoreEstimator:
    cache_id = id(cache)
    if cache_id not in _SCORE_ESTIMATORS:
        _SCORE_ESTIMATORS[cache_id] = ScoreEstimator(config, cache)
    return _SCORE_ESTIMATORS[cache_id]


def turbo_attention_forward(
    query_states: Tensor,
    key_states: Tensor,
    value_states: Tensor,
    layer_idx: int,
    cache: TurboQuantCache,
    config: TurboQuantConfig,
    attention_mask: Optional[Tensor] = None,
    is_prefill: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]:
    batch_size, num_heads, seq_len, head_dim = query_states.shape

    if is_prefill:
        cache.update(layer_idx, key_states, value_states)

        scale = 1.0 / math.sqrt(head_dim)
        scores = torch.matmul(query_states, key_states.transpose(-1, -2)) * scale

        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights.to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        return attn_output, attn_weights

    cache.update(layer_idx, key_states, value_states)

    estimator = get_or_create_estimator(config, cache)
    attn_output, attn_weights = estimator.compute_attention(
        layer_idx, query_states, attention_mask
    )

    return attn_output, attn_weights
