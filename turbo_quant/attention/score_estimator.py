import math

import torch
from torch import Tensor

from turbo_quant.config import TurboQuantConfig
from turbo_quant.cache.turbo_cache import TurboQuantCache


class ScoreEstimator:
    def __init__(self, config: TurboQuantConfig, cache: TurboQuantCache):
        self.config = config
        self.cache = cache

    def compute_attention(
        self,
        layer_idx: int,
        query_states: Tensor,
        attention_mask: Tensor = None,
    ) -> Tensor:
        head_dim = query_states.shape[-1]
        scale = 1.0 / math.sqrt(head_dim)

        scores = self.cache.get_attention_scores(layer_idx, query_states)
        scores = scores * scale

        if attention_mask is not None:
            seq_len = scores.shape[-1]
            if attention_mask.shape[-1] >= seq_len:
                mask = attention_mask[..., :seq_len]
            else:
                mask = attention_mask
            scores = scores + mask

        attn_weights = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights.to(query_states.dtype)

        output = self.cache.get_value_output(layer_idx, attn_weights)

        return output, attn_weights
