from torch import nn

from turbo_quant.config import TurboQuantConfig
from turbo_quant.cache.turbo_cache import TurboQuantCache
from turbo_quant.models.llama import patch_llama


def patch_gemma(model: nn.Module, config: TurboQuantConfig, cache: TurboQuantCache):
    patch_llama(model, config, cache)
