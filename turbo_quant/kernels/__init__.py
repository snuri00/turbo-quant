try:
    from turbo_quant.kernels.quant_ops import turbo_quantize_cuda, turbo_dequantize_cuda
    from turbo_quant.kernels.score_ops import turbo_score_cuda, qjl_score_cuda
    HAS_CUDA_KERNELS = True
except ImportError:
    HAS_CUDA_KERNELS = False
