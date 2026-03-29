import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cuda_sources = [
    "csrc/turbo_quant_kernel.cu",
    "csrc/turbo_dequant_kernel.cu",
    "csrc/turbo_score_kernel.cu",
    "csrc/turbo_gqa_score_kernel.cu",
    "csrc/qjl_quant_kernel.cu",
    "csrc/qjl_score_kernel.cu",
    "csrc/value_quant_kernel.cu",
    "csrc/polar_transform_kernel.cu",
]

ext_modules = []
try:
    ext_modules.append(
        CUDAExtension(
            name="turbo_quant._C",
            sources=cuda_sources,
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math", "-lineinfo"],
            },
        )
    )
except Exception:
    pass

setup(
    name="turbo_quant",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension} if ext_modules else {},
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "scipy>=1.10.0",
        "numpy>=1.24.0",
    ],
    python_requires=">=3.9",
)
