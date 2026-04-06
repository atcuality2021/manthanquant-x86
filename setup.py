"""
ManthanQuant x86 — TurboQuant KV Cache Compression for vLLM
BiltIQ AI

Build:
    pip install -e .                    # PyTorch-only (Phase 1)
    MANTHANQUANT_BUILD_CUDA=1 pip install -e .  # With CUDA kernels (Phase 2)
"""

import os
from setuptools import setup, find_packages

# Check if CUDA build is requested
BUILD_CUDA = os.environ.get("MANTHANQUANT_BUILD_CUDA", "0") == "1"

ext_modules = []
cmdclass = {}

if BUILD_CUDA:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    # Target architectures:
    # SM 8.0 = A100, 8.6 = RTX 3090, 8.9 = RTX 4090/4070
    # SM 9.0 = H100, 12.0 = RTX 6000 Blackwell
    cuda_arch = os.environ.get(
        "TORCH_CUDA_ARCH_LIST",
        "8.0 8.6 8.9 9.0 12.0"
    )
    os.environ["TORCH_CUDA_ARCH_LIST"] = cuda_arch

    ext_modules.append(
        CUDAExtension(
            name="manthanquant._C",
            sources=[
                "csrc/bindings.cpp",
                "csrc/turboquant_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-std=c++17",
                    "--threads=4",
                    "--allow-unsupported-compiler",
                    "-ccbin", os.environ.get("CXX", "g++-14"),
                ],
            },
        )
    )
    cmdclass["build_ext"] = BuildExtension

setup(
    name="manthanquant-x86",
    version="0.2.0",
    description="TurboQuant KV Cache Compression for vLLM on x86 GPUs",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="BiltIQ AI",
    url="https://github.com/atcuality2021/manthanquant-x86",
    license="Apache-2.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
    ],
    extras_require={
        "test": ["pytest>=7.0"],
    },
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
