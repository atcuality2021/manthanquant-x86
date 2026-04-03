from setuptools import setup, find_packages

setup(
    name="manthanquant-x86",
    version="0.1.0",
    description="TurboQuant KV Cache Compression for vLLM on x86 GPUs",
    author="BiltIQ AI",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
    ],
    extras_require={
        "test": ["pytest>=7.0"],
    },
)
