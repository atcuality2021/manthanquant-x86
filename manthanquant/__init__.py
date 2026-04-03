"""
ManthanQuant x86 — TurboQuant KV Cache Compression for vLLM

3-bit Lloyd-Max vector quantization for KV cache on x86 discrete GPUs.
Based on TurboQuant (Zandieh et al., 2025).

Algorithm:
  1. Compute radius r = ||x||_2, normalize x_hat = x / r
  2. Scale to N(0,1) space: x_scaled = x_hat * sqrt(D)
  3. Quantize each coordinate with 3-bit Lloyd-Max centroids
  4. Bit-pack indices into int32 words
  5. Store: radius (float32) + packed indices (int32)

Compression: 5.12x (bf16 [N,D=128] = 256B -> radius 4B + packed 48B = 52B)
Quality: 0.978+ cosine similarity, quality-neutral at 3.5 bits/channel
"""

__version__ = "0.1.0"
