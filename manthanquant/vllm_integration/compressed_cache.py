"""
CompressedKVCache — Two-tier KV cache with TurboQuant compression.

Architecture:
  HOT tier:  Standard vLLM paged cache (bf16/fp16) — used for attention
  COLD tier: Compressed TurboQuant cache (3-bit packed) — 5x smaller

Flow:
  1. vLLM writes K/V to HOT cache normally via reshape_and_cache_flash()
  2. After a block is fully written, compress it to COLD tier
  3. When a previously-evicted block is needed, decompress from COLD to HOT
  4. Attention ALWAYS reads from HOT tier (no attention kernel changes)

This gives real memory savings: we can fit 5x more blocks in the COLD tier
than the HOT tier, allowing much longer contexts.

Block lifecycle:
  NEW → HOT (vLLM writes normally)
  HOT → COLD (block is full, not recently accessed → compress)
  COLD → HOT (block needed for attention → decompress)
  COLD → FREE (request finished → release)

Supports ALL model architectures because it hooks below the attention backends
at the cache write level (_custom_ops.reshape_and_cache_flash).
"""

import logging
import torch
from dataclasses import dataclass, field

from manthanquant.core.quantizer import TurboQuantEncoder, TurboQuantDecoder

logger = logging.getLogger("manthanquant.cache")


@dataclass
class CompressedBlock:
    """One compressed KV cache block.

    Stores both K and V for one block (block_size tokens, all kv_heads).
    """
    # Key compressed data
    k_radii: torch.Tensor    # [block_size * num_kv_heads] float32
    k_packed: torch.Tensor   # [block_size * num_kv_heads, num_words] int32

    # Value compressed data
    v_radii: torch.Tensor    # [block_size * num_kv_heads] float32
    v_packed: torch.Tensor   # [block_size * num_kv_heads, num_words] int32

    # Metadata
    head_dim: int
    num_kv_heads: int
    block_size: int
    num_tokens: int           # actual tokens written (may be < block_size)
    original_dtype: torch.dtype


@dataclass
class CacheStats:
    """Track compression statistics."""
    hot_blocks: int = 0
    cold_blocks: int = 0
    compressions: int = 0
    decompressions: int = 0
    total_original_bytes: int = 0
    total_compressed_bytes: int = 0

    @property
    def ratio(self) -> float:
        if self.total_compressed_bytes == 0:
            return 0.0
        return self.total_original_bytes / self.total_compressed_bytes

    @property
    def saved_mb(self) -> float:
        return (self.total_original_bytes - self.total_compressed_bytes) / (1024 * 1024)


class CompressedKVCache:
    """Two-tier compressed KV cache manager.

    This wraps vLLM's standard paged KV cache. The HOT tier is vLLM's normal
    cache tensor. The COLD tier is our compressed storage.

    Usage:
        cache = CompressedKVCache(bits=3, device='cuda:0')

        # After vLLM writes a block:
        cache.compress_block(block_id, key_cache, value_cache, ...)

        # Before attention needs an old block:
        cache.decompress_block(block_id, key_cache, value_cache, ...)

        # When request finishes:
        cache.release_block(block_id)
    """

    def __init__(
        self,
        bits: int = 3,
        device: torch.device | str = "cuda:0",
    ):
        self.bits = bits
        self.device = torch.device(device)
        self.encoder = TurboQuantEncoder(bits=bits)
        self.decoder = TurboQuantDecoder(bits=bits)

        # block_id → CompressedBlock
        self._cold_store: dict[int, CompressedBlock] = {}

        # Track which blocks are compressed
        self._compressed_block_ids: set[int] = set()

        self.stats = CacheStats()

    def compress_block(
        self,
        block_id: int,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        num_tokens: int | None = None,
    ) -> None:
        """Compress a KV cache block from HOT tier to COLD tier.

        Args:
            block_id: The block identifier in vLLM's page table
            key_cache: [block_size, num_kv_heads, head_dim] — the key data
            value_cache: [block_size, num_kv_heads, head_dim] — the value data
            block_size: Number of token slots per block
            num_kv_heads: Number of KV attention heads
            head_dim: Dimension of each head
            num_tokens: Actual tokens in this block (None = full block)
        """
        if num_tokens is None:
            num_tokens = block_size

        original_dtype = key_cache.dtype

        # Flatten: [block_size, num_kv_heads, head_dim] → [block_size * num_kv_heads, head_dim]
        k_flat = key_cache[:num_tokens].reshape(-1, head_dim)
        v_flat = value_cache[:num_tokens].reshape(-1, head_dim)

        # Encode
        k_radii, k_packed = self.encoder.encode(k_flat)
        v_radii, v_packed = self.encoder.encode(v_flat)

        # Store compressed block
        self._cold_store[block_id] = CompressedBlock(
            k_radii=k_radii,
            k_packed=k_packed,
            v_radii=v_radii,
            v_packed=v_packed,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            block_size=block_size,
            num_tokens=num_tokens,
            original_dtype=original_dtype,
        )
        self._compressed_block_ids.add(block_id)

        # Update stats
        self.stats.compressions += 1
        self.stats.cold_blocks = len(self._cold_store)
        original_bytes = 2 * num_tokens * num_kv_heads * head_dim * 2  # K+V, bf16
        compressed_bytes = (
            (k_radii.nelement() + v_radii.nelement()) * 4 +
            (k_packed.nelement() + v_packed.nelement()) * 4
        )
        self.stats.total_original_bytes += original_bytes
        self.stats.total_compressed_bytes += compressed_bytes

    def decompress_block(
        self,
        block_id: int,
        key_cache_out: torch.Tensor,
        value_cache_out: torch.Tensor,
    ) -> bool:
        """Decompress a block from COLD tier back to HOT tier.

        Writes decompressed K/V data directly into the provided cache tensors.

        Args:
            block_id: The block to decompress
            key_cache_out: [block_size, num_kv_heads, head_dim] — write K here
            value_cache_out: [block_size, num_kv_heads, head_dim] — write V here

        Returns:
            True if block was found and decompressed, False if not in cold store
        """
        if block_id not in self._cold_store:
            return False

        block = self._cold_store[block_id]

        # Decode
        k_flat = self.decoder.decode(block.k_radii, block.k_packed, block.head_dim)
        v_flat = self.decoder.decode(block.v_radii, block.v_packed, block.head_dim)

        # Reshape: [num_tokens * num_kv_heads, head_dim] → [num_tokens, num_kv_heads, head_dim]
        k_reshaped = k_flat.reshape(block.num_tokens, block.num_kv_heads, block.head_dim)
        v_reshaped = v_flat.reshape(block.num_tokens, block.num_kv_heads, block.head_dim)

        # Write to output tensors (cast back to original dtype)
        key_cache_out[:block.num_tokens].copy_(k_reshaped.to(block.original_dtype))
        value_cache_out[:block.num_tokens].copy_(v_reshaped.to(block.original_dtype))

        self.stats.decompressions += 1
        return True

    def is_compressed(self, block_id: int) -> bool:
        """Check if a block is in the cold store."""
        return block_id in self._compressed_block_ids

    def release_block(self, block_id: int) -> None:
        """Release a compressed block (request finished)."""
        if block_id in self._cold_store:
            del self._cold_store[block_id]
            self._compressed_block_ids.discard(block_id)
            self.stats.cold_blocks = len(self._cold_store)

    def clear(self) -> None:
        """Release all compressed blocks."""
        self._cold_store.clear()
        self._compressed_block_ids.clear()
        self.stats = CacheStats()

    def get_stats(self) -> dict:
        """Return current compression statistics."""
        return {
            "cold_blocks": self.stats.cold_blocks,
            "compressions": self.stats.compressions,
            "decompressions": self.stats.decompressions,
            "compression_ratio": round(self.stats.ratio, 2),
            "memory_saved_mb": round(self.stats.saved_mb, 2),
            "total_original_mb": round(self.stats.total_original_bytes / (1024 * 1024), 2),
            "total_compressed_mb": round(self.stats.total_compressed_bytes / (1024 * 1024), 2),
            "bits": self.bits,
        }
