"""
vLLM Patch — Intercept KV cache writes for TurboQuant compression.

Strategy:
  We replace vLLM's _custom_ops.reshape_and_cache_flash() with a wrapper
  that calls the original, then compresses the written block if it's full.

  This works for ALL standard attention models (Llama, Gemma, Qwen, Mistral,
  etc.) because they all write KV cache through this single function.

  For MLA models (DeepSeek), we similarly wrap concat_and_cache_mla().

  For Mamba/SSM models, compression is handled separately at the state
  level (not KV cache — they don't have traditional KV).

Activation:
  Set MANTHANQUANT_ENABLED=1 before starting vLLM.

  Or call install() programmatically before vLLM model loading.
"""

import logging
import os
import torch
from typing import Optional

from manthanquant.vllm_integration.compressed_cache import CompressedKVCache

logger = logging.getLogger("manthanquant.patch")

# Global state
_installed = False
_cache: Optional[CompressedKVCache] = None

# Original functions we replace
_original_reshape_and_cache_flash = None
_original_concat_and_cache_mla = None

# Block tracking: which blocks are full and ready for compression
# Maps (block_id) -> number of tokens written
_block_token_counts: dict[int, int] = {}

# Configuration
_block_size: int = 16  # Updated from vLLM's actual config during install
_num_kv_heads: int = 0
_head_dim: int = 0


def _wrapped_reshape_and_cache_flash(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    """Wrapper around reshape_and_cache_flash that adds compression.

    The original function writes K/V into the paged cache at the given slots.
    We call it normally, then check if any blocks became full and should be
    compressed to the cold tier.

    The key insight: slot_mapping tells us exactly which block slots are being
    written. slot = block_id * block_size + offset_within_block.
    """
    # Call original — KV data is written to cache normally
    _original_reshape_and_cache_flash(
        key, value, key_cache, value_cache,
        slot_mapping, kv_cache_dtype, k_scale, v_scale,
    )

    # Track block fills for compression (skip if profiling/warmup)
    if _cache is None or slot_mapping.numel() == 0:
        return

    # Determine which blocks were written to
    # slot = block_id * block_size + offset
    valid_slots = slot_mapping[slot_mapping >= 0]
    if valid_slots.numel() == 0:
        return

    block_ids = (valid_slots // _block_size).unique()

    for block_id_tensor in block_ids:
        block_id = block_id_tensor.item()

        # Count tokens in this block from this write
        block_slots = valid_slots[
            (valid_slots >= block_id * _block_size) &
            (valid_slots < (block_id + 1) * _block_size)
        ]
        _block_token_counts[block_id] = _block_token_counts.get(block_id, 0) + block_slots.numel()

        # Check if block is full
        if _block_token_counts[block_id] >= _block_size:
            # Extract block data from cache and compress
            _compress_attention_block(
                block_id, key_cache, value_cache
            )


def _compress_attention_block(
    block_id: int,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
) -> None:
    """Compress a full attention block to the cold tier.

    key_cache shape:   [num_blocks, block_size, num_kv_heads, head_size]
    value_cache shape: [num_blocks, block_size, num_kv_heads, head_size]
    """
    if _cache is None or _cache.is_compressed(block_id):
        return

    try:
        # Extract this block's data
        # Note: key_cache/value_cache are the full paged tensors, block_id indexes into them
        k_block = key_cache[block_id]   # [block_size, num_kv_heads, head_size]
        v_block = value_cache[block_id]  # [block_size, num_kv_heads, head_size]

        num_tokens = min(_block_token_counts.get(block_id, _block_size), _block_size)

        _cache.compress_block(
            block_id=block_id,
            key_cache=k_block,
            value_cache=v_block,
            block_size=_block_size,
            num_kv_heads=k_block.shape[1],
            head_dim=k_block.shape[2],
            num_tokens=num_tokens,
        )
    except Exception as e:
        # Never crash vLLM — log and skip
        logger.warning("Compression failed for block %d: %s", block_id, e)


def _wrapped_concat_and_cache_mla(
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str = "auto",
    scale: Optional[torch.Tensor] = None,
) -> None:
    """Wrapper around concat_and_cache_mla for MLA model compression.

    MLA (Multi-head Latent Attention) stores compressed KV in a different
    format: kv_c_normed (compressed latent) + k_pe (position encoding).
    We can still apply TurboQuant on top for additional compression.
    """
    _original_concat_and_cache_mla(
        kv_c_normed, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale
    )

    # MLA compression is a Phase 2 feature — for now just track stats
    # MLA models already compress KV through their latent projection,
    # so additional compression may not be needed


def install(
    bits: int = 3,
    device: str = "cuda:0",
) -> bool:
    """Install ManthanQuant compression hooks into vLLM.

    Call this before vLLM loads the model, or set MANTHANQUANT_ENABLED=1.

    Args:
        bits: Quantization bits (2, 3, or 4)
        device: CUDA device string

    Returns:
        True if installed successfully
    """
    global _installed, _cache
    global _original_reshape_and_cache_flash, _original_concat_and_cache_mla

    if _installed:
        logger.info("ManthanQuant already installed")
        return True

    try:
        from vllm import _custom_ops as ops

        # Save originals
        _original_reshape_and_cache_flash = ops.reshape_and_cache_flash
        _original_concat_and_cache_mla = getattr(ops, 'concat_and_cache_mla', None)

        # Create compressed cache
        _cache = CompressedKVCache(bits=bits, device=device)

        # Replace functions
        ops.reshape_and_cache_flash = _wrapped_reshape_and_cache_flash

        if _original_concat_and_cache_mla is not None:
            ops.concat_and_cache_mla = _wrapped_concat_and_cache_mla

        _installed = True
        msg = f"ManthanQuant x86 TurboQuant ACTIVE: {bits}-bit compression, device={device}"
        logger.info(msg)
        print(f"\n{'='*60}\n  {msg}\n{'='*60}\n", flush=True)
        return True

    except ImportError as e:
        logger.error("Failed to import vllm._custom_ops: %s", e)
        return False
    except Exception as e:
        logger.error("Failed to install ManthanQuant: %s", e)
        return False


def uninstall() -> None:
    """Remove ManthanQuant hooks, restore original vLLM functions."""
    global _installed, _cache
    global _original_reshape_and_cache_flash, _original_concat_and_cache_mla

    if not _installed:
        return

    try:
        from vllm import _custom_ops as ops

        if _original_reshape_and_cache_flash is not None:
            ops.reshape_and_cache_flash = _original_reshape_and_cache_flash

        if _original_concat_and_cache_mla is not None:
            ops.concat_and_cache_mla = _original_concat_and_cache_mla

    except ImportError:
        pass

    _installed = False
    _cache = None
    _block_token_counts.clear()
    logger.info("ManthanQuant x86 uninstalled")


def get_cache() -> Optional[CompressedKVCache]:
    """Get the global compressed cache instance."""
    return _cache


def get_stats() -> dict:
    """Get compression statistics."""
    if _cache is None:
        return {"installed": False}
    stats = _cache.get_stats()
    stats["installed"] = True
    return stats


def configure(block_size: int = 16, **kwargs) -> None:
    """Update runtime configuration from vLLM's actual settings.

    Called after vLLM initializes so we know the real block size.
    """
    global _block_size
    _block_size = block_size
    logger.info("ManthanQuant configured: block_size=%d", block_size)


# Auto-install if environment variable is set.
# At .pth import time, vllm._custom_ops may not exist yet.
# We use a deferred approach: hook __import__ to patch when _custom_ops loads.
if os.environ.get("MANTHANQUANT_ENABLED") == "1":
    _auto_bits = int(os.environ.get("MANTHANQUANT_BITS", "3"))
    _auto_device = os.environ.get("MANTHANQUANT_DEVICE", "cuda:0")

    # Try immediate install first (works if vllm is already imported)
    if not install(bits=_auto_bits, device=_auto_device):
        # Deferred: hook the import system to patch when _custom_ops loads
        import builtins
        _orig_import = builtins.__import__
        _deferred_hooking = False

        def _deferred_import(name, *args, **kwargs):
            global _deferred_hooking
            result = _orig_import(name, *args, **kwargs)
            if (not _deferred_hooking and not _installed
                    and "_custom_ops" in name and "vllm" in str(args)):
                _deferred_hooking = True
                builtins.__import__ = _orig_import  # Restore original
                install(bits=_auto_bits, device=_auto_device)
            return result

        builtins.__import__ = _deferred_import
        logger.info("ManthanQuant deferred install registered (waiting for vllm._custom_ops)")
