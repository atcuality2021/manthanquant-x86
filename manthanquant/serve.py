"""
Launch vLLM with ManthanQuant compression active.

This wrapper sets up a sitecustomize.py that deferred-patches
vllm._custom_ops.reshape_and_cache_flash in ALL processes
(parent API server AND child EngineCore).

Usage:
    python scripts/vllm_serve_with_compression.py serve <model> [vllm args...]

This is equivalent to: vllm serve ... but with compression hooks active.
"""


def main():
    import os
    import sys
    import tempfile
    import atexit
    import shutil

    # Force ManthanQuant path
    mq_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if mq_path not in sys.path:
        sys.path.insert(0, mq_path)

    # Set env for child processes
    # NOTE: Do NOT set MANTHANQUANT_ENABLED=1 — the old .pth file in site-packages
    # would load the broken vllm_patch_x86.py from the mcms repo.
    # Use MANTHANQUANT_X86=1 instead (checked by our sitecustomize).
    os.environ["MANTHANQUANT_X86"] = "1"
    os.environ["MANTHANQUANT_PATH"] = mq_path
    os.environ.setdefault("MANTHANQUANT_BITS", "3")

    # Create a temporary sitecustomize module that patches _custom_ops
    # in every process (parent + child EngineCore)
    _tmpdir = tempfile.mkdtemp(prefix="manthanquant_")
    _sitecustom_path = os.path.join(_tmpdir, "sitecustomize.py")

    # Use threading to patch AFTER imports are complete (avoids circular import)
    _sitecustom_code = f'''
import os, sys, threading

_mq_path = {mq_path!r}
if _mq_path not in sys.path:
    sys.path.insert(0, _mq_path)

def _deferred_patch():
    """Wait for vllm._custom_ops to be in sys.modules, then patch."""
    import time
    for _ in range(600):  # Wait up to 60 seconds
        if "vllm._custom_ops" in sys.modules:
            break
        time.sleep(0.1)
    else:
        return  # Never loaded

    try:
        import torch
        ops = sys.modules["vllm._custom_ops"]
        from manthanquant.vllm_integration.compressed_cache import CompressedKVCache

        _bits = int(os.environ.get("MANTHANQUANT_BITS", "3"))
        _cache = CompressedKVCache(bits=_bits)
        _stats = {{"calls": 0, "errors": 0}}
        _original = ops.reshape_and_cache_flash

        def _patched(key, value, key_cache, value_cache,
                    slot_mapping, kv_cache_dtype, k_scale, v_scale):
            _original(key, value, key_cache, value_cache,
                     slot_mapping, kv_cache_dtype, k_scale, v_scale)
            try:
                _stats["calls"] += 1
                valid = slot_mapping[slot_mapping >= 0]
                if valid.numel() > 0 and key_cache.ndim >= 3:
                    bs = key_cache.shape[1] if key_cache.ndim == 4 else 16
                    bids = (valid // bs).unique()
                    for bid_t in bids:
                        bid = bid_t.item()
                        if not _cache.is_compressed(bid) and bid < key_cache.shape[0]:
                            kb = key_cache[bid]
                            vb = value_cache[bid]
                            if kb.ndim == 3:
                                _cache.compress_block(
                                    block_id=bid, key_cache=kb, value_cache=vb,
                                    block_size=kb.shape[0], num_kv_heads=kb.shape[1],
                                    head_dim=kb.shape[2],
                                )
                if _stats["calls"] % 200 == 0:
                    s = _cache.get_stats()
                    print(f"[ManthanQuant pid={{os.getpid()}}] calls={{_stats['calls']}} "
                          f"compressed={{s['compressions']}} "
                          f"ratio={{s['compression_ratio']}}x "
                          f"saved={{s['memory_saved_mb']}}MB", flush=True)
            except Exception as e:
                _stats["errors"] += 1
                if _stats["errors"] <= 3:
                    print(f"[ManthanQuant] error: {{e}}", flush=True)

        ops.reshape_and_cache_flash = _patched
        print(f"===== ManthanQuant TurboQuant ACTIVE (pid={{os.getpid()}}) =====", flush=True)

        # V1 fix: patch cached function references in all attention backends.
        # Flash Attention backends cache reshape_and_cache_flash at import.
        # Triton Attention uses triton_reshape_and_cache_flash instead.
        _v1_flash_modules = [
            "vllm.v1.attention.backends.fa_utils",
            "vllm.v1.attention.backends.flash_attn",
        ]
        _v1_triton_modules = [
            ("vllm.v1.attention.ops.triton_reshape_and_cache_flash",
             "triton_reshape_and_cache_flash"),
            ("vllm.v1.attention.backends.triton_attn",
             "triton_reshape_and_cache_flash"),
        ]

        def _make_triton_wrapper(original_fn):
            """Wrap triton_reshape_and_cache_flash with compression hook."""
            def _triton_patched(key, value, key_cache, value_cache,
                               slot_mapping, kv_cache_dtype, k_scale, v_scale):
                original_fn(key, value, key_cache, value_cache,
                           slot_mapping, kv_cache_dtype, k_scale, v_scale)
                try:
                    _stats["calls"] += 1
                    valid = slot_mapping[slot_mapping >= 0]
                    if valid.numel() > 0 and key_cache.ndim >= 3:
                        # Triton uses [num_blocks, block_size, num_heads, head_size]
                        bs = key_cache.shape[1]
                        bids = (valid // bs).unique()
                        for bid_t in bids:
                            bid = bid_t.item()
                            if not _cache.is_compressed(bid) and bid < key_cache.shape[0]:
                                kb = key_cache[bid]
                                vb = value_cache[bid]
                                if kb.ndim == 3:
                                    _cache.compress_block(
                                        block_id=bid, key_cache=kb, value_cache=vb,
                                        block_size=kb.shape[0], num_kv_heads=kb.shape[1],
                                        head_dim=kb.shape[2],
                                    )
                    if _stats["calls"] % 200 == 0:
                        s = _cache.get_stats()
                        print(f"[ManthanQuant pid={{os.getpid()}}] calls={{_stats['calls']}} "
                              f"compressed={{s['compressions']}} "
                              f"ratio={{s['compression_ratio']}}x "
                              f"saved={{s['memory_saved_mb']}}MB", flush=True)
                except Exception as e:
                    _stats["errors"] += 1
                    if _stats["errors"] <= 3:
                        print(f"[ManthanQuant] error: {{e}}", flush=True)
            return _triton_patched

        def _patch_v1_ref(mod_name, attr="reshape_and_cache_flash", wrapper=None):
            if mod_name in sys.modules:
                m = sys.modules[mod_name]
                if hasattr(m, attr):
                    if wrapper is None:
                        setattr(m, attr, _patched)
                    else:
                        # Create wrapper around the original triton function
                        orig = getattr(m, attr)
                        setattr(m, attr, wrapper(orig))
                    print(f"[ManthanQuant] {{mod_name}}.{{attr}} patched (pid={{os.getpid()}})", flush=True)
                    return True
            return False

        # Immediate patch for already-loaded modules
        for mn in _v1_flash_modules:
            _patch_v1_ref(mn)
        for mn, attr in _v1_triton_modules:
            _patch_v1_ref(mn, attr=attr, wrapper=_make_triton_wrapper)

        # Deferred patch for modules not yet loaded
        def _patch_v1_deferred():
            import time as _t2
            _patched_set = set()
            all_targets = (
                [(mn, "reshape_and_cache_flash", None) for mn in _v1_flash_modules]
                + [(mn, attr, _make_triton_wrapper) for mn, attr in _v1_triton_modules]
            )
            for _ in range(300):  # Wait up to 30 seconds
                for mn, attr, wrapper in all_targets:
                    if mn not in _patched_set and mn in sys.modules:
                        _patch_v1_ref(mn, attr=attr, wrapper=wrapper)
                        _patched_set.add(mn)
                if len(_patched_set) == len(all_targets):
                    break
                _t2.sleep(0.1)

        import threading as _th2
        _th2.Thread(target=_patch_v1_deferred, daemon=True).start()
    except Exception as e:
        print(f"ManthanQuant patch failed: {{e}}", flush=True)

# Start patcher in background thread — fires after _custom_ops loads
_t = threading.Thread(target=_deferred_patch, daemon=True)
_t.start()
'''

    with open(_sitecustom_path, 'w') as f:
        f.write(_sitecustom_code)

    # Set PYTHONPATH so ALL child processes get sitecustomize + manthanquant
    existing_pp = os.environ.get("PYTHONPATH", "")
    parts = [_tmpdir, mq_path]
    if existing_pp:
        parts.append(existing_pp)
    os.environ["PYTHONPATH"] = ":".join(parts)

    # Clean up on exit
    def _cleanup():
        shutil.rmtree(_tmpdir, ignore_errors=True)
    atexit.register(_cleanup)

    print(f"ManthanQuant: hooks will activate when _custom_ops loads", flush=True)

    # Run vllm CLI
    from vllm.entrypoints.cli.main import main as vllm_main
    sys.exit(vllm_main())


if __name__ == "__main__":
    main()
