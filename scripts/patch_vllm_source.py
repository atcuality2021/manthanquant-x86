"""
Patch vLLM's _custom_ops.py to add ManthanQuant compression hooks.

This modifies the installed vLLM to call ManthanQuant's compression
after each KV cache write when MANTHANQUANT_ENABLED=1.

Usage:
    python scripts/patch_vllm_source.py /path/to/vllm-env

Reverting:
    python scripts/patch_vllm_source.py /path/to/vllm-env --revert
"""

import sys
import os
import shutil

PATCH_MARKER = "# >>> MANTHANQUANT PATCH <<<"

HOOK_CODE = '''
# >>> MANTHANQUANT PATCH <<<
# ManthanQuant x86 — TurboQuant KV cache compression hooks (BiltIQ AI)
import os as _mq_os
_mq_enabled = _mq_os.environ.get("MANTHANQUANT_ENABLED") == "1"
_mq_cache = None
_mq_block_size = 16
_mq_stats = {"calls": 0, "errors": 0}

if _mq_enabled:
    try:
        _mq_path = _mq_os.environ.get("MANTHANQUANT_PATH", "")
        if not _mq_path:
            print("MANTHANQUANT_PATH not set — cannot load compression hooks", flush=True)
            _mq_enabled = False
            raise ImportError("MANTHANQUANT_PATH required")
        import sys as _mq_sys
        if _mq_path not in _mq_sys.path:
            _mq_sys.path.insert(0, _mq_path)
        from manthanquant.vllm_integration.compressed_cache import CompressedKVCache
        _mq_bits = int(_mq_os.environ.get("MANTHANQUANT_BITS", "3"))
        _mq_cache = CompressedKVCache(bits=_mq_bits)
        print(f"\\n{'='*60}")
        print(f"  ManthanQuant x86 TurboQuant ACTIVE: {_mq_bits}-bit compression")
        print(f"{'='*60}\\n", flush=True)
    except Exception as _mq_e:
        print(f"ManthanQuant init failed: {_mq_e}", flush=True)
# >>> END MANTHANQUANT PATCH <<<
'''

NEW_RESHAPE_AND_CACHE = '''def reshape_and_cache_flash(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    torch.ops._C_cache_ops.reshape_and_cache_flash(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )
    # ManthanQuant: compress written blocks
    if _mq_enabled and _mq_cache is not None:
        try:
            _mq_stats["calls"] += 1
            valid = slot_mapping[slot_mapping >= 0]
            if valid.numel() > 0:
                block_ids = (valid // _mq_block_size).unique()
                for bid_t in block_ids:
                    bid = bid_t.item()
                    if not _mq_cache.is_compressed(bid) and bid < key_cache.shape[0]:
                        k_blk = key_cache[bid]
                        v_blk = value_cache[bid]
                        _mq_cache.compress_block(
                            block_id=bid,
                            key_cache=k_blk,
                            value_cache=v_blk,
                            block_size=k_blk.shape[0],
                            num_kv_heads=k_blk.shape[1],
                            head_dim=k_blk.shape[2],
                        )
            if _mq_stats["calls"] % 500 == 0:
                s = _mq_cache.get_stats()
                print(f"[ManthanQuant] calls={_mq_stats['calls']} "
                      f"compressed={s['compressions']} "
                      f"ratio={s['compression_ratio']}x "
                      f"saved={s['memory_saved_mb']}MB", flush=True)
        except Exception as _e:
            _mq_stats["errors"] += 1
            if _mq_stats["errors"] <= 5:
                print(f"[ManthanQuant] compress error: {_e}", flush=True)
'''

ORIGINAL_RESHAPE_AND_CACHE = '''def reshape_and_cache_flash(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    torch.ops._C_cache_ops.reshape_and_cache_flash(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )
'''


def find_custom_ops(venv_path: str) -> str:
    import glob
    matches = glob.glob(os.path.join(venv_path, "lib", "python*", "site-packages", "vllm", "_custom_ops.py"))
    if not matches:
        print(f"ERROR: Could not find vllm/_custom_ops.py in {venv_path}")
        sys.exit(1)
    return matches[0]


def patch(filepath: str):
    with open(filepath) as f:
        content = f.read()

    if PATCH_MARKER in content:
        print(f"Already patched: {filepath}")
        return

    # Backup
    backup = filepath + ".manthanquant.bak"
    if not os.path.exists(backup):
        shutil.copy2(filepath, backup)
        print(f"Backup: {backup}")

    # Insert hook code before the function
    old_func = ORIGINAL_RESHAPE_AND_CACHE.strip()
    new_func = NEW_RESHAPE_AND_CACHE.strip()

    if old_func not in content:
        print("ERROR: Could not find reshape_and_cache_flash function to patch")
        sys.exit(1)

    # Add hook code at the top (after imports)
    # Find the end of imports section
    lines = content.split('\n')
    insert_idx = 0
    for i, line in enumerate(lines):
        if line.startswith('def ') or line.startswith('class '):
            insert_idx = i
            break

    # Insert the hook code
    hook_lines = HOOK_CODE.strip().split('\n')
    for j, hl in enumerate(hook_lines):
        lines.insert(insert_idx + j, hl)

    content = '\n'.join(lines)

    # Replace the function
    content = content.replace(old_func, new_func)

    with open(filepath, 'w') as f:
        f.write(content)

    print(f"Patched: {filepath}")


def revert(filepath: str):
    backup = filepath + ".manthanquant.bak"
    if os.path.exists(backup):
        shutil.copy2(backup, filepath)
        print(f"Reverted: {filepath}")
    else:
        print(f"No backup found: {backup}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} /path/to/vllm-env [--revert]")
        sys.exit(1)

    venv = sys.argv[1]
    filepath = find_custom_ops(venv)

    if "--revert" in sys.argv:
        revert(filepath)
    else:
        patch(filepath)
