"""
Install ManthanQuant autoload into vLLM's Python environment.

Creates a .pth file in site-packages that adds manthanquant to sys.path
and auto-imports the patch when MANTHANQUANT_ENABLED=1 is set.

Usage:
    python scripts/install_autoload.py /path/to/vllm-env
"""

import sys
import os
import site


def install(venv_path: str, manthanquant_path: str):
    """Install .pth autoloader into venv's site-packages."""

    # Find site-packages
    sp_dirs = [
        os.path.join(venv_path, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages"),
    ]

    sp_dir = None
    for d in sp_dirs:
        if os.path.isdir(d):
            sp_dir = d
            break

    if sp_dir is None:
        # Try glob
        import glob
        matches = glob.glob(os.path.join(venv_path, "lib", "python*", "site-packages"))
        if matches:
            sp_dir = matches[0]

    if sp_dir is None:
        print(f"ERROR: Could not find site-packages in {venv_path}")
        sys.exit(1)

    pth_content = f"""\
import os, sys
# ManthanQuant x86 — TurboQuant KV Cache Compression
_mq_path = {manthanquant_path!r}
if _mq_path not in sys.path:
    sys.path.insert(0, _mq_path)
if os.environ.get("MANTHANQUANT_ENABLED") == "1":
    try:
        import manthanquant.vllm_integration.patch
    except Exception as e:
        print(f"ManthanQuant autoload failed: {{e}}")
"""

    pth_file = os.path.join(sp_dir, "manthanquant_autoinstall.pth")
    with open(pth_file, "w") as f:
        f.write(pth_content)

    print(f"Installed: {pth_file}")
    print(f"ManthanQuant path: {manthanquant_path}")
    print(f"Activate: MANTHANQUANT_ENABLED=1 vllm serve ...")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} /path/to/vllm-env [/path/to/manthanquant]")
        sys.exit(1)

    venv = sys.argv[1]
    mq_path = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    install(venv, mq_path)
