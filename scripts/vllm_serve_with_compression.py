#!/usr/bin/env python3
"""
Launch vLLM with ManthanQuant compression active.

Thin wrapper that delegates to manthanquant.serve.main().

Usage:
    python scripts/vllm_serve_with_compression.py serve <model> [vllm args...]
"""

import sys
import os

# Ensure manthanquant is importable
mq_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if mq_path not in sys.path:
    sys.path.insert(0, mq_path)

from manthanquant.serve import main

if __name__ == "__main__":
    main()
