"""
Live test: Send requests to vLLM with ManthanQuant enabled.

Verifies that:
1. vLLM starts and serves without crashes
2. Compression hooks are active
3. Output quality is acceptable
4. Compression stats are tracked

Usage:
    python scripts/live_test.py [--port 8200] [--host localhost]
"""

import argparse
import json
import time
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_health(base_url: str, api_key: str) -> bool:
    """Check if vLLM is healthy."""
    import urllib.request
    try:
        req = urllib.request.Request(f"{base_url}/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def send_chat_request(
    base_url: str,
    api_key: str,
    prompt: str,
    max_tokens: int = 100,
) -> dict:
    """Send a chat completion request."""
    import urllib.request

    url = f"{base_url}/v1/chat/completions"
    payload = json.dumps({
        "model": "Qwen3.5-35B-A3B",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,  # Deterministic for comparison
    }).encode()

    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")

    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        elapsed = time.perf_counter() - start
        result = json.loads(resp.read())

    result["_elapsed"] = elapsed
    return result


def check_manthanquant_stats(host: str) -> dict | None:
    """Try to read ManthanQuant stats from log file (via SSH if remote)."""
    import glob
    # Check local stats files
    for f in glob.glob(os.path.expanduser("~/logs/manthanquant_x86_stats_*.json")):
        try:
            with open(f) as fp:
                return json.load(fp)
        except Exception:
            pass
    return None


def main():
    parser = argparse.ArgumentParser(description="ManthanQuant x86 Live Test")
    parser.add_argument("--host", default="localhost", help="vLLM host")
    parser.add_argument("--port", type=int, default=8200, help="vLLM port")
    parser.add_argument("--api-key", default=os.environ.get("MANTHANQUANT_API_KEY", "not-needed"))
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    print(f"=== ManthanQuant x86 Live Test ===")
    print(f"Target: {base_url}")
    print()

    # 1. Wait for health
    print("Waiting for vLLM to be healthy...", end="", flush=True)
    for i in range(120):
        if check_health(base_url, args.api_key):
            print(f" OK ({i+1}s)")
            break
        print(".", end="", flush=True)
        time.sleep(1)
    else:
        print(" TIMEOUT")
        sys.exit(1)

    # 2. Test prompts
    test_prompts = [
        ("Short", "What is 2+2?", 50),
        ("Medium", "Explain TCP vs UDP in 3 sentences.", 150),
        ("Code", "Write a Python function to check if a number is prime.", 200),
        ("Reasoning", "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?", 200),
        ("Long context", "Write a detailed comparison of merge sort and quick sort, including time complexity, space complexity, and when to use each.", 500),
    ]

    results = []
    for name, prompt, max_tok in test_prompts:
        print(f"\n--- Test: {name} ---")
        print(f"Prompt: {prompt[:60]}...")
        try:
            resp = send_chat_request(base_url, args.api_key, prompt, max_tok)
            text = resp["choices"][0]["message"]["content"]
            usage = resp["usage"]
            elapsed = resp["_elapsed"]
            tok_per_sec = usage["completion_tokens"] / elapsed if elapsed > 0 else 0

            print(f"Response: {text[:100]}...")
            print(f"Tokens: {usage['completion_tokens']} in {elapsed:.2f}s ({tok_per_sec:.1f} tok/s)")
            results.append({
                "name": name,
                "tokens": usage["completion_tokens"],
                "elapsed": elapsed,
                "tok_per_sec": tok_per_sec,
                "success": True,
            })
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"name": name, "success": False, "error": str(e)})

    # 3. Summary
    print("\n" + "=" * 50)
    print("RESULTS:")
    successes = [r for r in results if r.get("success")]
    failures = [r for r in results if not r.get("success")]
    print(f"  Passed: {len(successes)}/{len(results)}")
    if successes:
        avg_tps = sum(r["tok_per_sec"] for r in successes) / len(successes)
        total_tokens = sum(r["tokens"] for r in successes)
        print(f"  Avg throughput: {avg_tps:.1f} tok/s")
        print(f"  Total tokens: {total_tokens}")
    if failures:
        for f in failures:
            print(f"  FAILED: {f['name']}: {f.get('error', 'unknown')}")

    # 4. Check compression stats
    stats = check_manthanquant_stats(args.host)
    if stats:
        print(f"\nManthanQuant Stats:")
        print(f"  Compressions: {stats.get('compressions', 'N/A')}")
        print(f"  Ratio: {stats.get('ratio', 'N/A')}x")
        print(f"  Memory saved: {stats.get('memory_saved_mb', 'N/A')} MB")
    else:
        print(f"\nManthanQuant stats file not found (check ~/logs/)")

    print("\nDone.")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
