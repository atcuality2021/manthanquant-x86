#!/usr/bin/env python3
"""
ManthanQuant x86 — Full Benchmark Suite

Tests:
  1. Output quality: same prompt → compare tokens with/without compression
  2. Throughput: single request tok/s across prompt types
  3. Latency: TTFT (time to first token)
  4. Concurrent scaling: 1/2/4/8 simultaneous requests
  5. Cross-cluster: RTX 6000 (with ManthanQuant) vs GB10 (baseline)

Usage:
    python scripts/benchmark_full.py
"""

import json
import time
import sys
import os
import concurrent.futures
from datetime import datetime, timezone

# ── Config ───────────────────────────────────────────────────────────────

MK = os.environ.get("MANTHANQUANT_API_KEY", "not-needed")

ENDPOINTS = {
    "rtx6000_manthanquant": {
        "url": "http://192.168.29.139:8200",
        "label": "RTX 6000 + ManthanQuant 3-bit",
        "gpu": "RTX PRO 6000 Blackwell (96GB)",
    },
    "gb10_baseline": {
        "url": "http://192.168.29.113:8000",
        "label": "GB10 Baseline (no compression)",
        "gpu": "DGX Spark GB10 (128GB unified)",
    },
}

MODEL = "Qwen3.5-35B-A3B"

TEST_PROMPTS = {
    "math_simple": {
        "prompt": "What is the sum of the first 10 prime numbers?",
        "max_tokens": 100,
    },
    "code_python": {
        "prompt": "Write a Python function that implements binary search on a sorted list. Include docstring and type hints.",
        "max_tokens": 300,
    },
    "reasoning": {
        "prompt": "A farmer has 17 sheep. All but 9 die. How many sheep are left? Explain your reasoning step by step.",
        "max_tokens": 200,
    },
    "summarization": {
        "prompt": "Explain the difference between TCP and UDP protocols. Cover reliability, ordering, speed, and use cases. Be concise.",
        "max_tokens": 300,
    },
    "long_generation": {
        "prompt": "Write a detailed comparison of merge sort and quick sort algorithms, including time complexity analysis for best, average, and worst cases, space complexity, stability, and practical recommendations.",
        "max_tokens": 500,
    },
    "multi_turn_context": {
        "prompt": "I'm building a web application with React frontend and FastAPI backend. The app needs real-time updates. What architecture pattern should I use and why? Consider WebSockets vs SSE vs polling.",
        "max_tokens": 400,
    },
}

CONCURRENT_PROMPT = {
    "prompt": "Explain what a hash table is in 2 sentences.",
    "max_tokens": 100,
}


# ── HTTP client ──────────────────────────────────────────────────────────

def send_request(base_url: str, prompt: str, max_tokens: int, temperature: float = 0.0, stream: bool = False):
    """Send chat completion request. Returns (response_dict, elapsed_seconds, ttft_seconds)."""
    import urllib.request

    url = f"{base_url}/v1/chat/completions"
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }).encode()

    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {MK}")

    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            if stream:
                # Read SSE for TTFT measurement
                ttft = None
                full_content = ""
                for line in resp:
                    line = line.decode().strip()
                    if line.startswith("data: ") and line != "data: [DONE]":
                        if ttft is None:
                            ttft = time.perf_counter() - start
                        chunk = json.loads(line[6:])
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        full_content += delta.get("content", "")
                elapsed = time.perf_counter() - start
                return {"content": full_content, "ttft": ttft, "elapsed": elapsed}
            else:
                elapsed = time.perf_counter() - start
                result = json.loads(resp.read())
                return {
                    "content": result["choices"][0]["message"]["content"],
                    "prompt_tokens": result["usage"]["prompt_tokens"],
                    "completion_tokens": result["usage"]["completion_tokens"],
                    "elapsed": elapsed,
                }
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {"error": str(e), "elapsed": elapsed}


# ── Tests ────────────────────────────────────────────────────────────────

def test_output_quality(endpoints: dict):
    """Compare output tokens between endpoints for same prompts."""
    print("\n" + "=" * 70)
    print("TEST 1: OUTPUT QUALITY COMPARISON")
    print("=" * 70)

    results = {}
    for pname, pconfig in TEST_PROMPTS.items():
        results[pname] = {}
        print(f"\n--- {pname} ---")
        print(f"Prompt: {pconfig['prompt'][:70]}...")

        for ename, econfig in endpoints.items():
            resp = send_request(econfig["url"], pconfig["prompt"], pconfig["max_tokens"])
            if "error" in resp:
                print(f"  {econfig['label']}: ERROR {resp['error']}")
                results[pname][ename] = {"error": resp["error"]}
            else:
                content = resp["content"]
                tokens = resp["completion_tokens"]
                tps = tokens / resp["elapsed"] if resp["elapsed"] > 0 else 0
                print(f"  {econfig['label']}: {tokens} tok in {resp['elapsed']:.2f}s ({tps:.1f} tok/s)")
                print(f"    Output: {content[:120]}...")
                results[pname][ename] = {
                    "content": content,
                    "tokens": tokens,
                    "elapsed": resp["elapsed"],
                    "tok_per_sec": round(tps, 1),
                }

    return results


def test_throughput(endpoints: dict):
    """Measure throughput for each prompt type on each endpoint."""
    print("\n" + "=" * 70)
    print("TEST 2: THROUGHPUT (tok/s)")
    print("=" * 70)

    results = {}
    for ename, econfig in endpoints.items():
        results[ename] = {}
        print(f"\n{econfig['label']}:")
        for pname, pconfig in TEST_PROMPTS.items():
            resp = send_request(econfig["url"], pconfig["prompt"], pconfig["max_tokens"])
            if "error" in resp:
                print(f"  {pname}: ERROR")
                results[ename][pname] = {"error": resp["error"]}
            else:
                tps = resp["completion_tokens"] / resp["elapsed"]
                print(f"  {pname}: {tps:.1f} tok/s ({resp['completion_tokens']} tokens, {resp['elapsed']:.2f}s)")
                results[ename][pname] = {
                    "tok_per_sec": round(tps, 1),
                    "tokens": resp["completion_tokens"],
                    "elapsed": round(resp["elapsed"], 3),
                }

    return results


def test_concurrent_scaling(endpoints: dict):
    """Test concurrent request handling: 1, 2, 4 simultaneous."""
    print("\n" + "=" * 70)
    print("TEST 3: CONCURRENT SCALING")
    print("=" * 70)

    results = {}
    for ename, econfig in endpoints.items():
        results[ename] = {}
        print(f"\n{econfig['label']}:")

        for n_concurrent in [1, 2, 4]:
            start = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_concurrent) as pool:
                futures = [
                    pool.submit(
                        send_request, econfig["url"],
                        CONCURRENT_PROMPT["prompt"],
                        CONCURRENT_PROMPT["max_tokens"],
                    )
                    for _ in range(n_concurrent)
                ]
                responses = [f.result() for f in concurrent.futures.as_completed(futures)]

            wall_time = time.perf_counter() - start
            total_tokens = sum(r.get("completion_tokens", 0) for r in responses if "error" not in r)
            successes = sum(1 for r in responses if "error" not in r)
            agg_tps = total_tokens / wall_time if wall_time > 0 else 0

            print(f"  {n_concurrent} concurrent: {agg_tps:.1f} agg tok/s, {successes}/{n_concurrent} success, {wall_time:.2f}s")
            results[ename][str(n_concurrent)] = {
                "agg_tok_per_sec": round(agg_tps, 1),
                "wall_time": round(wall_time, 3),
                "total_tokens": total_tokens,
                "successes": successes,
            }

    return results


def test_compression_stats(endpoints: dict):
    """Get ManthanQuant compression statistics from RTX 6000."""
    print("\n" + "=" * 70)
    print("TEST 4: MANTHANQUANT COMPRESSION STATS")
    print("=" * 70)

    # Send a few more requests to generate stats
    rtx_url = endpoints.get("rtx6000_manthanquant", {}).get("url", "")
    if not rtx_url:
        return {}

    for _ in range(3):
        send_request(rtx_url, "Explain recursion in programming.", 200)

    # Read stats from log (via the endpoint's stats output)
    print("  Compression stats are logged to vLLM output.")
    print("  Check: grep 'ManthanQuant' ~/logs/vllm-manthanquant-test.log")
    return {"note": "check vLLM log for detailed stats"}


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    print(f"{'=' * 70}")
    print(f"ManthanQuant x86 — Full Benchmark Suite")
    print(f"BiltIQ AI | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {MODEL}")
    print(f"{'=' * 70}")

    # Check endpoint health
    active_endpoints = {}
    for ename, econfig in ENDPOINTS.items():
        try:
            import urllib.request
            req = urllib.request.Request(f"{econfig['url']}/health")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    print(f"  {econfig['label']}: HEALTHY")
                    active_endpoints[ename] = econfig
        except Exception as e:
            print(f"  {econfig['label']}: OFFLINE ({e})")

    if not active_endpoints:
        print("ERROR: No endpoints available")
        return 1

    # Run tests
    all_results = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": MODEL,
            "endpoints": {k: v["label"] for k, v in active_endpoints.items()},
        },
    }

    all_results["quality"] = test_output_quality(active_endpoints)
    all_results["throughput"] = test_throughput(active_endpoints)
    all_results["concurrent"] = test_concurrent_scaling(active_endpoints)
    all_results["compression"] = test_compression_stats(active_endpoints)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for ename, econfig in active_endpoints.items():
        throughput_data = all_results["throughput"].get(ename, {})
        avg_tps = 0
        count = 0
        for pdata in throughput_data.values():
            if isinstance(pdata, dict) and "tok_per_sec" in pdata:
                avg_tps += pdata["tok_per_sec"]
                count += 1
        avg_tps = avg_tps / count if count > 0 else 0
        print(f"  {econfig['label']}: avg {avg_tps:.1f} tok/s across {count} tests")

    # Save results
    report_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks")
    os.makedirs(report_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    json_path = os.path.join(report_dir, f"benchmark_manthanquant_x86_{ts}.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved: {json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
