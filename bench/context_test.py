#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx>=0.27"]
# ///
"""Context-length test for the ANIMA Thor coding engine.

32768 is a conservative default — Qwen3.6-35B-A3B has aggressive GQA (2 KV heads),
so KV is only ~40 KB/token and the box can serve far longer context. This tool finds
the REAL ceiling and verifies long-context *quality*, not just that it loads:

  1. serve <model> at a target max_model_len via the UI (/api/serve)
  2. read vLLM's own startup log for the actual GPU-KV-cache token count
     ("GPU KV cache size: N tokens") — the ground-truth pool, not a guess
  3. needle-in-haystack: bury a unique marker deep in a long prompt and check the
     model recalls it — proves attention works at depth, not just that it accepts the
     tokens. Also reports decode tok/s at long context.

Usage:
  uv run context_test.py --ui http://127.0.0.1:7000 \
      --model nvidia/Qwen3.6-35B-A3B-NVFP4 --served-name qwen36 \
      --maxlen 131072 --util 0.75 --needle-tokens 24000
"""
import argparse
import json
import re
import sys
import time

import httpx

KV_RE = re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens", re.I)
CONC_RE = re.compile(r"Maximum concurrency for\s*([\d,]+)\s*tokens", re.I)


def _serve(ui: str, args) -> dict:
    body = {"model": args.model, "served_name": args.served_name,
            "max_model_len": args.maxlen, "gpu_memory_utilization": str(args.util),
            "kv_cache_dtype": "fp8", "attention_backend": "TRITON_ATTN",
            "profile": "latency", "spec_decode": "off"}
    r = httpx.post(f"{ui}/api/serve", json=body, timeout=180)
    print(f"  /api/serve -> {r.status_code} {r.text[:200]}")
    return r.json() if r.status_code < 300 else {"error": r.text}


def _wait_ready(ui: str, timeout_s: int = 420) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            h = httpx.get(f"{ui}/healthz", timeout=5).json()
            if h.get("engine_ready"):
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(10)
    return False


def _kv_from_logs(ui: str) -> dict:
    try:
        logs = httpx.get(f"{ui}/api/logs", params={"tail": 400}, timeout=10).json().get("logs", "")
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    if (m := KV_RE.search(logs)):
        out["kv_cache_tokens"] = int(m.group(1).replace(",", ""))
    if (m := CONC_RE.search(logs)):
        out["max_concurrency_tokens"] = int(m.group(1).replace(",", ""))
    return out


def _needle(ui: str, served: str, n_tokens: int) -> dict:
    """Bury a unique marker mid-context; ask the model to return it verbatim."""
    secret = "PURPLE-GIRAFFE-4827"
    # ~4 chars/token of filler code; place the secret ~60% deep
    filler_line = "def process_batch(idx): return idx * 2 + 1  # routine helper\n"
    n_lines = max(1, (n_tokens * 4) // len(filler_line))
    head = filler_line * int(n_lines * 0.6)
    tail = filler_line * int(n_lines * 0.4)
    doc = (head
           + f"\n# IMPORTANT CONFIG TOKEN: the deploy key is {secret}\n\n"
           + tail)
    prompt = (doc + "\n\nQuestion: what is the exact deploy key mentioned in the config "
                    "token comment above? Reply with ONLY the key, nothing else.")
    approx_tok = len(prompt) // 4
    t0 = time.perf_counter()
    # max_tokens generous: Qwen3.6 is a reasoning model — it emits a <think> chain
    # before the answer, so a tiny budget cuts it off mid-thought. We check whether
    # the secret appears ANYWHERE in the full completion.
    r = httpx.post(f"{ui}/v1/chat/completions",
                   json={"model": served, "messages": [{"role": "user", "content": prompt}],
                         "max_tokens": 512, "temperature": 0.0}, timeout=300)
    dt = time.perf_counter() - t0
    if r.status_code >= 300:
        return {"ok": False, "err": r.text[:200], "approx_prompt_tokens": approx_tok}
    d = r.json()
    ans = (d.get("choices") or [{}])[0].get("message", {}).get("content", "")
    usage = d.get("usage", {})
    return {"ok": secret in ans, "answer": ans.strip()[-120:],
            "approx_prompt_tokens": approx_tok,
            "prompt_tokens_actual": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "wall_s": round(dt, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ui", default="http://127.0.0.1:7000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--served-name", default="qwen36")
    ap.add_argument("--maxlen", type=int, default=131072)
    ap.add_argument("--util", type=float, default=0.75)
    ap.add_argument("--needle-tokens", type=int, default=24000,
                    help="approx prompt length for the recall test (0 = skip)")
    ap.add_argument("--no-serve", action="store_true",
                    help="assume engine already serving at the target; just measure + needle")
    args = ap.parse_args()

    print(f"== context test: {args.model} @ maxlen={args.maxlen:,} util={args.util} ==")
    if not args.no_serve:
        res = _serve(args.ui, args)
        if "error" in res:
            print(f"SERVE FAILED: {res['error'][:300]}")
            sys.exit(2)
        print("  waiting for engine_ready (KV-cache sizing happens here)...")
        if not _wait_ready(args.ui):
            print("ENGINE DID NOT BECOME READY — check /api/logs (likely maxlen > KV pool).")
            sys.exit(3)

    kv = _kv_from_logs(args.ui)
    print(f"  vLLM KV cache: {kv or '(not found in logs)'}")
    if kv.get("kv_cache_tokens"):
        print(f"  => REAL ceiling: a single sequence can be up to "
              f"{kv['kv_cache_tokens']:,} tokens of context")
        if kv["kv_cache_tokens"] < args.maxlen:
            print(f"  !! pool ({kv['kv_cache_tokens']:,}) < requested maxlen ({args.maxlen:,}) "
                  f"— vLLM would have capped or refused.")

    if args.needle_tokens > 0:
        print(f"  needle test (~{args.needle_tokens:,} token prompt)...")
        nd = _needle(args.ui, args.served_name, args.needle_tokens)
        verdict = "PASS ✅" if nd.get("ok") else "FAIL ❌"
        print(f"  needle {verdict}: {json.dumps(nd)}")

    print("== done ==")


if __name__ == "__main__":
    main()
