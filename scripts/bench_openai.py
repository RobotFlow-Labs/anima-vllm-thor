# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx>=0.27"]
# ///
"""Unified OpenAI-/v1 token/s benchmark + recorder for Thor.
Works against ANY OpenAI-compatible endpoint: vLLM, llama.cpp/cyber-inference, SGLang.
Measures TTFT, decode tok/s (per stream), total throughput at concurrency, appends to results.tsv.

Usage:
  uv run bench_openai.py --base-url http://localhost:8000/v1 --model <id> \
      --engine vllm --quant nvfp4 --concurrency 8 --num-prompts 32 --max-tokens 256
"""
import argparse, asyncio, time, json, statistics, os, datetime
import httpx

PROMPT = ("You are a helpful assistant. Write a detailed, technical explanation of how "
          "speculative decoding (MTP / EAGLE-3) speeds up LLM inference, including the "
          "draft-and-verify loop, acceptance rate, and why it helps memory-bound decode. "
          "Be thorough and precise.")

async def one_request(client, base_url, model, max_tokens):
    t0 = time.perf_counter(); ttft = None; n_tok = 0
    payload = {"model": model, "messages": [{"role": "user", "content": PROMPT}],
               "max_tokens": max_tokens, "temperature": 0.7, "stream": True}
    async with client.stream("POST", f"{base_url}/chat/completions", json=payload) as r:
        async for line in r.aiter_lines():
            if not line or not line.startswith("data:"): continue
            data = line[5:].strip()
            if data == "[DONE]": break
            try: obj = json.loads(data)
            except Exception: continue
            delta = obj.get("choices", [{}])[0].get("delta", {}).get("content")
            if delta:
                if ttft is None: ttft = time.perf_counter() - t0
                n_tok += 1
    dur = time.perf_counter() - t0
    decode_tps = (n_tok - 1) / (dur - ttft) if (ttft and n_tok > 1 and dur > ttft) else 0.0
    return {"ttft": ttft or dur, "n_tok": n_tok, "dur": dur, "decode_tps": decode_tps}

async def run(args):
    limits = httpx.Limits(max_connections=args.concurrency + 4)
    async with httpx.AsyncClient(timeout=600, limits=limits) as client:
        sem = asyncio.Semaphore(args.concurrency)
        async def task():
            async with sem: return await one_request(client, args.base_url, args.model, args.max_tokens)
        wall0 = time.perf_counter()
        results = await asyncio.gather(*[task() for _ in range(args.num_prompts)], return_exceptions=True)
        wall = time.perf_counter() - wall0
    ok = [r for r in results if isinstance(r, dict) and r["n_tok"] > 0]
    if not ok:
        print("ALL REQUESTS FAILED:", results[:2]); return
    ttfts = [r["ttft"] for r in ok]; decs = [r["decode_tps"] for r in ok]
    total_out = sum(r["n_tok"] for r in ok)
    agg_tps = total_out / wall
    row = {
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "engine": args.engine, "model": args.model, "quant": args.quant,
        "concurrency": args.concurrency, "num_prompts": args.num_prompts,
        "max_tokens": args.max_tokens,
        "ttft_med_s": round(statistics.median(ttfts), 4),
        "decode_tps_med": round(statistics.median(decs), 2),
        "decode_tps_mean": round(statistics.mean(decs), 2),
        "agg_tps": round(agg_tps, 2), "wall_s": round(wall, 2),
        "ok": len(ok), "fail": len(results) - len(ok),
    }
    print("\n=== RESULT ===")
    for k, v in row.items(): print(f"{k:16}: {v}")
    ledger = os.path.join(os.path.dirname(__file__), "results.tsv")
    new = not os.path.exists(ledger)
    with open(ledger, "a") as f:
        if new: f.write("\t".join(row.keys()) + "\n")
        f.write("\t".join(str(v) for v in row.values()) + "\n")
    print(f"\nappended -> {ledger}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--engine", default="vllm")
    p.add_argument("--quant", default="nvfp4")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--num-prompts", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=256)
    asyncio.run(run(p.parse_args()))
