# Benchmarks

Token/s benchmarking for the ANIMA Thor vLLM engine. All numbers are measured on a
single Jetson AGX Thor (Blackwell `sm_110a`, 128 GB unified, ~60 W) running
`anima-vllm-thor:thor-sm110-vllm0.23`.

## Files
- **`bench_openai.py`** — single-point benchmark against any OpenAI-`/v1` endpoint.
  Measures TTFT, per-stream decode tok/s, and aggregate tok/s at a given concurrency;
  appends one row to `results.tsv`. (Lives on the Thor box under `~/thor-serve/bench/`.)
- **`sweep.sh`** — full concurrency sweep (c1 → c48) against an already-running engine.
  This is the harness behind the headline aggregate number: it shows both single-stream
  latency and how continuous batching scales aggregate throughput on one box.
- **`results.tsv`** — append-only log of every run (date, engine, model, quant,
  concurrency, TTFT, decode tok/s, aggregate tok/s, wall time, ok/fail).

## Run a sweep
The engine must already be serving the model (e.g. via the ANIMA Thor UI autoserve):
```bash
cd ~/thor-serve/bench
./sweep.sh <served_name> <engine_label>
# e.g. ./sweep.sh qwen36 vllm-0.23-qwen36
```

## Methodology notes
- **Single-stream decode tok/s** excludes TTFT (prompt prefill) — it's the steady-state
  generation rate, memory-bandwidth-bound: `tok/s ≈ 273 GB/s ÷ (active_params × ~1.2 B)`.
- **Aggregate tok/s** is summed across all concurrent streams; it rises with concurrency
  until the GPU saturates, then per-stream rate falls as aggregate keeps climbing.
- `--max-tokens` is kept short at high concurrency to bound wall time; single-stream uses
  a longer generation for a clean decode-rate measurement.
