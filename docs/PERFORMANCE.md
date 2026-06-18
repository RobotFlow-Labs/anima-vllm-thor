<p align="center"><img src="../assets/bench_chart.png" alt="ANIMA Thor benchmarks — 747 tok/s aggregate" width="100%"></p>

# ANIMA vLLM Thor — Performance

All figures **measured on real hardware**, honestly — one NVIDIA Jetson AGX Thor (Blackwell `sm_110a`,
128 GB unified, MAXN ~60 W), vLLM 0.23, NVFP4, fully offline. Model: `nvidia/Qwen3.6-35B-A3B-NVFP4`
(35 B total, 3 B active MoE). Bench: `scripts/bench_openai.py` (2026-06-18).

## TL;DR
**A 35-billion-parameter coding model, served from a 128 GB box that fits in a backpack and runs at
~60 W — fully offline, behind the OpenAI *and* Anthropic APIs. ~80 tok/s to a single user, and up to
747 tok/s aggregate across concurrent users.** No cloud, no data center, no per-token bill.

## Two profiles (pick per workload)
| Profile | Config | Single-stream | Peak aggregate | Best for |
|---|---|---|---|---|
| **Low-latency** | TRITON_ATTN · fp8 KV | **79 tok/s** | 294 @ 8 | one user, snappiest replies |
| **High-throughput** | + prefix-caching + `max-num-seqs 48` | 65 tok/s | **747 @ 48** | many users / agents, shared context |

## Aggregate scaling (high-throughput profile)
| Concurrency | per-stream tok/s | **aggregate tok/s** | TTFT |
|---|---|---|---|
| 1 | 65 | 46 | 0.07 s |
| 8 | 42 | 284 | 0.19 s |
| 16 | 32 | 466 | 0.32 s |
| 32 | 22 | 659 | 0.49 s |
| **48** | 17 | **747** | 0.75 s |

## Honest framing (why these numbers are what they are)
- **Single-stream is memory-bandwidth-bound:** ~273 GB/s ÷ ~3 B active params ⇒ a hard ~79 tok/s ceiling.
  No trick beats that for one stream of fresh tokens.
- **Aggregate scales with continuous batching** — the same bandwidth feeds many streams, so *total*
  throughput climbs to **747 tok/s @ 48** even as per-stream naturally drops (shared bandwidth).
- **Prefix caching** trades ~14 tok/s of single-stream for huge *effective* speedups on shared context
  (system prompts, repo context, agent loops) — it skips prefill on a cache hit. Worth it for agents/code.
- **What didn't help on this box:** full CUDA-graph capture and spec-decode (ngram/MTP) didn't reliably
  beat the baseline on general prompts on `sm_110a` (the fused MoE-FP4 kernel is SM100-gated → Marlin
  fallback). We report it straight rather than cherry-pick.

## vs NVIDIA's official Jetson image
NVIDIA's `nvidia-ai-iot/vllm:latest-jetson-thor` ships vLLM 0.19, which **can't even load**
`nvidia/Qwen3.6-35B-A3B-NVFP4` (`KeyError: w2_input_scale`). This image (vLLM 0.23) loads and serves it.

## The model lineup (NVFP4, measured)
| Model | single / aggregate | note |
|---|---|---|
| Qwen3.6-35B-A3B | 79 / 747 | hero · the numbers above |
| Nemotron-Nano-30B-A3B | 68 / 240 | reliable general |
| Qwen3-Next-80B-A3B | 34 / 150 | 80 B brain, hybrid = slower |
| Qwen2.5-Coder-14B (ours) | quantized + served on-device → [HF](https://huggingface.co/ilessio-aiflowlab/Qwen2.5-Coder-14B-Instruct-NVFP4-anima) | proof of the quantize→serve→publish loop |

## Reproduce
```bash
docker pull ilessio/anima-vllm-thor:thor-sm110-vllm0.23
# low-latency:
vllm serve nvidia/Qwen3.6-35B-A3B-NVFP4 --trust-remote-code --attention-backend TRITON_ATTN \
  --gpu-memory-utilization 0.6 --kv-cache-dtype fp8 --max-model-len 32768
# high-throughput (aggregate):
#   + --enable-prefix-caching --max-num-seqs 48 --max-num-batched-tokens 8192
```
