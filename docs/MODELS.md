# Model compatibility on Jetson Thor (anima-vllm-thor)

What runs, what doesn't, and how to pick — distilled from real measurements on a 128 GB Jetson AGX Thor
(`sm_110a`, vLLM 0.23, NVFP4). The UI's **Discover** tab applies all of this automatically and flags each
model `ROCKS / BALANCED / SMART·SLOW / TOO BIG / UNTESTED ARCH / QUANTIZE→NVFP4`.

## The two limits
1. **Serving fits in unified memory.** ~63 GB is reserved for the GPU, leaving **~62 GB at boot** for
   weights + KV (more with `util:auto` on a fresh box, up to ~100 GB usable). NVFP4 weights ≈ `params × 0.55`.
2. **Decode is memory-bandwidth-bound.** `tok/s ≈ 273 GB/s ÷ (active_params × ~1.2)`. So **MoE with low
   active params wins**: a 35B-A3B does ~79 tok/s; a dense 32B does ~7.

## Context length — 128K is the coding default, not 32K
The Qwen hero uses aggressive GQA (**2 KV heads**, head_dim 256, 40 layers) → KV is only ~40 KB/token.
**Measured on Thor** (Qwen3.6-35B-A3B-NVFP4, fp8 KV, util 0.62): vLLM allocates a **64.8 GiB / 6.44 M-token
KV pool** — enough for **49 concurrent 128K-context sequences**. So 32768 was leaving ~90% of capability
on the table. The UI now defaults coding/latency presets to **131072 (128K)**; native max is 262144 (256K),
which still allows ~24× concurrency.

- **Verified, not just loaded:** a needle-in-haystack recall at a **59,042-token** prompt (key buried at
  60% depth, ~2× the old 32K ceiling) returns the exact key. `bench/context_test.py` reproduces this.
- **Cost of long context ≈ free here** thanks to GQA — raise `max_model_len`, not util. Keep **throughput**
  profile lower (32–64K): 48-way continuous batching multiplies KV demand.
- **Cold-start caveat:** first serve at 128K does weight-load + KV-profile + CUDA-graph capture (~6.5 min on
  the 35B); the CUDA-graph cache makes later boots fast. Autoserve waits up to ~12 min for this.

## Architectures
| Arch | Serve on vLLM 0.23? | Quantize (ModelOpt)? | Notes |
|---|---|---|---|
| `qwen3_5_moe` | ✅ native | ✅ | **best** — Qwen3.6-35B-A3B is the hero (79/747) |
| `qwen3_moe` | ✅ native | ✅ | Qwen3-30B-A3B, Qwen3-Coder-30B-A3B |
| `nemotron_h` | ✅ native | ⚠️ hybrid (Mamba2) — use pre-made NVFP4 | Nemotron-Nano-30B-A3B = 68/240 |
| `qwen2`, `llama`, `mistral`, `gemma2/3` | ✅ native | ✅ | Qwen2.5-Coder, Llama-3.x, etc. |
| `qwen3_next` | ✅ native | ❌ hybrid (DeltaNet) | works but **slow** (33 tok/s) — linear-attn on Triton |
| **`qwen3_5_text`** (dense Qwen3.5/3.6) | ❌ **needs vLLM 0.24** | ✅ | quantizes fine, but our 0.23 can't serve it |
| `gemma4` | ❌ (too new for 0.23.0) | — | failed engine init |
| VLM / `*_vl` (Holo, MiniMax-M3-VL) | ⚠️ Jetson VLM path flaky | — | text-mode unreliable |

## Quantize-it-yourself ceiling
Calibration needs the model **in bf16** resident → fits up to **~27 B / ~54 GB bf16** on Thor.
30 B+ (Qwen3-Coder-30B = 61 GB) **OOMs** — use a pre-made NVFP4 build or a ≤27 B model. Standard archs
(`qwen2/qwen3/llama/mistral`) quantize cleanly; hybrids (`nemotron_h`, `qwen3_next`) don't.

## Recommended (all NVFP4, fit Thor)
| Model | single / agg | use |
|---|---|---|
| **Qwen3.6-35B-A3B** | 79 / 747 | general/coding hero |
| Nemotron-Nano-30B-A3B | 68 / 240 | reliable general |
| Qwen2.5-Coder-14B (quantize yourself) | — | coding, fits with margin |
| Qwen3-Next-80B-A3B | 34 / 150 | bigger brain, slower |

## Measured NVFP4 weight sizes — the real fit boundary
Actual on-disk weights, measured on Thor. The serving ceiling is **~62 GB** (weights + KV), so the
boundary sits between the 80B (fits) and the 120B (doesn't):

| Model | NVFP4 weights | Serves on vLLM 0.23? |
|---|---|---|
| Gemma-4-26B-A4B | 18 GB | ❌ — fits in memory, but `gemma4` arch unsupported |
| Nemotron-Nano-30B-A3B | 19 GB | ✅ |
| **Qwen3.6-35B-A3B** (hero) | 22 GB | ✅ |
| Qwen3-Next-80B-A3B | **48 GB** | ✅ — largest that fits |
| Nemotron-Super-120B-A12B | **75 GB** | ❌ — exceeds the ~62 GB ceiling |

So "won't fit" isn't just the frontier giants — it kicks in around **~60 GB of NVFP4 weights** on this box.

## Too big for Thor (frontier, multi-GPU class)
MiniMax-M3 (427 B), DeepSeek-V4 (671 B), Kimi-K2.x (~1 T), GLM-5.x, Qwen3-235B-A22B — won't fit; the UI
flags them `TOO BIG`.

## Operational reality
Stopping a served model **leaks its GPU memory** (Jetson driver doesn't reclaim the dead CUDA context;
reclaims slowly). **Rule: one model per boot; to swap, reboot** — the UI auto-serves the last model after.
The model-aware guard + `util:auto` let small models still serve on a partly-leaked box.
