# Model compatibility on Jetson Thor (anima-vllm-thor)

What runs, what doesn't, and how to pick — distilled from real measurements on a 128 GB Jetson AGX Thor
(`sm_110a`, vLLM 0.23, NVFP4). The UI's **Discover** tab applies all of this automatically and flags each
model `ROCKS / BALANCED / SMART·SLOW / TOO BIG / UNTESTED ARCH / QUANTIZE→NVFP4`.

## The two limits
1. **Serving fits in unified memory.** ~63 GB is reserved for the GPU, leaving **~62 GB at boot** for
   weights + KV (more with `util:auto` on a fresh box, up to ~100 GB usable). NVFP4 weights ≈ `params × 0.55`.
2. **Decode is memory-bandwidth-bound.** `tok/s ≈ 273 GB/s ÷ (active_params × ~1.2)`. So **MoE with low
   active params wins**: a 35B-A3B does ~79 tok/s; a dense 32B does ~7.

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

## Too big for Thor (frontier, multi-GPU class)
MiniMax-M3 (427 B), DeepSeek-V4 (671 B), Kimi-K2.x (~1 T), GLM-5.x, Qwen3-235B-A22B — won't fit; the UI
flags them `TOO BIG`.

## Operational reality
Stopping a served model **leaks its GPU memory** (Jetson driver doesn't reclaim the dead CUDA context;
reclaims slowly). **Rule: one model per boot; to swap, reboot** — the UI auto-serves the last model after.
The model-aware guard + `util:auto` let small models still serve on a partly-leaked box.
