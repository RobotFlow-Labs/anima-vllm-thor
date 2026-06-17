# ANIMA Thor UI

**A control plane for the [`anima-vllm:thor-latest`](https://github.com/RobotFlow-Labs/anima-vllm-thor) engine on the NVIDIA Jetson AGX Thor.** vLLM has no web UI — this is ours, in the ANIMA Industrial-Cyberpunk skin.

No local chat. It does four things, well:

| | |
|---|---|
| 🎛 **Engine control** | Pick a model + vLLM config (context, GPU util, KV dtype, attention backend, spec-decode) → launches the `anima-vllm:thor-latest` container. Start / stop / tail logs. |
| 📦 **Model manager** | List downloaded models with sizes, delete them, serve them in one click. |
| 🔍 **Thor-ready discovery** | Searches HuggingFace for NVFP4 models and **ranks them for *this* box** — fits-in-128 GB check + estimated decode tok/s from active params. Flags `ROCKS` / `BALANCED` / `SMART·SLOW` / `TOO BIG` / `UNTESTED ARCH`. |
| 🔌 **API + Swagger** | OpenAI `/v1` (also **Factory Droid**-compatible) and **Anthropic** `/v1/messages` (translated), documented at `/docs`. |

No passwords (single-user edge box). HF token baked via env so downloads are fast and silent.

## How it ranks models for Thor
Decode is memory-bandwidth-bound: `tok/s ≈ 273 GB/s ÷ (active_params × 1.2 bytes)`, calibrated to our measured
Qwen3.6-A3B (79) and Nano-A3B (68). A model is **served-ready** when NVFP4 weights (`params × 0.55`) fit under
the 95 GB budget *and* its architecture is one we've verified on vLLM 0.23 / sm_110a. Example output:

```
ROCKS         nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4    44.0GB  est 75tps  qwen3_next
BALANCED      nvidia/Gemma-4-26B-A4B-NVFP4                14.3GB  est 56tps  gemma4
SMART/SLOW    nvidia/NVIDIA-Nemotron-3-Super-120B-A12B    66.0GB  est 18tps  nemotron_h
TOO BIG       MiniMax-M3 (427B)                          235GB   —          (won't fit)
```

## Run on Thor
```bash
git clone https://github.com/RobotFlow-Labs/anima-vllm-thor && cd anima-vllm-thor/ui
cp .env.example .env        # set HF_TOKEN + HF_HOME
./run.sh                    # http://<thor>:7000   ·   Swagger at /7000/docs
```
Needs the `anima-vllm:thor-latest` image present and direct access to the host docker daemon
(it manages the engine container the same way `serve.sh` does).

## Endpoints
| Use | URL |
|---|---|
| OpenAI SDK / **Factory Droid** | `http://<thor>:7000/v1` |
| Anthropic SDK (`base_url`) | `http://<thor>:7000/v1/messages` |
| Swagger / OpenAPI | `http://<thor>:7000/docs` |

## Stack
FastAPI + httpx · vanilla HTML/CSS/JS (no build step, offline-friendly) · drives the engine via the `docker` CLI.
ANIMA design system: Industrial Orange `#FF3B00`, void black, Oswald / JetBrains Mono / Chakra Petch.

Built by [RobotFlow Labs](https://github.com/RobotFlow-Labs) for the ANIMA edge-AI stack.
