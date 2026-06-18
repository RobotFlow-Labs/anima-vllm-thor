# ANIMA Thor UI

**A control plane for the [`anima-vllm:thor-latest`](https://github.com/RobotFlow-Labs/anima-vllm-thor) engine on the NVIDIA Jetson AGX Thor.** vLLM has no web UI — this is ours, in the ANIMA Industrial-Cyberpunk skin.

No local chat. It does this, well:

| | |
|---|---|
| 🎛 **Engine control** | Pick a model + **profile** (latency / throughput) → launches the engine. **GPU util `auto`** fits free RAM (no OOM). Start / stop (graceful) / tail logs. Live **tok/s meter** + free-RAM from vLLM metrics. |
| 📦 **Model manager** | List downloaded models with sizes, delete, serve in one click. Curated **presets** (hero / throughput-747 / your quantized coder / …). |
| 🔍 **Thor-ready discovery** | Searches HF for NVFP4 models and **ranks them for *this* box** — fits-128 GB + est tok/s. Paste an exact repo → if it isn't NVFP4, a **`→ quantize`** path. Flags `ROCKS / BALANCED / SMART·SLOW / TOO BIG / UNTESTED ARCH`. |
| 🧪 **Quantize → NVFP4** | Turn any standard bf16 HF model into a Thor-ready NVFP4 build on-device (NVIDIA ModelOpt), then one-click **publish to HF**. |
| 🔌 **API + Swagger** | **OpenAI** `/v1` (also **Factory Droid**) + **Anthropic** `/v1/messages` + **Ollama** `/api/chat,tags,generate` — all translated to the engine. Drop-in for Cursor, Open WebUI, n8n, LangChain, the Anthropic/OpenAI SDKs. Docs at `/docs`. |
| ♻️ **Self-healing** | Container **auto-restarts on reboot**; **auto-serves** a default model on boot; **Reboot Thor** button reclaims leaked GPU memory — full hands-free recovery. |

No passwords (single-user edge box). HF token baked via env so downloads are fast and silent.

### Profiles (measured on Qwen3.6-35B-A3B)
- **latency** — 79 tok/s single-stream (snappiest for one user)
- **throughput** — prefix-cache + batch → **747 tok/s aggregate** @ 48 concurrent
See [`../docs/PERFORMANCE.md`](../docs/PERFORMANCE.md).

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

## Run on Thor (Docker — auto-restarts on reboot, recommended)
```bash
git clone https://github.com/RobotFlow-Labs/anima-vllm-thor && cd anima-vllm-thor/ui
# secrets come from ~/thor-serve/.env (HF_TOKEN, HF_HOME); edit if needed
docker compose up -d --build   # or: ./run_docker.sh
                               # http://<thor>:7000   ·   Swagger at /docs
```
The container ships the Docker CLI and mounts the host docker socket + HF cache (at identical
in/out paths) so it can drive the **sibling** vLLM + quantize containers. `--restart unless-stopped`
+ an enabled `docker.service` means it comes back automatically after a power-down.

To redeploy after editing code: `rsync` the `ui/` dir to the host path, then `docker restart anima-thor-ui`.

<details><summary>Bare-metal alternative (tmux, no auto-restart)</summary>

```bash
cp .env.example .env && ./run.sh   # uv venv + python -m anima_thor_ui.main
```
</details>
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
