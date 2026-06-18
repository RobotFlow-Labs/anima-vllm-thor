# Changelog

## v0.2.0 — 2026-06-18
The self-healing, multi-API, tested release.

**Engine**
- Published image `ilessio/anima-vllm-thor:thor-sm110-vllm0.23` — vLLM 0.23 + PyTorch 2.11 + CUDA 13,
  native `sm_110a`. Loads NVFP4 MoE models NVIDIA's 0.19 Jetson image can't.
- Measured: Qwen3.6-35B-A3B-NVFP4 = **79 tok/s single / 747 aggregate** (48 concurrent), ~60 W.

**Control plane (`ui/`)**
- **Triple API:** OpenAI `/v1` (+ Factory Droid), Anthropic `/v1/messages`, Ollama `/api/chat,tags,generate`
  — verified incl. tool-calling; clean "loading"/"offline" errors; optional `ANIMA_API_KEY` auth.
- **Self-healing:** containerized auto-restart UI · auto-serve last model on boot (3× retry) ·
  `wait_mem_stable` · model-aware memory guard + low-floor `util:auto` · one-click reboot-to-reclaim.
  Recovered a fully-leaked box to a serving hero **without a reboot**.
- **Perf/UX:** latency (79) / throughput (747) profiles · live tok/s + queue meter · curated presets ·
  copy-buttons · favicon · served-model recovery after restart.
- **On-device quantize → NVFP4 → publish** (NVIDIA ModelOpt) — first model published to HF.

**Quality / project**
- 16 unit tests + CI (ruff + pytest, green) · `/healthz` + request logs · `scripts/smoke.sh`
  (health + 3 dialects) · Makefile · docker-compose · `.dockerignore`.
- Docs: `PERFORMANCE.md`, `MODELS.md` (compatibility), GitHub Pages landing site, `CONTRIBUTING.md`,
  `SECURITY.md`, Apache-2.0 `LICENSE`.

**Known limits (Thor):** stopping a served model leaks its GPU memory (only a reboot reclaims) →
**one model per boot**. Quantize ceiling ~27 B / ~54 GB bf16. Dense Qwen3.6 (`qwen3_5_text`) needs
vLLM 0.24 to serve; VLM path is flaky.

## v0.1.0
Initial: latest-vLLM Thor image + recipe (9 build gotchas) + first control UI.
