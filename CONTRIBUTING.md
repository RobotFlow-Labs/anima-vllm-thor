# Contributing

Thanks for helping make edge inference better! This repo has two parts:

- **the engine image** ([`Dockerfile`](Dockerfile) + [`scripts/`](scripts)) — latest vLLM compiled
  native for the Jetson Thor (`sm_110a`). The 9 build gotchas are documented inline.
- **the control plane** ([`ui/`](ui)) — FastAPI + a vanilla SPA that drives the engine and exposes
  OpenAI / Anthropic / Ollama APIs. **This is where most contributions land.**

## Dev setup (the UI)
```bash
cd ui
make install        # uv venv + deps + pytest + ruff
make test           # 16 unit tests — no engine/GPU needed
make lint           # ruff
```
CI runs `ruff` + `pytest` on every `ui/**` change; please keep both green.

## Run / deploy
```bash
make run            # docker compose up -d --build (local)
make deploy THOR=thor   # rsync to a Thor + restart the UI container (engine survives)
make smoke BASE=http://<thor>:7000   # end-to-end check (health + 3 API dialects)
```

## Things to know before you change behavior
- **One model per boot.** On Thor, stopping a served model leaks its GPU memory regardless of how it's
  stopped; only a reboot reclaims it. Serve logic must `wait_mem_stable()` and use `util:auto`. See
  [`docs/MODELS.md`](docs/MODELS.md).
- **Arch support is curated** (`hf_models.SUPPORTED_ARCHS`, `quantize.QUANTIZABLE_ARCHS`). If you verify
  a new arch serves/quantizes on `sm_110a`, add it *with the measurement* in the PR.
- **Memory math** lives in `vllm_manager.auto_util` / `model_size_gb` / the serve guard — keep it
  model-size-aware so small models still serve on a partly-leaked box.
- **Pure logic gets a unit test** (`ui/tests/test_core.py`); handlers/translation too where feasible.

## PRs
Small, focused, green CI, a one-line measurement if you touched perf/compat. Be honest in benchmarks —
report what helped *and* what didn't (it's the house style).
