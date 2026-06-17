# ANIMA vLLM Thor — Latest vLLM on NVIDIA Jetson AGX Thor (sm_110a)

**A reproducible recipe + prebuilt image to run the latest vLLM (0.23.x) on the NVIDIA Jetson AGX Thor
(Blackwell, compute capability `sm_110a`), built on JetPack 7 / CUDA 13 / arm64.**

> Why this exists: as of June 2026 there is **no prebuilt latest-vLLM image for Jetson Thor**. NVIDIA's
> `ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor` ships **vLLM 0.19.0**, which **cannot load** modern
> NVFP4 MoE checkpoints like `nvidia/Qwen3.6-35B-A3B-NVFP4` (fails with
> `KeyError: layers.0.mlp.experts.w2_input_scale` in its ModelOpt loader) and lacks the newer
> speculative-decoding paths. This image upgrades the stack to **vLLM 0.23.x + PyTorch 2.11 + CUDA 13**,
> compiled natively for `sm_110a`, so those models load and run on the GPU.

## What you get
- **vLLM 0.23.x** (`0.23.1.dev0`) · **PyTorch 2.11.0+cu130** · **CUDA 13.0** · **flashinfer** · arm64 / SBSA
- Native `sm_110a` kernels (verified with a real GPU matmul — see `scripts/verify_sm110.py`)
- Loads **NVFP4 MoE** models the 0.19 image can't: Nemotron-3 (Nano/Super), **Qwen3.6-35B-A3B-NVFP4**
- `TRITON_ATTN` attention (the reliable backend for NVFP4 MoE on `sm_110`; the fused FP4-MoE kernel is
  SM100-gated and falls back to Marlin on Thor)
- Speculative decoding available: **ngram** (no draft model — universal), MTP (when the checkpoint has
  heads), EAGLE-3 (with a draft checkpoint)
- OpenAI-compatible `/v1` server

## Quick start (on a Jetson AGX Thor)
```bash
docker pull ilessio/anima-vllm-thor:thor-sm110-vllm0.23     # linux/arm64, Thor only
docker run --rm -it --runtime nvidia --network host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v $HOME/.cache/vllm:/root/.cache/vllm \
  ilessio/anima-vllm-thor:thor-sm110-vllm0.23 \
  vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 \
    --trust-remote-code --attention-backend TRITON_ATTN \
    --gpu-memory-utilization 0.6 --kv-cache-dtype fp8 \
    --max-model-len 200000 --port 8000
```
Use `--runtime nvidia` (not `--gpus all`) on Jetson.

## The 8 gotchas that make this build work (the hard part — all in the Dockerfile)
1. **Unset `PIP_CONSTRAINT`.** The NVIDIA base bakes `PIP_CONSTRAINT=torch==2.10.0`; it propagates even into
   a fresh venv and blocks torch 2.11 with `ResolutionImpossible`. → `ENV PIP_CONSTRAINT=""`.
2. **Uninstall vLLM 0.19 *before* upgrading torch** — vLLM 0.19 pins `torch==2.10.0`. → `pip uninstall -y vllm flashinfer-*` first.
3. **`torch 2.11.0+cu130` aarch64 official wheel HAS `sm_110` kernels** (`download.pytorch.org/whl/cu130/`) —
   verified with a real CUDA matmul (`get_device_capability()==(11,0)`). No need to build PyTorch from source.
4. **Build vLLM 0.23 from source** against that torch via `use_existing_torch.py`, `TORCH_CUDA_ARCH_LIST=11.0a`,
   `FLASHINFER_CUDA_ARCH_LIST=11.0a` (~30–50 min, 356 CUDA objects).
5. **`LD_PRELOAD` the real driver libcuda** — `_C_stable_libtorch.abi3.so` has an unresolved
   `cuTensorMapEncodeTiled`; preload `/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1` (final `ENV`, runtime-only).
6. **Reset `WORKDIR /` + remove `/build`** — else the cloned source tree shadows the installed wheel and
   `import vllm._C` fails (`No module named 'vllm._C'`).
7. **Upgrade `compressed-tensors` to 0.17.x** — base's is stale (`No module named compressed_tensors.compressors.pack_quantized`).
8. **Guard the optional quant backends** (`humming`, `inc`, `fp_quant`, `torchao`, `quark`, `deepseek_v4`,
   `mxfp4`) — vLLM 0.23 eagerly imports *all* quant configs; these aren't on arm64 and break NVFP4 too. We
   wrap them in try/except. Also: `flashinfer-python==0.6.8` (0.6.12 isn't published for arm64 yet — works fine).

Full recipe in [`Dockerfile`](./Dockerfile); the quant-guard patch in [`scripts/patch_quant.py`](./scripts/patch_quant.py).

## Build it yourself
```bash
DOCKER_BUILDKIT=1 docker build -t anima-vllm:thor-latest -f Dockerfile .
# ~30–50 min compile (356 CUDA objects). Needs ~90 GB RAM free (don't run while serving a model).
# If OOM: lower MAX_JOBS (default 12 → 8).
```
Verify sm_110 first (fast, ~2 min): `scripts/verify_sm110.py` inside the image (or any torch-2.11 venv).

## Benchmarks — measured on this image (Jetson AGX Thor, sm_110a, MAXN, vLLM 0.23.1, TRITON_ATTN, fp8 KV)
> Measured with `scripts/bench_openai.py` (2026-06-17).

| Model | quant | ctx | per-stream tok/s | aggregate (conc 8) | TTFT | notes |
|---|---|---|---|---|---|---|
| Nemotron-3-Nano-30B-A3B | NVFP4 | 8K | **67.7** | **240** | 0.09 s | matches vLLM 0.19 → **no regression** |
| **Qwen3.6-35B-A3B** | NVFP4 | 32K | **79.4** | **294** | 0.07 s | **vLLM 0.19 CANNOT load this** (`KeyError: w2_input_scale`) — the whole point |

Decode is memory-bandwidth-bound (273 GB/s, 3B active per token). These are without speculative decoding;
**ngram / MTP spec-decode is the lever past 100 tok/s** (`--speculative-config '{"method":"ngram",...}'`).
Big context is ~free over 100K (KV reserved, not read) until the conversation actually fills it.

## Hardware / software target
Jetson AGX Thor · Blackwell `sm_110a` · 128 GB unified LPDDR5X (~273 GB/s) · JetPack 7 / L4T r38.4 ·
CUDA 13.0.88 · Ubuntu 24.04 · arm64 (SBSA). Built FROM `ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor`.

## License & attribution
vLLM is Apache-2.0; PyTorch is BSD. This image is built **FROM** NVIDIA's base container and includes
NVIDIA CUDA components — redistribution of the **image** is subject to the NVIDIA base-container / CUDA
license terms. If you only want the unambiguous parts, **build from the `Dockerfile` here** (the recipe is
the shareable contribution). See `NOTICE`.

## Credits
Built by [RobotFlow Labs / AIFLOW LABS](https://github.com/RobotFlow-Labs) for the ANIMA edge-AI stack.
Base image: NVIDIA `nvidia-ai-iot/vllm`. Inspiration for the control-plane UX: `RamboRogers/cyber-inference`.
