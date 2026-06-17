#!/usr/bin/env bash
# GOLDEN baseline launcher for Thor — the KNOWN-GOOD config (79/294 on A3B).
# This is the rollback target. DO NOT edit until a go-fast step (gofast_step.sh) has PASSED
# both gates and you are deliberately promoting it. Container name: anima-vllm.
set -euo pipefail
MODEL="${1:?usage: serve_stable.sh <hf-repo-or-path> [served_name]}"
NAME="${2:-$(basename "$MODEL")}"
set -a; [ -f "$HOME/thor-serve/.env" ] && . "$HOME/thor-serve/.env"; set +a
: "${HF_HOME:=$HOME/thor-serve/models/huggingface}"

sudo docker rm -f anima-vllm 2>/dev/null || true
sudo docker run -d --name anima-vllm --runtime nvidia --network host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e HF_TOKEN="${HF_TOKEN:-}" -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
  -v "$HF_HOME":/root/.cache/huggingface -v "$HOME/thor-vllm-cache":/root/.cache/vllm \
  anima-vllm:thor-latest vllm serve "$MODEL" --trust-remote-code \
  --attention-backend TRITON_ATTN --gpu-memory-utilization 0.70 --kv-cache-dtype fp8 \
  --max-model-len 32768 --served-model-name "$NAME" --port 8000 >/dev/null
echo "[stable] launching $MODEL as '$NAME' (TRITON_ATTN · fp8 KV · util 0.70 · no spec)"
echo "[stable] watch: sudo docker logs -f anima-vllm | grep -i 'startup complete'"
