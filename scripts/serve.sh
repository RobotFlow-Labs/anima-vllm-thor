#!/usr/bin/env bash
# Canonical vLLM launcher for Thor — graceful teardown, GPU-free wait, full DEBUG logging.
# Override via env: IMAGE MODEL SERVED ATTN GPU_UTIL MAXLEN MAXSEQS NAME PORT EXTRA KVDTYPE
set -uo pipefail
set -a; . "$HOME/thor-serve/.env"; set +a

IMAGE="${IMAGE:-ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor}"
MODEL="${MODEL:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4}"
SERVED="${SERVED:-nemotron-nano}"
ATTN="${ATTN:-FLASHINFER}"
GPU_UTIL="${GPU_UTIL:-0.70}"
MAXLEN="${MAXLEN:-32768}"
MAXSEQS="${MAXSEQS:-8}"
KVDTYPE="${KVDTYPE:-fp8}"
NAME="${NAME:-vllm}"
PORT="${PORT:-8000}"
EXTRA="${EXTRA:-}"
TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="$HOME/thor-serve/logs"; mkdir -p "$LOGDIR"
LOG="$LOGDIR/${NAME}_${TS}.log"
ln -sf "$LOG" "$LOGDIR/${NAME}_latest.log"

echo "[serve] $(date) image=$IMAGE model=$MODEL attn=$ATTN util=$GPU_UTIL maxlen=$MAXLEN" | tee -a "$LOG"

# --- graceful teardown so the GPU actually frees (the 3s-rush bug) ---
echo "[serve] stopping any existing '$NAME'..." | tee -a "$LOG"
sudo docker stop -t 30 "$NAME" >/dev/null 2>&1 || true
sudo docker rm -f "$NAME" >/dev/null 2>&1 || true

# --- wait until >=100 GiB free on the unified GPU pool ---
echo "[serve] waiting for GPU memory to free (need >=60GiB)..." | tee -a "$LOG"
for i in $(seq 1 40); do
  FREE=$(sudo docker run --rm --runtime nvidia "$IMAGE" \
        python -c "import torch;f,_=torch.cuda.mem_get_info();print(int(f/1024**3))" 2>/dev/null | tail -1)
  FREE="${FREE:-0}"
  echo "[serve]   free=${FREE}GiB (try $i)" | tee -a "$LOG"
  case "$FREE" in (*[!0-9]*) FREE=0;; esac
  [ "$FREE" -ge 60 ] && break
  sleep 4
done

echo "[serve] launching vLLM ($NAME) on :$PORT — log: $LOG" | tee -a "$LOG"
exec sudo docker run --rm --name "$NAME" --network host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 --runtime nvidia \
  -e HF_TOKEN="$HF_TOKEN" \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
  -e VLLM_LOGGING_LEVEL=DEBUG \
  -v "$HOME/thor-serve/models/huggingface:/root/.cache/huggingface" \
  -v "$HOME/thor-vllm-cache:/root/.cache/vllm" \
  "$IMAGE" \
  vllm serve "$MODEL" --trust-remote-code \
    --attention-backend "$ATTN" \
    --gpu-memory-utilization "$GPU_UTIL" --kv-cache-dtype "$KVDTYPE" \
    --max-model-len "$MAXLEN" --max-num-seqs "$MAXSEQS" \
    --served-model-name "$SERVED" --port "$PORT" $EXTRA 2>&1 | tee -a "$LOG"
