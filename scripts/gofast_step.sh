#!/usr/bin/env bash
# Risk-free go-fast A/B (see trainops notes/thor/02_vllm_optimization.md §8.0).
# Runs the stable config + ONE extra flag in an ISOLATED serve (separate vLLM cache, container
# 'anima-exp'), checks FUNCTIONAL then PERF gates, and ALWAYS tears the experiment down on exit.
# It NEVER edits serve_stable.sh or the UI default — promotion stays a manual, deliberate step.
#
#   gofast_step.sh "<label>" <model> "<one extra serve flag>"
#   e.g. gofast_step.sh prefix-cache nvidia/Qwen3.6-35B-A3B-NVFP4 "--enable-prefix-caching"
set -uo pipefail
LABEL="${1:?label}"; MODEL="${2:?model}"; EXTRA="${3:-}"
set -a; [ -f "$HOME/thor-serve/.env" ] && . "$HOME/thor-serve/.env"; set +a
: "${HF_HOME:=$HOME/thor-serve/models/huggingface}"
NAME="exp"; TSV="$HOME/thor-serve/bench/results.tsv"
cleanup(){ sudo docker rm -f anima-exp 2>/dev/null || true; }
trap cleanup EXIT
cleanup

echo "[$LABEL] serve (isolated): $MODEL  + extra: '$EXTRA'"
sudo docker run -d --name anima-exp --runtime nvidia --network host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e HF_TOKEN="${HF_TOKEN:-}" -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
  -v "$HF_HOME":/root/.cache/huggingface -v "$HOME/thor-vllm-cache-exp":/root/.cache/vllm \
  anima-vllm:thor-latest vllm serve "$MODEL" --trust-remote-code \
  --attention-backend TRITON_ATTN --gpu-memory-utilization 0.70 --kv-cache-dtype fp8 \
  --max-model-len 32768 --served-model-name "$NAME" --port 8000 $EXTRA >/dev/null

# --- wait for ready / catch hard error ---
OK=0
for i in $(seq 1 120); do
  curl -s http://localhost:8000/v1/models 2>/dev/null | grep -q "$NAME" && { OK=1; break; }
  if sudo docker logs anima-exp 2>&1 | grep -qiE "No module|not support|out of memory|CUDA error|RuntimeError|Engine core init.*fail|ValueError"; then
    echo "[$LABEL] ✕ FUNCTIONAL FAIL (engine error):"
    sudo docker logs anima-exp 2>&1 | grep -iE "Error|not support|memory|module|Invalid" | tail -6
    echo -e "$LABEL\t$MODEL\t$EXTRA\t-\t-\tFUNC_FAIL" >> "$TSV"; exit 1
  fi
  sleep 15
done
[ "$OK" = 1 ] || { echo "[$LABEL] ✕ FUNCTIONAL FAIL (timeout)"; echo -e "$LABEL\t$MODEL\t$EXTRA\t-\t-\tTIMEOUT" >> "$TSV"; exit 1; }

# --- FUNCTIONAL gate: coherent, non-empty output ---
ANS=$(curl -s http://localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d "{\"model\":\"$NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 2+2? Reply with only the number.\"}],\"max_tokens\":10,\"temperature\":0}" \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null)
echo "[$LABEL] smoke answer: '$ANS'"
if ! echo "$ANS" | grep -q "4"; then
  echo "[$LABEL] ✕ FUNCTIONAL FAIL (incoherent output — rejecting regardless of speed)"
  echo -e "$LABEL\t$MODEL\t$EXTRA\t-\t-\tINCOHERENT" >> "$TSV"; exit 1
fi
echo "[$LABEL] ✓ FUNCTIONAL GATE PASS"

# --- PERF gate ---
cd "$HOME/thor-serve/bench"
S=$(uv run bench_openai.py --base-url http://localhost:8000/v1 --model "$NAME" --engine "$LABEL" --quant nvfp4 --concurrency 1 --num-prompts 4 --max-tokens 256 2>&1 | rg -oP "decode_tps_med\s*:\s*\K[0-9.]+" | head -1)
A=$(uv run bench_openai.py --base-url http://localhost:8000/v1 --model "$NAME" --engine "$LABEL" --quant nvfp4 --concurrency 8 --num-prompts 24 --max-tokens 128 2>&1 | rg -oP "agg_tps\s*:\s*\K[0-9.]+" | head -1)
echo "[$LABEL] PERF: single=${S:-?} tok/s  agg@8=${A:-?} tok/s"
echo -e "$LABEL\t$MODEL\t$EXTRA\t${S:-?}\t${A:-?}\tEXPERIMENT" >> "$TSV"
echo "[$LABEL] DONE. Compare single=${S:-?} vs BASELINE. Promote into serve_stable.sh + UI ONLY if it passes (no regression + target improved). Box is back on golden config."
