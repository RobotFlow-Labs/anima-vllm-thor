#!/usr/bin/env bash
# Full concurrency sweep against an ALREADY-RUNNING engine (the ANIMA Thor UI's
# vLLM, port 8000 by default). Records every point to results.tsv via bench_openai.py.
#
# This is the harness that produced the hero's headline aggregate number — it walks
# concurrency c1 -> c48 so you see both single-stream latency AND how continuous
# batching scales aggregate throughput on one Thor.
#
#   ./sweep.sh <served_name> [engine_label] [base_url]
#   ./sweep.sh qwen3next vllm-0.23-qwen3next http://localhost:8000/v1
#
# Requires: bench_openai.py beside this script, `uv`, and the engine serving <served_name>.
set -uo pipefail
NAME="${1:?usage: sweep.sh <served_name> [engine_label] [base_url]}"
ENGINE="${2:-$NAME}"
BASE="${3:-http://localhost:8000/v1}"
HERE="$(cd "$(dirname "$0")" && pwd)"

# concurrency : num_prompts : max_tokens   (single stream uses long gen for clean decode_tps)
SWEEP=( "1:3:256" "8:32:128" "16:64:128" "32:128:128" "48:144:128" )

echo "== sweep $NAME on $BASE (engine=$ENGINE) =="
# fail fast if the engine isn't actually serving this model
if ! curl -s --max-time 10 "$BASE/models" | grep -q "$NAME"; then
  echo "ERROR: engine at $BASE is not serving '$NAME' (check /v1/models)"; exit 1
fi

for s in "${SWEEP[@]}"; do
  IFS=":" read -r conc np mt <<< "$s"
  echo "--- concurrency=$conc prompts=$np max_tokens=$mt ---"
  ( cd "$HERE" && uv run bench_openai.py --base-url "$BASE" --model "$NAME" \
      --engine "$ENGINE" --quant nvfp4 --concurrency "$conc" \
      --num-prompts "$np" --max-tokens "$mt" 2>&1 ) | rg -i "decode_tps_med|agg_tps|ttft_med|wall_s|ok|fail" || true
done
echo "== sweep done — tail results.tsv =="
tail -6 "$HERE/results.tsv" 2>/dev/null
