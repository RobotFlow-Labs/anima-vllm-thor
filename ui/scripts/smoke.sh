#!/usr/bin/env bash
# End-to-end smoke test for a running anima-thor-ui + engine.
#   ./smoke.sh [base_url]        (default http://127.0.0.1:7000)
# Verifies health + all three API dialects return coherent output. Exit 0 = all pass.
set -uo pipefail
BASE="${1:-http://127.0.0.1:7000}"
pass=0; fail=0
ck(){ local name="$1" cond="$2" got="$3"
  # cond is a count/flag: pass on any non-zero integer (non-stream checks send 1;
  # streaming checks send the SSE chunk count, e.g. 66).
  if [ "${cond:-0}" -gt 0 ] 2>/dev/null; then echo "  ✓ $name"; pass=$((pass+1)); else echo "  ✗ $name  ($got)"; fail=$((fail+1)); fi; }

echo "== smoke @ $BASE =="
H=$(curl -s --max-time 5 "$BASE/healthz")
ck "healthz status ok"   "$(echo "$H" | grep -c '"status":"ok"')" "$H"
ck "engine ready"        "$(echo "$H" | grep -c '"engine_ready":true')" "$H"
MODEL=$(echo "$H" | python3 -c "import sys,json;print(json.load(sys.stdin).get('model') or 'default')" 2>/dev/null)

OAI=$(curl -s --max-time 60 "$BASE/v1/chat/completions" -H 'content-type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with: ok\"}],\"max_tokens\":256}")
ck "OpenAI /v1 coherent"   "$(echo "$OAI" | grep -c '"content"')" "$OAI"

ANT=$(curl -s --max-time 60 "$BASE/v1/messages" -H 'content-type: application/json' \
  -d "{\"model\":\"$MODEL\",\"max_tokens\":256,\"messages\":[{\"role\":\"user\",\"content\":\"Reply with: ok\"}]}")
ck "Anthropic /v1/messages" "$(echo "$ANT" | grep -c '"type":"text"')" "$ANT"

OLL=$(curl -s --max-time 60 "$BASE/api/chat" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with: ok\"}],\"stream\":false}")
ck "Ollama /api/chat"      "$(echo "$OLL" | grep -cE '"done":[[:space:]]*true')" "$OLL"
ck "Ollama /api/tags"      "$(curl -s --max-time 5 "$BASE/api/tags" | grep -c '"models"')" "tags"

# --- streaming paths (distinct code) ---
OS=$(curl -s --max-time 60 -N "$BASE/v1/chat/completions" -H 'content-type: application/json' \
  -d "{\"model\":\"$MODEL\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"count to 3\"}],\"max_tokens\":64}")
ck "OpenAI stream (SSE)"    "$(echo "$OS" | grep -c '^data: ')" "stream"
AS=$(curl -s --max-time 60 -N "$BASE/v1/messages" -H 'content-type: application/json' \
  -d "{\"model\":\"$MODEL\",\"max_tokens\":64,\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"count to 3\"}]}")
ck "Anthropic stream (SSE)" "$(echo "$AS" | grep -c 'content_block_delta')" "stream"
LS=$(curl -s --max-time 60 -N "$BASE/api/chat" \
  -d "{\"model\":\"$MODEL\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"count to 3\"}]}")
ck "Ollama stream (ndjson)" "$(echo "$LS" | grep -cE '"done":[[:space:]]*true')" "stream"

echo "== $pass passed, $fail failed =="
[ "$fail" = 0 ]
