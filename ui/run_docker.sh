#!/usr/bin/env bash
# Build + run anima-thor-ui as an auto-restarting Docker container on Thor.
# Survives reboots (--restart unless-stopped + docker.service enabled on boot).
# Run this ON Thor from the deployed app dir (e.g. ~/anima-thor-ui).
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"          # host path to the app (same path used inside)
: "${HF_HOME:=$HOME/thor-serve/models/huggingface}"
ENV_FILE="$HOME/thor-serve/.env"
NAME="anima-thor-ui"

echo "[ui] building image…"
sudo docker build -t anima-thor-ui:latest -f "$APP_DIR/Dockerfile" "$APP_DIR"

echo "[ui] (re)starting container with restart policy…"
sudo docker rm -f "$NAME" 2>/dev/null || true

# ensure the docker service itself comes up on boot (so the restart policy can fire)
sudo systemctl enable docker >/dev/null 2>&1 || true

sudo docker run -d --name "$NAME" \
  --restart unless-stopped \
  --network host \
  ${ENV_FILE:+--env-file "$ENV_FILE"} \
  -e HF_HOME="$HF_HOME" \
  -e ANIMA_UI_PORT=7000 \
  -e ANIMA_VLLM_CACHE="$HOME/thor-vllm-cache" \
  -e PYTHONUNBUFFERED=1 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$APP_DIR":"$APP_DIR" \
  -v "$HF_HOME":"$HF_HOME" \
  -w "$APP_DIR" \
  anima-thor-ui:latest >/dev/null

sleep 4
echo "[ui] status:"; sudo docker ps --filter "name=$NAME" --format "{{.Names}} {{.Status}}"
echo "[ui] http://$(hostname -I | awk '{print $1}'):7000   (Swagger at /docs)"
