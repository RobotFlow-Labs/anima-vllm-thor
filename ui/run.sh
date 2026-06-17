#!/usr/bin/env bash
# Run anima-thor-ui on Thor (no container — needs direct host access to the docker daemon).
set -euo pipefail
cd "$(dirname "$0")"
[ -f .env ] && { set -a; . ./.env; set +a; }
command -v uv >/dev/null || { echo "install uv first: https://docs.astral.sh/uv/"; exit 1; }
uv venv --python 3.12 .venv 2>/dev/null || true
uv pip install -e . >/dev/null
exec uv run python -m anima_thor_ui.main
