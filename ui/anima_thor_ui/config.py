"""Runtime configuration for anima-thor-ui.

All settings come from environment variables (see .env.example). No secrets are
written to disk by the app; HF_TOKEN is read from the environment only.
"""
from __future__ import annotations

import os
from pathlib import Path


def _bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).lower() in ("1", "true", "yes", "on")


class Settings:
    # --- this UI ---
    UI_HOST: str = os.environ.get("ANIMA_UI_HOST", "0.0.0.0")
    UI_PORT: int = int(os.environ.get("ANIMA_UI_PORT", "7000"))

    # --- the vLLM engine we drive ---
    VLLM_IMAGE: str = os.environ.get("ANIMA_VLLM_IMAGE", "anima-vllm:thor-latest")
    VLLM_CONTAINER: str = os.environ.get("ANIMA_VLLM_CONTAINER", "anima-vllm")
    VLLM_PORT: int = int(os.environ.get("ANIMA_VLLM_PORT", "8000"))
    VLLM_HOST: str = os.environ.get("ANIMA_VLLM_HOST", "127.0.0.1")

    # --- storage (host paths, mounted into the engine container) ---
    HF_HOME: Path = Path(os.environ.get("HF_HOME", str(Path.home() / "thor-serve/models/huggingface")))
    VLLM_CACHE: Path = Path(os.environ.get("ANIMA_VLLM_CACHE", str(Path.home() / "thor-vllm-cache")))

    # --- secrets / auth ---
    HF_TOKEN: str | None = os.environ.get("HF_TOKEN") or None
    # optional API key: if set, inference + control endpoints require Bearer/x-api-key.
    # Empty = open (single-user edge default). Set for exposed/fleet deployments.
    API_KEY: str = os.environ.get("ANIMA_API_KEY", "")

    # --- Thor hardware envelope (for the compatibility filter) ---
    UNIFIED_MEM_GB: float = float(os.environ.get("ANIMA_MEM_GB", "128"))
    MEM_BANDWIDTH_GBS: float = float(os.environ.get("ANIMA_BW_GBS", "273"))
    # leave headroom for KV-cache + activations + system; weights must fit below this
    WEIGHT_BUDGET_GB: float = float(os.environ.get("ANIMA_WEIGHT_BUDGET_GB", "95"))

    # HF account to publish our quantized models under
    HF_USER: str = os.environ.get("ANIMA_HF_USER", "ilessio-aiflowlab")

    # self-healing: on UI startup, auto-serve this model if nothing is running
    # (combined with the container's restart policy → full recovery after a reboot/power-loss)
    AUTOSERVE_MODEL: str = os.environ.get("ANIMA_AUTOSERVE_MODEL", "")
    AUTOSERVE_PROFILE: str = os.environ.get("ANIMA_AUTOSERVE_PROFILE", "latency")

    @property
    def vllm_base_url(self) -> str:
        return f"http://{self.VLLM_HOST}:{self.VLLM_PORT}"

    @property
    def hub_dir(self) -> Path:
        return self.HF_HOME / "hub"


settings = Settings()

# Architectures we have proven (or are confident) load on vLLM 0.23 / sm_110a.
# nemotron_h + qwen3_5_moe are measured-good; the rest are standard transformers archs.
# NOTE: gemma4 is excluded — Gemma-4 MoE failed engine init on the 0.23.0 build
# (arch too new / not yet registered). gemma3/gemma2 are fine.
SUPPORTED_ARCHS = {
    "qwen3_5_moe", "qwen3_moe", "qwen3_next", "qwen3", "qwen2", "qwen2_moe",
    "nemotron_h", "nemotron-nas", "llama", "llama4", "gemma3", "gemma2",
    "mistral", "mixtral", "phi3", "minimax_m2", "glm_moe_dsa", "deepseek_v3", "deepseek_v4",
}

# NVFP4 effective bytes per weight param (4-bit + group scales).
NVFP4_BYTES_PER_PARAM = 0.55
# Empirically-calibrated effective bytes moved per active param per decoded token
# (Qwen3.6-A3B measured 79 tok/s, Nano-A3B 68 tok/s -> ~1.2 incl. attention/overhead).
DECODE_BYTES_PER_ACTIVE_PARAM = 1.2
