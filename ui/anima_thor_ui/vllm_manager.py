"""Lifecycle control for the anima-vllm:thor-latest engine container.

Drives the engine the same way ``serve.sh`` does — ``docker run`` with the
Jetson runtime, host networking, the NVFP4/Marlin-fallback env, and the HF +
vLLM cache mounts. We shell out to the ``docker`` CLI (no docker SDK dep — keeps
the arm64 image tiny and the behaviour identical to the documented commands).
"""
from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass, asdict

from .config import settings

_started_at: float | None = None
_current: dict | None = None  # the ServeConfig used for the running container


@dataclass
class ServeConfig:
    model: str                                   # HF repo id (must be in local cache)
    served_name: str = ""                        # OpenAI model id; defaults to a slug of model
    max_model_len: int = 32768
    gpu_memory_utilization: float = 0.70
    kv_cache_dtype: str = "fp8"                  # fp8 | auto
    attention_backend: str = "TRITON_ATTN"       # the reliable NVFP4-MoE backend on sm_110
    spec_decode: str = "off"                     # off | ngram | mtp
    port: int = settings.VLLM_PORT

    def slug(self) -> str:
        return self.served_name or self.model.split("/")[-1].lower()


def _docker(*args: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)


def docker_available() -> bool:
    return shutil.which("docker") is not None


def is_running() -> bool:
    r = _docker("ps", "--filter", f"name=^{settings.VLLM_CONTAINER}$", "--format", "{{.Names}}")
    return settings.VLLM_CONTAINER in (r.stdout or "")


def stop() -> dict:
    global _started_at, _current
    _docker("rm", "-f", settings.VLLM_CONTAINER, timeout=120)
    _started_at = None
    _current = None
    return {"stopped": True}


def serve(cfg: ServeConfig) -> dict:
    """Launch the engine with the given config. Replaces any running engine."""
    global _started_at, _current
    if not docker_available():
        raise RuntimeError("docker CLI not found on host")
    stop()  # graceful replace

    cmd = [
        "run", "-d", "--name", settings.VLLM_CONTAINER,
        "--runtime", "nvidia", "--network", "host",
        "--shm-size=16g", "--ulimit", "memlock=-1", "--ulimit", "stack=67108864",
        "-e", "VLLM_USE_FLASHINFER_MOE_FP4=0",
    ]
    if settings.HF_TOKEN:
        cmd += ["-e", f"HF_TOKEN={settings.HF_TOKEN}"]
    cmd += [
        "-v", f"{settings.HF_HOME}:/root/.cache/huggingface",
        "-v", f"{settings.VLLM_CACHE}:/root/.cache/vllm",
        settings.VLLM_IMAGE,
        "vllm", "serve", cfg.model,
        "--trust-remote-code",
        "--attention-backend", cfg.attention_backend,
        "--gpu-memory-utilization", str(cfg.gpu_memory_utilization),
        "--kv-cache-dtype", cfg.kv_cache_dtype,
        "--max-model-len", str(cfg.max_model_len),
        "--served-model-name", cfg.slug(),
        "--port", str(cfg.port),
    ]
    if cfg.spec_decode == "ngram":
        cmd += ["--speculative-config",
                '{"method":"ngram","prompt_lookup_max":8,"prompt_lookup_min":1,"num_speculative_tokens":5}']
    elif cfg.spec_decode == "mtp":
        cmd += ["--speculative-config", '{"method":"mtp","num_speculative_tokens":3}']

    r = _docker(*cmd, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"docker run failed: {r.stderr.strip() or r.stdout.strip()}")
    _started_at = time.time()
    _current = asdict(cfg)
    return {"started": True, "container": settings.VLLM_CONTAINER, "config": _current}


def logs(tail: int = 80) -> str:
    r = _docker("logs", "--tail", str(tail), settings.VLLM_CONTAINER, timeout=30)
    return (r.stdout or "") + (r.stderr or "")


def status() -> dict:
    running = is_running()
    out: dict = {
        "running": running,
        "image": settings.VLLM_IMAGE,
        "container": settings.VLLM_CONTAINER,
        "base_url": settings.vllm_base_url,
        "config": _current,
        "uptime_s": int(time.time() - _started_at) if (_started_at and running) else 0,
    }
    if running:
        # quick "is it actually ready" probe is done by the proxy /v1/models; here we
        # just surface that the container exists. Readiness is reported by the UI poll.
        tail = logs(12)
        out["ready"] = "Application startup complete" in tail or "Starting vLLM API server" in tail
    return out
