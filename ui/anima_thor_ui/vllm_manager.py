"""Lifecycle control for the anima-vllm:thor-latest engine container.

Drives the engine the same way ``serve.sh`` does — ``docker run`` with the
Jetson runtime, host networking, the NVFP4/Marlin-fallback env, and the HF +
vLLM cache mounts. We shell out to the ``docker`` CLI (no docker SDK dep — keeps
the arm64 image tiny and the behaviour identical to the documented commands).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass, asdict

from .config import settings

_started_at: float | None = None
_current: dict | None = None  # the ServeConfig used for the running container
_STATE = settings.HF_HOME / "anima_last_serve.json"   # persists across reboots (mounted)


@dataclass
class ServeConfig:
    model: str                                   # HF repo id / path (must be in local cache)
    served_name: str = ""                        # OpenAI model id; defaults to a slug of model
    max_model_len: int = 32768
    gpu_memory_utilization: float | str = "auto" # "auto" → fit to free memory; or an explicit 0.3–0.9
    kv_cache_dtype: str = "fp8"                  # fp8 | auto
    attention_backend: str = "TRITON_ATTN"       # the reliable NVFP4-MoE backend on sm_110
    spec_decode: str = "off"                     # off | ngram | mtp
    profile: str = "latency"                     # latency | throughput (sets the flags below)
    enable_prefix_caching: bool = False
    max_num_seqs: int = 0                         # 0 = vLLM default
    max_num_batched_tokens: int = 0              # 0 = vLLM default
    port: int = settings.VLLM_PORT

    def slug(self) -> str:
        return self.served_name or self.model.split("/")[-1].lower().replace("/", "-")

    def apply_profile(self):
        """High-throughput profile = prefix-cache + big batch (measured 747 tok/s agg @48)."""
        if self.profile == "throughput":
            self.enable_prefix_caching = True
            if not self.max_num_seqs:
                self.max_num_seqs = 48
            if not self.max_num_batched_tokens:
                self.max_num_batched_tokens = 8192   # must be >= KV block_size or vLLM asserts


def _docker(*args: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)


def docker_available() -> bool:
    return shutil.which("docker") is not None


def mem_gb() -> tuple[float, float]:
    """(available, total) host memory in GiB, from /proc/meminfo (host-wide in our container)."""
    avail = total = 0.0
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal:"):
                total = int(line.split()[1]) / 1048576
            elif line.startswith("MemAvailable:"):
                avail = int(line.split()[1]) / 1048576
    except OSError:
        pass
    return round(avail, 1), round(total, 1)


def wait_mem_stable(window: float = 1.0, need: int = 3, max_wait: int = 240) -> float:
    """Wait until free memory is TRULY flat — Thor's leaked GPU memory reclaims as a slow
    continuous climb (26→118 GB over minutes), and vLLM's ~30 s init memory-profiling ERRORS
    if free memory shifts mid-profile. Require `need` consecutive samples within `window` GiB."""
    last = -1.0
    stable = 0
    for _ in range(int(max_wait / 8)):
        avail, _t = mem_gb()
        if last >= 0 and abs(avail - last) <= window:
            stable += 1
            if stable >= need:
                return avail
        else:
            stable = 0
        last = avail
        time.sleep(8)
    return last


def auto_util(reserve_gb: float = 4.0) -> float:
    """Pick a gpu-memory-utilization that fits current free memory (avoids vLLM's
    'free memory < desired' error). Low floor (0.12) so small models still serve on
    a leaked box; capped at 0.85."""
    avail, total = mem_gb()
    if total <= 0:
        return 0.55
    return round(max(0.12, min(0.85, (avail - reserve_gb) / total)), 2)


def model_size_gb(model: str) -> float:
    """Best-effort on-disk size (GiB) of a model — the resident weight footprint.
    Handles both a quantized export path and an HF cache repo dir."""
    from pathlib import Path
    # engine-container path → the UI container's mounted host path
    m = model.replace("/root/.cache/huggingface/", str(settings.HF_HOME).rstrip("/") + "/")
    d = Path(m) if Path(m).is_dir() else settings.hub_dir / ("models--" + model.replace("/", "--"))
    if not d.exists():
        return 0.0
    total = 0
    for f in d.rglob("*"):
        if f.is_file() and not f.is_symlink():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return round(total / 1e9, 1)


def is_running() -> bool:
    r = _docker("ps", "--filter", f"name=^{settings.VLLM_CONTAINER}$", "--format", "{{.Names}}")
    return settings.VLLM_CONTAINER in (r.stdout or "")


def stop() -> dict:
    """Graceful `docker stop` (SIGTERM) — cleaner than force-kill, BUT measured fact:
    on Thor, stopping a served model leaks its GPU memory *regardless* of stop method
    (the Jetson driver doesn't reclaim the dead CUDA context). Only a REBOOT reclaims it.
    Practical rule: serve one model per boot; to swap models, reboot (UI auto-serves after)."""
    global _started_at, _current
    if is_running():
        _docker("stop", "-t", "40", settings.VLLM_CONTAINER, timeout=60)
    _docker("rm", "-f", settings.VLLM_CONTAINER, timeout=30)
    _started_at = None
    _current = None
    return {"stopped": True}


def serve(cfg: ServeConfig) -> dict:
    """Launch the engine with the given config. Replaces any running engine."""
    global _started_at, _current
    if not docker_available():
        raise RuntimeError("docker CLI not found on host")
    stop()  # graceful replace
    wait_mem_stable()  # wait out any in-progress leak-reclaim so vLLM profiling doesn't race

    # model-size-aware low-memory guard — need room for weights + a little KV/activation.
    # (Thor's GPU reservation + vLLM's leak-on-teardown can strand free RAM; only a reboot
    # fully reclaims it. But a SMALL model can still serve on a partly-leaked box.)
    avail, total = mem_gb()
    wsz = model_size_gb(cfg.model)
    need = (wsz + 6) if wsz else 30        # weights + KV headroom; fall back to 30 if unknown
    if total > 0 and avail < need:
        raise RuntimeError(
            f"Only {avail} GB free of {total} GB — not enough for this model (~{wsz or '?'} GB weights "
            f"+ headroom). Likely leaked GPU memory from a prior engine; reboot Thor to reclaim it "
            f"(the UI auto-restarts + auto-serves): use the Reboot button or `ssh thor sudo reboot`.")

    cfg.apply_profile()
    util = auto_util() if str(cfg.gpu_memory_utilization) in ("auto", "0", "0.0", "None") \
        else float(cfg.gpu_memory_utilization)
    cfg.gpu_memory_utilization = util  # record the resolved value

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
        "--gpu-memory-utilization", str(util),
        "--kv-cache-dtype", cfg.kv_cache_dtype,
        "--max-model-len", str(cfg.max_model_len),
        "--served-model-name", cfg.slug(),
        "--port", str(cfg.port),
    ]
    if cfg.enable_prefix_caching:
        cmd += ["--enable-prefix-caching"]
    if cfg.max_num_seqs:
        cmd += ["--max-num-seqs", str(cfg.max_num_seqs)]
    if cfg.max_num_batched_tokens:
        cmd += ["--max-num-batched-tokens", str(cfg.max_num_batched_tokens)]
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
    try:  # remember what we served so the box self-heals to the SAME state on boot
        _STATE.write_text(json.dumps({"model": cfg.model, "profile": cfg.profile,
                                      "max_model_len": cfg.max_model_len, "served_name": cfg.served_name}))
    except OSError:
        pass
    return {"started": True, "container": settings.VLLM_CONTAINER, "config": _current, "auto_util": util}


def reboot_host() -> dict:
    """Reboot the Thor host to reclaim leaked GPU memory. The UI container's
    restart policy + autoserve bring everything back automatically. Uses a
    privileged sibling container + nsenter into the host PID namespace (we only
    have the docker socket, not host init)."""
    if not docker_available():
        raise RuntimeError("docker CLI not available")
    # run in the already-present engine image (has util-linux/nsenter) — no pull needed
    r = _docker("run", "--rm", "--privileged", "--pid=host", settings.VLLM_IMAGE,
                "nsenter", "-t", "1", "-m", "-u", "-i", "-n", "-p", "--", "reboot",
                timeout=30)
    return {"rebooting": True, "detail": (r.stderr or r.stdout or "").strip()[:200]}


def autoserve_if_idle() -> None:
    """Called on UI startup — serve the LAST-served model (or the configured default)
    if nothing is running yet, so the box self-heals to the same state after a boot."""
    try:
        if is_running():
            return
        last = {}
        if _STATE.exists():
            try:
                last = json.loads(_STATE.read_text())
            except (OSError, ValueError):
                last = {}
        model = last.get("model") or settings.AUTOSERVE_MODEL
        if not model:
            return
        time.sleep(8)  # let the box settle after a boot before profiling memory
        serve(ServeConfig(model=model, profile=last.get("profile") or settings.AUTOSERVE_PROFILE,
                          max_model_len=last.get("max_model_len", 32768),
                          served_name=last.get("served_name", ""), gpu_memory_utilization="auto"))
    except Exception:  # noqa: BLE001 — never crash startup
        pass


def logs(tail: int = 80) -> str:
    r = _docker("logs", "--tail", str(tail), settings.VLLM_CONTAINER, timeout=30)
    return (r.stdout or "") + (r.stderr or "")


def status() -> dict:
    running = is_running()
    avail, total = mem_gb()
    out: dict = {
        "running": running,
        "image": settings.VLLM_IMAGE,
        "container": settings.VLLM_CONTAINER,
        "base_url": settings.vllm_base_url,
        "config": _current,
        "uptime_s": int(time.time() - _started_at) if (_started_at and running) else 0,
        "mem_avail_gb": avail,
        "mem_total_gb": total,
        "auto_util": auto_util(),
    }
    if running:
        # robust readiness: actually probe the OpenAI endpoint (more reliable than log-grep)
        out["ready"] = _engine_ready()
    return out


def _engine_ready() -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen(f"{settings.vllm_base_url}/v1/models", timeout=2) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001 — not up yet
        return False
