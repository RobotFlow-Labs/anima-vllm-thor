"""Quantize any bf16/fp16 HuggingFace model to NVFP4, on Thor, from the UI.

Pipeline (each step reported to the UI as a stage so the user sees real progress):
  validate -> stop engine (free RAM) -> download bf16 -> run ModelOpt NVFP4 PTQ
  in a container -> export to the cache -> appears in Local Models, ready to serve.

The heavy lifting runs inside the anima-vllm:thor-latest image (torch 2.11 / CUDA 13)
with nvidia-modelopt installed at runtime, driving scripts/quantize_nvfp4.py.
"""
from __future__ import annotations

import re
import subprocess
import threading
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from .config import settings
from . import vllm_manager

_api = HfApi(token=settings.HF_TOKEN)

# Dense + simple-MoE archs ModelOpt's NVFP4 PTQ handles well. Hybrid/linear-attention
# archs (mamba2 nemotron_h, deltanet qwen3_next) are NOT reliably supported — block them.
QUANTIZABLE_ARCHS = {
    "llama", "qwen2", "qwen3", "qwen3_moe", "qwen2_moe", "mistral", "mixtral",
    "gemma2", "gemma3", "phi3", "phi", "mpt", "gpt_neox", "starcoder2", "cohere",
}
HYBRID_BLOCKED = {"nemotron_h", "qwen3_next", "mamba", "mamba2", "jamba", "minimax_m2"}
ALREADY_Q_HINTS = ("nvfp4", "fp4", "awq", "gptq", "fp8", "int4", "int8", "modelopt", "compressed-tensors")

OUT_ROOT = settings.HF_HOME / "anima-nvfp4"           # host path
OUT_ROOT_CONTAINER = "/root/.cache/huggingface/anima-nvfp4"  # same dir inside engine

# single global job (one quantize at a time — it owns the whole box)
_job: dict | None = None
_lock = threading.Lock()

STAGES = ["validate", "stop_engine", "download", "load", "calibrate", "quantize", "export", "done"]
_STAGE_PCT = {"validate": 3, "stop_engine": 6, "download": 30, "load": 45,
              "calibrate": 60, "quantize": 80, "export": 95, "done": 100}


# ---------------------------------------------------------------- validation
def validate(repo_id: str) -> dict:
    """Cheap pre-flight: can we quantize this on Thor? Friendly, specific verdict."""
    try:
        info = _api.model_info(repo_id, files_metadata=True)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "verdict": "NOT FOUND",
                "reason": f"Couldn't read {repo_id} on HuggingFace ({str(e)[:120]}).", "fix": "Check the repo id."}

    tags = [t.lower() for t in (info.tags or [])]
    cfg = getattr(info, "config", None) or {}
    arch = (cfg.get("model_type") or "").lower()
    blob = " ".join(tags) + " " + repo_id.lower()

    # already quantized?
    if any(h in blob for h in ALREADY_Q_HINTS):
        return {"ok": False, "verdict": "ALREADY QUANTIZED",
                "reason": "This model is already quantized — just download it directly.",
                "fix": "Use the Discover/Download tab instead."}

    # arch gate
    if arch in HYBRID_BLOCKED or any(h in blob for h in HYBRID_BLOCKED):
        return {"ok": False, "verdict": "UNSUPPORTED ARCH",
                "reason": f"'{arch or 'hybrid'}' uses Mamba/linear-attention layers ModelOpt's NVFP4 path can't quantize yet.",
                "fix": "Use a pre-made NVFP4 build of this family from the Discover tab."}
    arch_ok = arch in QUANTIZABLE_ARCHS

    # size: sum safetensors bytes ~= bf16 weight footprint needed for calibration
    bf16_gb = round(sum((s.size or 0) for s in (info.siblings or [])
                        if s.rfilename.endswith(".safetensors")) / 1e9, 1)
    if bf16_gb == 0:  # no safetensors metadata — estimate from name
        m = re.search(r"(\d+(?:\.\d+)?)B", repo_id)
        bf16_gb = round(float(m.group(1)) * 2, 1) if m else 0.0
    out_gb = round(bf16_gb * 0.28, 1) if bf16_gb else None
    # need bf16 weights + activations resident; keep under ~100 GB of the 128
    fits = 0 < bf16_gb <= 100
    est_min = int(8 + bf16_gb * 0.8) if bf16_gb else None  # rough wall-clock

    if not fits:
        return {"ok": False, "verdict": "TOO BIG TO QUANTIZE", "arch": arch,
                "bf16_gb": bf16_gb, "reason":
                f"Calibration needs the model in bf16 (~{bf16_gb} GB) in memory; that exceeds Thor's safe budget.",
                "fix": "Pick a smaller model (≤ ~45 B), or find a ready NVFP4 build."}

    verdict = "READY" if arch_ok else "LIKELY OK"
    return {
        "ok": True, "verdict": verdict, "arch": arch or "(unknown)", "arch_known": arch_ok,
        "bf16_gb": bf16_gb, "out_gb": out_gb, "est_min": est_min,
        "reason": (f"{arch} model, ~{bf16_gb} GB in bf16 → ~{out_gb} GB NVFP4. Fits Thor for calibration."
                   + ("" if arch_ok else " Arch not in our verified list — should work, but unproven.")),
        "fix": "", "out_name": _out_name(repo_id),
    }


def _out_name(repo_id: str) -> str:
    return repo_id.split("/")[-1] + "-NVFP4-anima"


# ---------------------------------------------------------------- job runner
def status() -> dict | None:
    with _lock:
        return dict(_job) if _job else None


def _set(**kw):
    with _lock:
        if _job is not None:
            _job.update(kw)
            if "stage" in kw:
                _job["pct"] = _STAGE_PCT.get(kw["stage"], _job.get("pct", 0))


def start(repo_id: str, calib_samples: int = 64) -> dict:
    global _job
    with _lock:
        if _job and _job["status"] == "running":
            return {"started": False, "reason": "a quantization is already running", "job": dict(_job)}
        _job = {"repo_id": repo_id, "status": "running", "stage": "validate", "pct": 0,
                "msg": "", "log": "", "started": time.time(), "out_path": None, "out_name": _out_name(repo_id)}
    threading.Thread(target=_run, args=(repo_id, calib_samples), daemon=True).start()
    return {"started": True, "repo_id": repo_id}


def _run(repo_id: str, calib_samples: int):
    try:
        v = validate(repo_id)
        if not v["ok"]:
            _set(status="error", msg=f"{v['verdict']}: {v['reason']}")
            return

        # free ALL RAM/VRAM — stop the serving engine
        _set(stage="stop_engine", msg="Stopping the engine to free memory…")
        vllm_manager.stop()
        time.sleep(3)

        _set(stage="download", msg=f"Downloading {repo_id} (bf16)…")
        snapshot_download(repo_id, token=settings.HF_TOKEN, cache_dir=str(settings.hub_dir))

        out_name = _out_name(repo_id)
        host_out = OUT_ROOT / out_name
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        cont_out = f"{OUT_ROOT_CONTAINER}/{out_name}"

        _set(stage="load", msg="Loading model + ModelOpt in the engine container…")
        rc = _run_worker(repo_id, cont_out, calib_samples)
        if rc != 0:
            _set(status="error", msg="Quantization worker failed — see log.")
            return
        if not (host_out / "config.json").exists():
            _set(status="error", msg="Export finished but no checkpoint was written — see log.")
            return

        _set(stage="done", status="done", out_path=str(host_out),
             msg=f"✓ {out_name} ready — it's now in Local Models.")
    except Exception as e:  # noqa: BLE001
        _set(status="error", msg=str(e)[:300])


def _run_worker(repo_id: str, cont_out: str, calib_samples: int) -> int:
    """Run scripts/quantize_nvfp4.py inside the engine image; stream stages from its stdout."""
    worker = Path(__file__).resolve().parent.parent / "scripts" / "quantize_nvfp4.py"
    cmd = [
        "docker", "run", "--rm", "--name", "anima-quant", "--runtime", "nvidia",
        "--network", "host", "--shm-size=16g", "--ulimit", "memlock=-1", "--ulimit", "stack=67108864",
        "-e", "PIP_CONSTRAINT=",
    ]
    if settings.HF_TOKEN:
        cmd += ["-e", f"HF_TOKEN={settings.HF_TOKEN}"]
    cmd += [
        "-v", f"{settings.HF_HOME}:/root/.cache/huggingface",
        "-v", f"{worker}:/tmp/quantize_nvfp4.py:ro",
        settings.VLLM_IMAGE,
        "bash", "-lc",
        f"pip install -q nvidia-modelopt 2>/dev/null; "
        f"python /tmp/quantize_nvfp4.py --model '{repo_id}' --out '{cont_out}' --calib {calib_samples}",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    buf: list[str] = []
    for line in proc.stdout:  # type: ignore
        buf.append(line)
        if len(buf) > 400:
            buf = buf[-400:]
        with _lock:
            if _job is not None:
                _job["log"] = "".join(buf)[-6000:]
        m = re.search(r"\[STAGE\]\s*(\w+)\s*(.*)", line)
        if m and m.group(1) in _STAGE_PCT:
            _set(stage=m.group(1), msg=m.group(2).strip() or _job.get("msg", ""))
    proc.wait()
    return proc.returncode


# ---------------------------------------------------------------- local quantized models
def list_quantized() -> list[dict]:
    out = []
    if OUT_ROOT.exists():
        for d in sorted(OUT_ROOT.iterdir()):
            if d.is_dir() and (d / "config.json").exists():
                gb = round(sum(p.stat().st_size for p in d.rglob("*") if p.is_file()) / 1e9, 2)
                out.append({"name": d.name, "serve_id": f"{OUT_ROOT_CONTAINER}/{d.name}",
                            "host_path": str(d), "size_gb": gb})
    return out
