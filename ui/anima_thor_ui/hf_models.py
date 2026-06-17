"""HuggingFace model management + Thor-compatibility ranking.

Three jobs:
  1. list local (downloaded) models from the HF cache, with sizes
  2. discover NVFP4 models on the Hub and rank them for *this* box (fits in
     128 GB unified + estimated decode tok/s from active params)
  3. download / delete models in the cache (download runs in a background thread)
"""
from __future__ import annotations

import re
import shutil
import threading
import time

from huggingface_hub import HfApi, snapshot_download

from .config import (
    settings, SUPPORTED_ARCHS, NVFP4_BYTES_PER_PARAM, DECODE_BYTES_PER_ACTIVE_PARAM,
)

_api = HfApi(token=settings.HF_TOKEN)

# repo_id -> {"status": queued|downloading|done|error, "msg", "started", "gb"}
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()

_NVFP4_HINTS = ("nvfp4", "fp4", "modelopt", "model optimizer")
_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)B(?:-A(\d+(?:\.\d+)?)B)?", re.IGNORECASE)


# ----------------------------------------------------------------------------- local cache
def _cache_repo_dir(repo_id: str):
    return settings.hub_dir / ("models--" + repo_id.replace("/", "--"))


def _dir_size_gb(path) -> float:
    total = 0
    if path.exists():
        for p in path.rglob("*"):
            if p.is_file() and not p.is_symlink():
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
    return round(total / 1e9, 2)


def list_local() -> list[dict]:
    out = []
    hub = settings.hub_dir
    if not hub.exists():
        return out
    for d in sorted(hub.glob("models--*")):
        repo = d.name[len("models--"):].replace("--", "/", 1).replace("--", "/")
        out.append({
            "repo_id": repo,
            "size_gb": _dir_size_gb(d),
            "is_serving": False,  # filled by caller against the running engine
        })
    return out


def delete_local(repo_id: str) -> dict:
    d = _cache_repo_dir(repo_id)
    if not d.exists():
        return {"deleted": False, "reason": "not found in cache"}
    shutil.rmtree(d, ignore_errors=True)
    return {"deleted": True, "repo_id": repo_id}


# ----------------------------------------------------------------------------- compat scoring
def _parse_size(repo_id: str) -> tuple[float | None, float | None]:
    """Return (total_B, active_B) parsed from the repo name, e.g. 80B-A3B -> (80, 3)."""
    m = _SIZE_RE.search(repo_id)
    if not m:
        return None, None
    total = float(m.group(1))
    active = float(m.group(2)) if m.group(2) else total  # dense: active == total
    return total, active


def _is_nvfp4(tags: list[str], repo_id: str) -> bool:
    blob = (" ".join(tags) + " " + repo_id).lower()
    return any(h in blob for h in _NVFP4_HINTS)


def _arch_of(tags: list[str]) -> str | None:
    for t in tags:
        if t in SUPPORTED_ARCHS:
            return t
    return None


def score_model(repo_id: str, tags: list[str], downloads: int, likes: int) -> dict | None:
    """Annotate a Hub model with Thor fit + estimated speed. None if irrelevant."""
    if not _is_nvfp4(tags, repo_id):
        return None
    total_b, active_b = _parse_size(repo_id)
    arch = _arch_of(tags)
    weight_gb = round(total_b * NVFP4_BYTES_PER_PARAM, 1) if total_b else None
    fits = (weight_gb is not None and weight_gb <= settings.WEIGHT_BUDGET_GB)
    est_tps = None
    if active_b:
        est_tps = int(settings.MEM_BANDWIDTH_GBS / (active_b * DECODE_BYTES_PER_ACTIVE_PARAM))
    # rank: must-fit first, then speed, then popularity
    rank = 0.0
    if fits:
        rank += 1000
    if est_tps:
        rank += min(est_tps, 200)
    rank += min(downloads / 50_000, 40) + min(likes, 40)
    arch_ok = arch is not None
    return {
        "repo_id": repo_id,
        "arch": arch or "(unverified)",
        "arch_supported": arch_ok,
        "total_b": total_b,
        "active_b": active_b,
        "weight_gb": weight_gb,
        "fits": fits,
        "est_single_tps": est_tps,
        "downloads": downloads,
        "likes": likes,
        "rank": round(rank, 1),
        "verdict": _verdict(fits, arch_ok, est_tps),
    }


def _verdict(fits, arch_ok, est_tps) -> str:
    if not fits:
        return "TOO BIG"
    if not arch_ok:
        return "UNTESTED ARCH"
    if est_tps and est_tps >= 60:
        return "ROCKS"
    if est_tps and est_tps >= 25:
        return "BALANCED"
    return "SMART/SLOW"


def direct_entry(repo_id: str) -> dict | None:
    """Look up one exact repo. NVFP4 -> normal ranked entry; otherwise a
    'quantize this' candidate so a pasted bf16 model still shows in Discover."""
    try:
        info = _api.model_info(repo_id, files_metadata=True)
    except Exception:  # noqa: BLE001
        return None
    tags = list(getattr(info, "tags", []) or [])
    dls = getattr(info, "downloads", 0) or 0
    likes = getattr(info, "likes", 0) or 0
    s = score_model(repo_id, tags, dls, likes)
    if s is not None:
        return s  # already NVFP4 — rank it normally
    cfg = getattr(info, "config", None) or {}
    arch = (cfg.get("model_type") or "").lower()
    bf16 = round(sum((x.size or 0) for x in (info.siblings or [])
                     if x.rfilename.endswith(".safetensors")) / 1e9, 1)
    if bf16 == 0:
        m = re.search(r"(\d+(?:\.\d+)?)B", repo_id)
        bf16 = round(float(m.group(1)) * 2, 1) if m else 0.0
    out_gb = round(bf16 * 0.28, 1) if bf16 else None
    quant_ok = 0 < bf16 <= 100
    return {
        "repo_id": repo_id, "arch": arch or "(unknown)", "arch_supported": False,
        "total_b": None, "active_b": None, "weight_gb": out_gb, "fits": False,
        "est_single_tps": None, "downloads": dls, "likes": likes, "rank": 1e9,
        "verdict": "QUANTIZE → NVFP4" if quant_ok else "TOO BIG TO QUANTIZE",
        "needs_quantize": quant_ok, "bf16_gb": bf16,
    }


def discover(query: str = "NVFP4", limit: int = 40) -> list[dict]:
    """Search the Hub for NVFP4 models and rank them for Thor.

    If the query is an exact repo id ('org/name'), look it up directly and place
    it first — even non-NVFP4 (it becomes a quantize candidate)."""
    scored: list[dict] = []
    seen: set[str] = set()
    q = query.strip()
    if "/" in q:
        d = direct_entry(q)
        if d:
            scored.append(d)
            seen.add(q)
    # general NVFP4 search + the nvidia org's curated NVFP4 set (best source)
    sources = (
        _api.list_models(search=query, sort="trendingScore", limit=limit * 3, full=True),
        _api.list_models(author="nvidia", search="NVFP4",
                         sort="trendingScore", limit=limit * 2, full=True),
    )
    for src in sources:
        for m in src:
            rid = m.id
            if rid in seen:
                continue
            seen.add(rid)
            tags = list(getattr(m, "tags", []) or [])
            s = score_model(rid, tags, getattr(m, "downloads", 0) or 0, getattr(m, "likes", 0) or 0)
            if s:
                scored.append(s)
    scored.sort(key=lambda x: x["rank"], reverse=True)
    return scored[:limit]


# ----------------------------------------------------------------------------- download
def _do_download(repo_id: str):
    with _jobs_lock:
        _jobs[repo_id] = {"status": "downloading", "msg": "", "started": time.time(), "gb": 0.0}
    try:
        # cache_dir points at the HF_HOME the engine container mounts, so a model
        # downloaded here is immediately servable.
        snapshot_download(repo_id, token=settings.HF_TOKEN, cache_dir=str(settings.hub_dir))
        with _jobs_lock:
            _jobs[repo_id].update(status="done", gb=_dir_size_gb(_cache_repo_dir(repo_id)))
    except Exception as e:  # noqa: BLE001
        with _jobs_lock:
            _jobs[repo_id].update(status="error", msg=str(e)[:400])


def start_download(repo_id: str) -> dict:
    with _jobs_lock:
        j = _jobs.get(repo_id)
        if j and j["status"] in ("queued", "downloading"):
            return {"started": False, "reason": "already downloading", "job": j}
    t = threading.Thread(target=_do_download, args=(repo_id,), daemon=True)
    t.start()
    return {"started": True, "repo_id": repo_id}


def download_status(repo_id: str | None = None) -> dict:
    with _jobs_lock:
        if repo_id:
            j = dict(_jobs.get(repo_id, {"status": "unknown"}))
            # live byte count while downloading
            if j.get("status") == "downloading":
                j["gb"] = _dir_size_gb(_cache_repo_dir(repo_id))
            return j
        return {k: dict(v) for k, v in _jobs.items()}
