"""anima-thor-ui — FastAPI control plane for the anima-vllm:thor-latest engine.

Surface:
  /                       ANIMA-themed control dashboard (static SPA)
  /docs                   Swagger UI for the inference + control endpoints
  /v1/*                   OpenAI-compatible passthrough to the engine (+ Factory Droid)
  /v1/messages            Anthropic Messages (translated)
  /api/*                  control plane (engine lifecycle, model DL/delete, discovery)
No auth by design (single-user edge box). HF token is read from the environment.
"""
from __future__ import annotations

from pathlib import Path

import httpx
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import hf_models, quantize, vllm_manager
from .anthropic_api import router as anthropic_router
from .ollama_api import router as ollama_router
from .config import settings

app = FastAPI(
    title="ANIMA Thor UI",
    version="0.1.0",
    description=(
        "Control plane for the **anima-vllm:thor-latest** engine on NVIDIA Jetson AGX Thor. "
        "Use the **OpenAI** `/v1` endpoints (also Factory-Droid compatible) or the **Anthropic** "
        "`/v1/messages` endpoint below. Manage models and the engine from the dashboard at `/`."
    ),
    openapi_tags=[
        {"name": "OpenAI API", "description": "OpenAI-compatible — point any OpenAI SDK / Factory Droid here."},
        {"name": "Anthropic API", "description": "Anthropic Messages — point the Anthropic SDK base_url here."},
        {"name": "Engine", "description": "Start / stop / inspect the vLLM engine."},
        {"name": "Models", "description": "Download, delete, and discover Thor-ready NVFP4 models."},
    ],
)

STATIC = Path(__file__).resolve().parent.parent / "static"

import logging
import time as _time
_log = logging.getLogger("anima")
logging.basicConfig(level=logging.INFO, format="%(asctime)s anima %(message)s")


@app.middleware("http")
async def _timing(request, call_next):
    t0 = _time.time()
    resp = await call_next(request)
    dt = (_time.time() - t0) * 1000
    if not request.url.path.startswith(("/style", "/app.js", "/favicon")):
        _log.info(f"{request.method} {request.url.path} -> {resp.status_code} {dt:.0f}ms")
    return resp


@app.get("/healthz", tags=["Engine"], summary="Liveness/health probe (for monitoring)")
def healthz():
    s = vllm_manager.status()
    return {"status": "ok", "engine_running": s.get("running"), "engine_ready": s.get("ready"),
            "mem_avail_gb": s.get("mem_avail_gb"), "model": (s.get("config") or {}).get("served_name")}


# ============================================================ control plane (/api)
api = APIRouter(prefix="/api")


class ServeRequest(BaseModel):
    model: str
    max_model_len: int = 32768
    gpu_memory_utilization: str = "auto"          # "auto" fits free memory, or "0.6" etc.
    kv_cache_dtype: str = "fp8"
    attention_backend: str = "TRITON_ATTN"
    spec_decode: str = "off"
    profile: str = "latency"                       # latency | throughput
    served_name: str = ""


@api.get("/status", tags=["Engine"], summary="Engine + box status")
def api_status():
    return vllm_manager.status()


_tel = {"tokens": 0.0, "t": 0.0}


@api.get("/telemetry", tags=["Engine"], summary="Live throughput (from vLLM Prometheus metrics)")
async def api_telemetry():
    """Live tok/s (sampled delta of generation_tokens_total) + in-flight requests."""
    import re
    import time as _t
    try:
        async with httpx.AsyncClient(timeout=4) as c:
            txt = (await c.get(f"{settings.vllm_base_url}/metrics")).text
    except Exception:  # noqa: BLE001
        return {"tok_s": 0, "running": 0, "waiting": 0}

    def g(name: str) -> float:
        m = re.search(rf"^{re.escape(name)}\{{[^}}]*\}}\s+([0-9.eE+]+)", txt, re.M)
        return float(m.group(1)) if m else 0.0

    tokens = g("vllm:generation_tokens_total")
    now = _t.time()
    tps = 0.0
    if _tel["t"] and now > _tel["t"]:
        dt, dtok = now - _tel["t"], tokens - _tel["tokens"]
        if dt > 0 and dtok >= 0:
            tps = round(dtok / dt, 1)
    _tel["tokens"], _tel["t"] = tokens, now
    return {"tok_s": tps, "running": int(g("vllm:num_requests_running")),
            "waiting": int(g("vllm:num_requests_waiting")), "gen_total": int(tokens)}


@api.get("/logs", tags=["Engine"], summary="Tail engine logs")
def api_logs(tail: int = 80):
    if not vllm_manager.is_running():
        return {"logs": "(engine not running)"}
    return {"logs": vllm_manager.logs(tail)}


@api.post("/serve", tags=["Engine"], summary="Launch the engine with a model + config")
def api_serve(req: ServeRequest):
    try:
        cfg = vllm_manager.ServeConfig(
            model=req.model, served_name=req.served_name,
            max_model_len=req.max_model_len,
            gpu_memory_utilization=req.gpu_memory_utilization,
            kv_cache_dtype=req.kv_cache_dtype, attention_backend=req.attention_backend,
            spec_decode=req.spec_decode, profile=req.profile,
        )
        return vllm_manager.serve(cfg)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, str(e))


@api.post("/stop", tags=["Engine"], summary="Stop the engine")
def api_stop():
    return vllm_manager.stop()


@api.post("/reboot", tags=["Engine"], summary="Reboot Thor to reclaim leaked GPU memory (self-heals)")
def api_reboot():
    """Reboots the host. The UI auto-restarts and (if configured) auto-serves — full recovery."""
    try:
        return vllm_manager.reboot_host()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, str(e))


@app.on_event("startup")
def _autoserve():
    import threading
    threading.Thread(target=vllm_manager.autoserve_if_idle, daemon=True).start()


@api.get("/models/local", tags=["Models"], summary="List downloaded models")
def api_models_local():
    cur = (vllm_manager.status().get("config") or {}).get("model")
    items = hf_models.list_local()
    for it in items:
        it["is_serving"] = (it["repo_id"] == cur)
    # our own NVFP4-quantized outputs, served by container path
    for q in quantize.list_quantized():
        items.append({"repo_id": q["serve_id"], "name": q["name"], "size_gb": q["size_gb"],
                      "host_path": q["host_path"], "quantized": True,
                      "is_serving": q["serve_id"] == cur})
    return {"models": items, "hub_dir": str(settings.hub_dir)}


@api.delete("/models/local", tags=["Models"], summary="Delete a downloaded model")
def api_models_delete(repo_id: str):
    if repo_id == (vllm_manager.status().get("config") or {}).get("model"):
        raise HTTPException(409, "model is currently being served — stop the engine first")
    return hf_models.delete_local(repo_id)


@api.get("/models/discover", tags=["Models"], summary="Find Thor-ready NVFP4 models on HF")
def api_models_discover(query: str = "NVFP4", limit: int = 40):
    return {"models": hf_models.discover(query, limit),
            "budget_gb": settings.WEIGHT_BUDGET_GB, "bandwidth_gbs": settings.MEM_BANDWIDTH_GBS}


@api.post("/models/download", tags=["Models"], summary="Download a model from HF")
def api_models_download(repo_id: str):
    return hf_models.start_download(repo_id)


@api.get("/models/download", tags=["Models"], summary="Download progress")
def api_models_download_status(repo_id: str | None = None):
    return hf_models.download_status(repo_id)


@api.get("/quantize/validate", tags=["Models"], summary="Pre-flight: can we NVFP4-quantize this HF model?")
def api_quant_validate(repo_id: str):
    return quantize.validate(repo_id)


@api.post("/quantize", tags=["Models"], summary="Quantize an HF model to NVFP4 (stops the engine first)")
def api_quant_start(repo_id: str, calib_samples: int = 64):
    return quantize.start(repo_id, calib_samples)


@api.get("/quantize", tags=["Models"], summary="Quantization job status + stage")
def api_quant_status():
    return quantize.status() or {"status": "idle"}


@api.post("/quantize/publish", tags=["Models"], summary="Publish a quantized model to HuggingFace")
def api_quant_publish(name: str, base_model: str = "", private: bool = False):
    return quantize.publish(name, base_model, private)


@api.get("/quantize/publish", tags=["Models"], summary="Publish job status")
def api_quant_publish_status():
    return quantize.publish_status() or {"status": "idle"}


app.include_router(api)


# ============================================================ OpenAI passthrough (/v1)
oai = APIRouter(tags=["OpenAI API"])
_PASS = "/v1/chat/completions", "/v1/completions", "/v1/embeddings"


@oai.get("/v1/models", summary="List models served by the engine")
async def v1_models():
    async with httpx.AsyncClient(timeout=30) as c:
        try:
            r = await c.get(f"{settings.vllm_base_url}/v1/models")
            return JSONResponse(r.json(), status_code=r.status_code)
        except httpx.ConnectError:
            return JSONResponse({"object": "list", "data": [], "engine": "offline"}, status_code=503)


async def _proxy(path: str, request: Request):
    body = await request.body()
    headers = {"content-type": "application/json"}
    stream = b'"stream":true' in body.replace(b" ", b"")
    if stream:
        async def gen():
            async with httpx.AsyncClient(timeout=None) as c:
                async with c.stream("POST", f"{settings.vllm_base_url}{path}",
                                    content=body, headers=headers) as r:
                    async for chunk in r.aiter_raw():
                        yield chunk
        return StreamingResponse(gen(), media_type="text/event-stream")
    async with httpx.AsyncClient(timeout=300) as c:
        try:
            r = await c.post(f"{settings.vllm_base_url}{path}", content=body, headers=headers)
            return JSONResponse(r.json(), status_code=r.status_code)
        except httpx.ConnectError:
            raise HTTPException(503, "engine offline — start it from the dashboard")


@oai.post("/v1/chat/completions", summary="Chat completions (OpenAI / Factory Droid)")
async def v1_chat(request: Request):
    return await _proxy("/v1/chat/completions", request)


@oai.post("/v1/completions", summary="Text completions")
async def v1_comp(request: Request):
    return await _proxy("/v1/completions", request)


app.include_router(oai)
app.include_router(anthropic_router)
app.include_router(ollama_router)


# ============================================================ static dashboard
if STATIC.exists():
    app.mount("/", StaticFiles(directory=str(STATIC), html=True), name="ui")


def run():
    import uvicorn
    uvicorn.run("anima_thor_ui.main:app", host=settings.UI_HOST, port=settings.UI_PORT)


if __name__ == "__main__":
    run()
