"""Ollama-compatible API — so the huge Ollama client ecosystem (Open WebUI, n8n,
Cursor, LangChain's Ollama provider, etc.) works against this box unchanged.

Implements the subset clients actually use: /api/tags, /api/chat, /api/generate,
/api/version, /api/ps, /api/show — all translated to the OpenAI-compatible vLLM engine.
"""
from __future__ import annotations

import json
import time

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import hf_models, vllm_manager
from .config import settings

router = APIRouter(tags=["Ollama API"])


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _served_name() -> str:
    cfg = vllm_manager.status().get("config") or {}
    return cfg.get("served_name") or (cfg.get("model", "").split("/")[-1].lower() if cfg.get("model") else "")


@router.get("/api/version", summary="Ollama version probe")
async def version():
    return {"version": "0.1.0-anima"}


@router.get("/api/tags", summary="List models (Ollama format)")
async def tags():
    """Local models + the served one, in Ollama's /api/tags shape."""
    models = []
    served = _served_name()
    seen = set()
    for m in hf_models.list_local():
        name = (m.get("name") or m["repo_id"]).split("/")[-1]
        if name in seen:
            continue
        seen.add(name)
        models.append({"name": f"{name}:latest", "model": f"{name}:latest",
                       "modified_at": _now(), "size": int(m.get("size_gb", 0) * 1e9),
                       "digest": "", "details": {"family": "nvfp4", "format": "compressed-tensors"}})
    if served and served not in seen:
        models.insert(0, {"name": f"{served}:latest", "model": f"{served}:latest",
                          "modified_at": _now(), "size": 0, "digest": "", "details": {"family": "nvfp4"}})
    return {"models": models}


@router.get("/api/ps", summary="Running models (Ollama format)")
async def ps():
    served = _served_name()
    if not served or not vllm_manager.is_running():
        return {"models": []}
    return {"models": [{"name": f"{served}:latest", "model": f"{served}:latest",
                        "size": 0, "details": {"family": "nvfp4"}}]}


@router.post("/api/show", summary="Model info (Ollama format)")
async def show(request: Request):
    body = await request.json()
    name = (body.get("name") or body.get("model") or "").split(":")[0]
    return {"license": "apache-2.0", "details": {"family": "nvfp4", "format": "compressed-tensors"},
            "model_info": {"general.basename": name}, "capabilities": ["completion", "tools"]}


def _model_id(ollama_name: str) -> str:
    """Strip Ollama ':tag' and route to whatever the engine actually serves."""
    served = _served_name()
    return served or (ollama_name or "").split(":")[0]


def _engine_down_error(fmt: str = "ollama"):
    msg = "engine offline — start a model in the ANIMA Thor UI"
    return JSONResponse(status_code=503, content={"error": msg})


@router.post("/api/chat", summary="Ollama chat → vLLM (OpenAI translated)")
async def chat(request: Request):
    if not vllm_manager.is_running():
        return _engine_down_error()
    body = await request.json()
    model = _model_id(body.get("model", ""))
    stream = body.get("stream", True)            # Ollama defaults to streaming
    opts = body.get("options") or {}
    oai = {"model": model, "messages": body.get("messages", []), "stream": bool(stream),
           "max_tokens": opts.get("num_predict", 1024)}
    for k_o, k_v in (("temperature", "temperature"), ("top_p", "top_p")):
        if k_o in opts:
            oai[k_v] = opts[k_o]
    if body.get("tools"):
        oai["tools"] = body["tools"]

    if oai["stream"]:
        return StreamingResponse(_chat_stream(oai, model), media_type="application/x-ndjson")
    async with httpx.AsyncClient(timeout=300) as c:
        r = await c.post(f"{settings.vllm_base_url}/v1/chat/completions", json=oai)
        if r.status_code != 200:
            return JSONResponse(status_code=r.status_code, content={"error": r.text})
        d = r.json()
        msg = (d.get("choices") or [{}])[0].get("message", {})
        return {"model": model, "created_at": _now(),
                "message": {"role": "assistant", "content": msg.get("content") or ""},
                "done": True, "done_reason": "stop",
                "eval_count": d.get("usage", {}).get("completion_tokens", 0),
                "prompt_eval_count": d.get("usage", {}).get("prompt_tokens", 0)}


async def _chat_stream(oai: dict, model: str):
    async with httpx.AsyncClient(timeout=None) as c:
        async with c.stream("POST", f"{settings.vllm_base_url}/v1/chat/completions", json=oai) as r:
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                try:
                    delta = (json.loads(data).get("choices") or [{}])[0].get("delta", {})
                except json.JSONDecodeError:
                    continue
                if delta.get("content"):
                    yield json.dumps({"model": model, "created_at": _now(),
                                      "message": {"role": "assistant", "content": delta["content"]},
                                      "done": False}, separators=(",", ":")) + "\n"
    yield json.dumps({"model": model, "created_at": _now(),
                      "message": {"role": "assistant", "content": ""}, "done": True,
                      "done_reason": "stop"}, separators=(",", ":")) + "\n"


@router.post("/api/generate", summary="Ollama generate → vLLM")
async def generate(request: Request):
    if not vllm_manager.is_running():
        return _engine_down_error()
    body = await request.json()
    model = _model_id(body.get("model", ""))
    stream = body.get("stream", True)
    opts = body.get("options") or {}
    oai = {"model": model, "messages": [{"role": "user", "content": body.get("prompt", "")}],
           "stream": bool(stream), "max_tokens": opts.get("num_predict", 1024)}
    if "temperature" in opts:
        oai["temperature"] = opts["temperature"]

    if oai["stream"]:
        async def gen():
            async for chunk in _chat_stream(oai, model):
                o = json.loads(chunk)
                yield json.dumps({"model": model, "created_at": _now(),
                                  "response": o["message"]["content"], "done": o["done"]},
                                 separators=(",", ":")) + "\n"
        return StreamingResponse(gen(), media_type="application/x-ndjson")
    async with httpx.AsyncClient(timeout=300) as c:
        r = await c.post(f"{settings.vllm_base_url}/v1/chat/completions", json=oai)
        d = r.json()
        txt = (d.get("choices") or [{}])[0].get("message", {}).get("content", "")
        return {"model": model, "created_at": _now(), "response": txt, "done": True, "done_reason": "stop"}
