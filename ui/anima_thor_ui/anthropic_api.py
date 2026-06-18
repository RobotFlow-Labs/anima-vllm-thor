"""Anthropic Messages API  <->  OpenAI Chat Completions translation.

Lets Anthropic-SDK clients (and tools that speak the Anthropic wire format) talk
to the OpenAI-compatible vLLM engine unchanged. Covers text, system prompts,
multi-turn, tool-use, and SSE streaming — the subset real clients use.
"""
from __future__ import annotations

import json
import time

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import vllm_manager
from .config import settings

router = APIRouter(tags=["Anthropic API"])


# ------------------------------------------------------------- request translation
def _to_openai(body: dict) -> dict:
    msgs: list[dict] = []
    sys = body.get("system")
    if isinstance(sys, str) and sys:
        msgs.append({"role": "system", "content": sys})
    elif isinstance(sys, list):  # Anthropic content-block system
        text = "".join(b.get("text", "") for b in sys if b.get("type") == "text")
        if text:
            msgs.append({"role": "system", "content": text})

    for m in body.get("messages", []):
        role, content = m.get("role"), m.get("content")
        if isinstance(content, str):
            msgs.append({"role": role, "content": content})
            continue
        # content is a list of blocks
        text_parts, tool_calls, tool_results = [], [], []
        for blk in content or []:
            t = blk.get("type")
            if t == "text":
                text_parts.append(blk.get("text", ""))
            elif t == "tool_use":
                tool_calls.append({
                    "id": blk.get("id"), "type": "function",
                    "function": {"name": blk.get("name"),
                                 "arguments": json.dumps(blk.get("input", {}))},
                })
            elif t == "tool_result":
                tool_results.append({
                    "role": "tool", "tool_call_id": blk.get("tool_use_id"),
                    "content": _flatten(blk.get("content")),
                })
        if tool_results:
            msgs.extend(tool_results)
        if text_parts or tool_calls:
            asst: dict = {"role": role}
            if text_parts:
                asst["content"] = "\n".join(text_parts)
            if tool_calls:
                asst["tool_calls"] = tool_calls
                asst.setdefault("content", None)
            msgs.append(asst)

    out: dict = {
        "model": body.get("model", "default"),
        "messages": msgs,
        "max_tokens": body.get("max_tokens", 1024),
        "stream": bool(body.get("stream", False)),
    }
    for k in ("temperature", "top_p", "stop_sequences"):
        if k in body:
            out["stop" if k == "stop_sequences" else k] = body[k]
    if "tools" in body:
        out["tools"] = [{"type": "function",
                         "function": {"name": t["name"],
                                      "description": t.get("description", ""),
                                      "parameters": t.get("input_schema", {})}}
                        for t in body["tools"]]
    return out


def _flatten(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content or "")


# ------------------------------------------------------------- response translation
def _to_anthropic(oai: dict, model: str) -> dict:
    choice = (oai.get("choices") or [{}])[0]
    msg = choice.get("message", {})
    blocks: list[dict] = []
    if msg.get("content"):
        blocks.append({"type": "text", "text": msg["content"]})
    for tc in msg.get("tool_calls", []) or []:
        fn = tc.get("function", {})
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        blocks.append({"type": "tool_use", "id": tc.get("id"),
                       "name": fn.get("name"), "input": args})
    stop_map = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}
    usage = oai.get("usage", {})
    return {
        "id": oai.get("id", f"msg_{int(time.time())}"),
        "type": "message", "role": "assistant", "model": model,
        "content": blocks or [{"type": "text", "text": ""}],
        "stop_reason": stop_map.get(choice.get("finish_reason"), "end_turn"),
        "stop_sequence": None,
        "usage": {"input_tokens": usage.get("prompt_tokens", 0),
                  "output_tokens": usage.get("completion_tokens", 0)},
    }


# ------------------------------------------------------------- streaming
async def _stream(oai_req: dict, model: str):
    msg_id = f"msg_{int(time.time())}"
    start = {"type": "message_start",
             "message": {"id": msg_id, "type": "message", "role": "assistant",
                         "model": model, "content": [], "stop_reason": None,
                         "usage": {"input_tokens": 0, "output_tokens": 0}}}
    yield _sse("message_start", start)
    yield _sse("content_block_start",
               {"type": "content_block_start", "index": 0,
                "content_block": {"type": "text", "text": ""}})
    yield _sse("ping", {"type": "ping"})

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", f"{settings.vllm_base_url}/v1/chat/completions",
                                 json=oai_req) as r:
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = (chunk.get("choices") or [{}])[0].get("delta", {})
                if delta.get("content"):
                    yield _sse("content_block_delta",
                               {"type": "content_block_delta", "index": 0,
                                "delta": {"type": "text_delta", "text": delta["content"]}})
    yield _sse("content_block_stop", {"type": "content_block_stop", "index": 0})
    yield _sse("message_delta",
               {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 0}})
    yield _sse("message_stop", {"type": "message_stop"})


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("/v1/messages", summary="Anthropic Messages (translated to the vLLM engine)")
async def messages(request: Request):
    """Anthropic-compatible endpoint. Point the Anthropic SDK's base_url here."""
    if not vllm_manager.is_running():
        return JSONResponse(status_code=503, content={"type": "error", "error": {
            "type": "overloaded_error", "message": "engine offline — start a model in the ANIMA Thor UI"}})
    body = await request.json()
    model = body.get("model", "default")
    oai_req = _to_openai(body)
    if oai_req["stream"]:
        return StreamingResponse(_stream(oai_req, model), media_type="text/event-stream")
    async with httpx.AsyncClient(timeout=300) as client:
        r = await client.post(f"{settings.vllm_base_url}/v1/chat/completions", json=oai_req)
        if r.status_code != 200:
            return JSONResponse(status_code=r.status_code,
                                content={"type": "error",
                                         "error": {"type": "api_error", "message": r.text}})
        return JSONResponse(_to_anthropic(r.json(), model))
