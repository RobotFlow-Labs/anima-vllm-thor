"""Handler tests — mock the vLLM backend so the OpenAI/Anthropic/Ollama translation
paths are covered in CI (no engine/GPU). Complements the live scripts/smoke.sh."""
import httpx
import respx
from fastapi.testclient import TestClient

from anima_thor_ui import main, ollama_api, vllm_manager

VLLM = "http://127.0.0.1:8000"
_OAI_RESP = {"id": "x", "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 2, "completion_tokens": 1}}


def _up(monkeypatch):
    monkeypatch.setattr(vllm_manager, "is_running", lambda: True)
    monkeypatch.setattr(ollama_api, "_served_name", lambda: "m")


@respx.mock
def test_ollama_chat_translates(monkeypatch):
    _up(monkeypatch)
    respx.post(f"{VLLM}/v1/chat/completions").mock(return_value=httpx.Response(200, json=_OAI_RESP))
    with TestClient(main.app) as c:
        r = c.post("/api/chat", json={"model": "m:latest", "stream": False,
                                      "messages": [{"role": "user", "content": "hi"}]})
    d = r.json()
    assert r.status_code == 200 and d["message"]["content"] == "hi" and d["done"] is True


@respx.mock
def test_anthropic_messages_translates(monkeypatch):
    _up(monkeypatch)
    respx.post(f"{VLLM}/v1/chat/completions").mock(return_value=httpx.Response(200, json=_OAI_RESP))
    with TestClient(main.app) as c:
        r = c.post("/v1/messages", json={"model": "m", "max_tokens": 16,
                                         "messages": [{"role": "user", "content": "hi"}]})
    d = r.json()
    assert d["type"] == "message" and d["content"][0] == {"type": "text", "text": "hi"}
    assert d["stop_reason"] == "end_turn"


def test_endpoints_503_when_engine_down(monkeypatch):
    monkeypatch.setattr(vllm_manager, "is_running", lambda: False)
    with TestClient(main.app) as c:
        assert c.post("/api/chat", json={"model": "m", "messages": [], "stream": False}).status_code == 503
        assert c.post("/v1/messages", json={"model": "m", "max_tokens": 8, "messages": []}).status_code == 503
