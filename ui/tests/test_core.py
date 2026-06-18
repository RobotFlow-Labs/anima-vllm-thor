"""Unit tests for anima-thor-ui pure logic (no engine/GPU needed)."""
from anima_thor_ui import hf_models, vllm_manager
from anima_thor_ui.vllm_manager import ServeConfig
from anima_thor_ui.anthropic_api import _to_openai, _to_anthropic


# ---- HF model scoring / Thor-fit ----
def test_parse_size_moe_and_dense():
    assert hf_models._parse_size("nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4") == (80.0, 3.0)
    assert hf_models._parse_size("nvidia/Qwen3-32B-NVFP4") == (32.0, 32.0)  # dense → active == total
    assert hf_models._parse_size("Hcompany/Holo-3.1-35B-A3B-NVFP4") == (35.0, 3.0)
    assert hf_models._parse_size("foo/bar-no-size") == (None, None)


def test_score_verdicts():
    rocks = hf_models.score_model("nvidia/Qwen3.6-35B-A3B-NVFP4", ["qwen3_5_moe", "nvfp4"], 1_000_000, 200)
    assert rocks and rocks["fits"] and rocks["arch"] == "qwen3_5_moe" and rocks["verdict"] == "ROCKS"

    toobig = hf_models.score_model("x/Giant-400B-NVFP4", ["nvfp4"], 0, 0)
    assert toobig and not toobig["fits"] and toobig["verdict"] == "TOO BIG"

    slow = hf_models.score_model("nvidia/Qwen3-32B-NVFP4", ["qwen3", "nvfp4"], 0, 0)  # dense 32B → ~7 tok/s
    assert slow and slow["fits"] and slow["verdict"] == "SMART/SLOW"

    untested = hf_models.score_model("nvidia/Gemma-4-31B-IT-NVFP4", ["nvfp4"], 0, 0)  # no known arch tag
    assert untested and untested["verdict"] == "UNTESTED ARCH"

    assert hf_models.score_model("meta-llama/Llama-3-8B", ["llama"], 0, 0) is None  # not NVFP4 → skipped


def test_gemma4_not_supported_arch():
    # gemma4 deliberately excluded (failed engine init on 0.23)
    assert "gemma4" not in hf_models.SUPPORTED_ARCHS
    assert "qwen3_5_moe" in hf_models.SUPPORTED_ARCHS


# ---- memory math ----
def test_auto_util_clamps(monkeypatch):
    monkeypatch.setattr(vllm_manager, "mem_gb", lambda: (118.0, 128.0))
    assert vllm_manager.auto_util() == 0.85                       # capped high
    monkeypatch.setattr(vllm_manager, "mem_gb", lambda: (26.0, 128.0))
    assert vllm_manager.auto_util() == round((26 - 4) / 128, 2)   # ≈0.17, fits a leaked box
    monkeypatch.setattr(vllm_manager, "mem_gb", lambda: (8.0, 128.0))
    assert vllm_manager.auto_util() == 0.12                       # floored


# ---- serve config / profiles ----
def test_profile_throughput_sets_flags():
    c = ServeConfig(model="x", profile="throughput")
    c.apply_profile()
    assert c.enable_prefix_caching and c.max_num_seqs == 48 and c.max_num_batched_tokens == 8192


def test_profile_latency_is_plain():
    c = ServeConfig(model="x", profile="latency")
    c.apply_profile()
    assert not c.enable_prefix_caching and c.max_num_seqs == 0


def test_slug():
    assert ServeConfig(model="nvidia/Qwen3.6-35B-A3B-NVFP4").slug() == "qwen3.6-35b-a3b-nvfp4"
    assert ServeConfig(model="x/y", served_name="hero").slug() == "hero"


# ---- Anthropic ↔ OpenAI translation ----
def test_anthropic_to_openai_system_and_messages():
    oai = _to_openai({"model": "m", "system": "be terse",
                      "messages": [{"role": "user", "content": "hi"}], "max_tokens": 50})
    assert oai["messages"][0] == {"role": "system", "content": "be terse"}
    assert oai["messages"][1] == {"role": "user", "content": "hi"}
    assert oai["max_tokens"] == 50


def test_openai_to_anthropic_shape():
    back = _to_anthropic({"choices": [{"message": {"content": "yo"}, "finish_reason": "stop"}],
                          "usage": {"prompt_tokens": 3, "completion_tokens": 1}}, "m")
    assert back["type"] == "message" and back["role"] == "assistant"
    assert back["content"][0] == {"type": "text", "text": "yo"}
    assert back["stop_reason"] == "end_turn"
    assert back["usage"]["output_tokens"] == 1


def test_anthropic_tool_use_translation():
    # Anthropic tool_use block → OpenAI tool_calls; tool_result → role:tool message
    oai = _to_openai({"model": "m", "max_tokens": 64, "messages": [
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "get", "input": {"q": 1}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "42"}]},
    ]})
    asst = [m for m in oai["messages"] if m["role"] == "assistant"][0]
    assert asst["tool_calls"][0]["function"]["name"] == "get"
    tool = [m for m in oai["messages"] if m["role"] == "tool"][0]
    assert tool["tool_call_id"] == "t1" and tool["content"] == "42"
    # OpenAI tool_calls back → Anthropic tool_use block
    back = _to_anthropic({"choices": [{"message": {"content": None, "tool_calls": [
        {"id": "t2", "function": {"name": "f", "arguments": '{"a":1}'}}]}, "finish_reason": "tool_calls"}],
        "usage": {}}, "m")
    tu = [b for b in back["content"] if b["type"] == "tool_use"][0]
    assert tu["name"] == "f" and tu["input"] == {"a": 1} and back["stop_reason"] == "tool_use"


def test_ollama_model_id_strips_tag(monkeypatch):
    from anima_thor_ui import ollama_api
    monkeypatch.setattr(ollama_api, "_served_name", lambda: "")   # no engine config
    assert ollama_api._model_id("qwen36:latest") == "qwen36"
    monkeypatch.setattr(ollama_api, "_served_name", lambda: "hero")  # route to what's served
    assert ollama_api._model_id("anything:7b") == "hero"


def test_model_size_gb_unknown_is_zero():
    # nonexistent repo/path → 0.0 (guard falls back), never raises
    assert vllm_manager.model_size_gb("nope/does-not-exist-123") == 0.0


def test_nvfp4_and_arch_detection():
    assert hf_models._is_nvfp4(["nvfp4", "modelopt"], "x/y")
    assert hf_models._is_nvfp4([], "nvidia/Foo-NVFP4")          # name hint
    assert not hf_models._is_nvfp4(["text-generation"], "meta/Llama-3-8B")
    assert hf_models._arch_of(["qwen3_5_moe", "safetensors"]) == "qwen3_5_moe"
    assert hf_models._arch_of(["unknown_arch"]) is None


def test_wait_mem_stable_returns_on_flat(monkeypatch):
    monkeypatch.setattr(vllm_manager.time, "sleep", lambda *_: None)   # no real waiting
    monkeypatch.setattr(vllm_manager, "mem_gb", lambda: (100.0, 128.0))  # perfectly flat
    assert vllm_manager.wait_mem_stable(max_wait=80) == 100.0


def test_optional_api_key_gate(monkeypatch):
    from fastapi.testclient import TestClient
    from anima_thor_ui import main
    from anima_thor_ui.config import settings
    monkeypatch.setattr(settings, "API_KEY", "secret")          # enable auth
    with TestClient(main.app) as c:
        assert c.get("/api/quantize").status_code == 401         # protected, no key
        assert c.get("/api/quantize", headers={"x-api-key": "secret"}).status_code == 200
        assert c.get("/api/version").status_code == 200          # Ollama probe stays open
