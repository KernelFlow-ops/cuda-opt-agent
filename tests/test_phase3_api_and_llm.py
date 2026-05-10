"""
Phase 3 测试 —— LLM API 连通性 + Prompt 模板 + 结构化输出。
标记 @pytest.mark.api 的测试需要真实 API key。
"""

import asyncio
import json
import os
import sys
import types

import pytest


def run_async(coro):
    return asyncio.run(coro)


def _load_api_env():
    from dotenv import load_dotenv

    load_dotenv(".env", override=False)


class TestPromptTemplates:
    """验证所有 Prompt 模板存在且格式正确。"""

    REQUIRED_TEMPLATES = [
        "bootstrap.md",
        "analyze.md",
        "decide_method.md",
        "propose_hp.md",
        "apply_method.md",
        "repair_compile.md",
        "reflect_success.md",
        "reflect_failure.md",
    ]

    def test_all_templates_exist(self):
        from cuda_opt_agent.agent.llm_client import PROMPTS_DIR
        for name in self.REQUIRED_TEMPLATES:
            path = PROMPTS_DIR / name
            assert path.exists(), f"Prompt 模板缺失: {name}"

    def test_templates_have_placeholders(self):
        from cuda_opt_agent.agent.llm_client import PROMPTS_DIR
        for name in self.REQUIRED_TEMPLATES:
            content = (PROMPTS_DIR / name).read_text(encoding="utf-8")
            assert len(content) > 50, f"模板 {name} 内容过短"
            # 至少应该有一些占位符
            assert "{" in content, f"模板 {name} 没有占位符"


class TestLLMClient:
    def test_init(self):
        from cuda_opt_agent.agent.llm_client import LLMClient
        client = LLMClient(provider="anthropic", model="claude-sonnet-4-20250514")
        assert client.provider == "anthropic"

    def test_load_prompt(self):
        from cuda_opt_agent.agent.llm_client import LLMClient
        client = LLMClient()
        prompt = client.load_prompt("bootstrap.md")
        assert "正确性优先" in prompt

    def test_load_nonexistent_prompt(self):
        from cuda_opt_agent.agent.llm_client import LLMClient
        client = LLMClient()
        with pytest.raises(FileNotFoundError):
            client.load_prompt("nonexistent.md")

    def test_format_prompt(self):
        from cuda_opt_agent.agent.llm_client import LLMClient
        client = LLMClient()
        formatted = client.format_prompt(
            "bootstrap.md",
            operator_name="gemm",
            signature="C = A @ B",
            dtypes='{"A": "fp16"}',
            shapes='{"A": [4096, 4096]}',
            constraints="无",
            gpu_name="A100",
            compute_capability="sm_80",
            sm_count=108,
            shared_mem_per_block_kb=164,
            l2_cache_mb=40,
            has_tensor_cores=True,
            cuda_version="12.4",
        )
        assert "gemm" in formatted
        assert "A100" in formatted

    def test_extract_json_from_code_block(self):
        from cuda_opt_agent.agent.llm_client import LLMClient
        text = '''Here is my analysis:
```json
{"method_name": "tiling", "has_hyperparams": true}
```
That's my recommendation.'''
        result = LLMClient._extract_json(text)
        assert result["method_name"] == "tiling"

    def test_extract_json_bare(self):
        from cuda_opt_agent.agent.llm_client import LLMClient
        text = 'Some text {"key": "value"} more text'
        result = LLMClient._extract_json(text)
        assert result["key"] == "value"

    def test_extract_json_nested(self):
        from cuda_opt_agent.agent.llm_client import LLMClient
        text = '{"outer": {"inner": [1, 2, 3]}, "b": true}'
        result = LLMClient._extract_json(text)
        assert result["outer"]["inner"] == [1, 2, 3]

    def test_extract_json_failure(self):
        from cuda_opt_agent.agent.llm_client import LLMClient
        with pytest.raises(ValueError):
            LLMClient._extract_json("no json here at all")

    def test_response_to_text_from_responses_content_blocks(self):
        from cuda_opt_agent.agent.llm_client import LLMClient

        class Message:
            content = [
                {"type": "text", "text": "OK", "annotations": []},
                {"type": "text", "text": " DONE", "annotations": []},
            ]

        assert LLMClient._response_to_text(Message()) == "OK DONE"

    def test_response_to_text_from_content_blocks_attribute(self):
        from cuda_opt_agent.agent.llm_client import LLMClient

        class Block:
            def __init__(self, text):
                self.text = text

        class Message:
            content = []
            content_blocks = [Block("A"), {"type": "text", "text": "B"}]

        assert LLMClient._response_to_text(Message()) == "AB"

    def test_astream_text_sends_tokens_to_sink(self, monkeypatch):
        from cuda_opt_agent.agent.llm_client import LLMClient

        class FakeLLM:
            async def astream(self, prompt):
                yield "OK"
                yield " DONE"

        class Sink:
            def __init__(self):
                self.events = []

            def start_node(self, node_name):
                self.events.append(("start", node_name))

            def on_token(self, chunk):
                self.events.append(("token", chunk))

            def finish_node(self, summary=""):
                self.events.append(("finish", summary))

            def on_error(self, error):
                self.events.append(("error", str(error)))

        sink = Sink()
        client = LLMClient(stream_sink=sink)
        monkeypatch.setattr(client, "_get_llm", lambda temperature=None: FakeLLM())

        text = run_async(client.astream_text("prompt", temperature=0.4, node_name="analyze"))

        assert text == "OK DONE"
        assert sink.events[:3] == [
            ("start", "analyze"),
            ("token", "OK"),
            ("token", " DONE"),
        ]
        assert sink.events[-1][0] == "finish"

    def test_ainvoke_json_streams_then_parses(self, monkeypatch):
        from cuda_opt_agent.agent.llm_client import LLMClient

        class FakeLLM:
            async def astream(self, prompt):
                yield '{"ok":'
                yield ' true}'

        client = LLMClient(use_tool_use=False)
        monkeypatch.setattr(client, "_get_llm", lambda temperature=None: FakeLLM())

        assert run_async(client.ainvoke_json("prompt", temperature=0.8, node_name="decide")) == {"ok": True}

    def test_ainvoke_json_uses_tool_use_when_enabled(self, monkeypatch):
        from cuda_opt_agent.agent.llm_client import LLMClient

        seen = {}

        class FakeStructuredLLM:
            async def ainvoke(self, prompt):
                seen["prompt"] = prompt
                return {"payload": {"ok": True}}

        class FakeLLM:
            def with_structured_output(self, schema):
                seen["schema"] = schema
                return FakeStructuredLLM()

        client = LLMClient(use_tool_use=True)
        monkeypatch.setattr(client, "_get_llm", lambda temperature=None: FakeLLM())

        assert run_async(client.ainvoke_json("prompt", temperature=0.8, node_name="decide")) == {"ok": True}
        assert seen["prompt"] == "prompt"
        assert seen["schema"].__name__ == "_JSONToolPayload"

    def test_openai_client_uses_responses_api(self, monkeypatch):
        from cuda_opt_agent.agent.llm_client import LLMClient

        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        client = LLMClient(provider="openai", model="gpt-4o-mini")
        llm = client._get_llm()

        assert llm.use_responses_api is True
        assert llm.output_version == "responses/v1"
        assert llm._use_responses_api({}) is True

    def test_openai_structured_output_uses_manual_parse(self, monkeypatch):
        from cuda_opt_agent.agent.llm_client import LLMClient

        client = LLMClient(provider="openai", model="gpt-4o-mini")
        monkeypatch.setattr(client, "invoke", lambda prompt: '{"status": "ok"}')

        assert client.invoke_structured("return json", dict) == {"status": "ok"}

    def test_get_llm_caches_by_temperature(self, monkeypatch):
        from cuda_opt_agent.agent.llm_client import LLMClient

        created = []

        class FakeChatAnthropic:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                created.append(self)

        module = types.SimpleNamespace(ChatAnthropic=FakeChatAnthropic)
        monkeypatch.setitem(sys.modules, "langchain_anthropic", module)

        client = LLMClient(provider="anthropic", model="fake-model")
        low_a = client._get_llm(temperature=0.1)
        low_b = client._get_llm(temperature=0.1)
        high = client._get_llm(temperature=0.8)

        assert low_a is low_b
        assert low_a is not high
        assert [llm.kwargs["temperature"] for llm in created] == [0.1, 0.8]

    def test_invoke_json_passes_temperature_to_invoke(self, monkeypatch):
        from cuda_opt_agent.agent.llm_client import LLMClient

        client = LLMClient()
        seen = {}

        def fake_invoke(prompt, temperature=None):
            seen["temperature"] = temperature
            return '{"ok": true}'

        monkeypatch.setattr(client, "invoke", fake_invoke)

        assert client.invoke_json("prompt", temperature=0.8) == {"ok": True}
        assert seen["temperature"] == 0.8


@pytest.mark.api
class TestLLMApiConnectivity:
    """
    真实 API 连通性测试。
    需要设置环境变量 ANTHROPIC_API_KEY 或 OPENAI_API_KEY。
    运行: pytest -m api
    """

    def test_anthropic_api_ping(self):
        """验证 Anthropic API 可连通。"""
        _load_api_env()
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY 未设置")

        from cuda_opt_agent.agent.llm_client import LLMClient
        client = LLMClient(provider="anthropic")
        response = client.invoke("回复 'OK' 两个字母,不要其他内容。")
        assert len(response) > 0
        assert "OK" in response.upper()

    def test_openai_api_ping(self):
        """验证 OpenAI API 可连通。"""
        _load_api_env()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY 未设置")

        from cuda_opt_agent.agent.llm_client import LLMClient
        model = os.getenv("OPENAI_MODEL", os.getenv("LLM_MODEL", "gpt-4o-mini"))
        client = LLMClient(provider="openai", model=model)
        assert client._get_llm()._use_responses_api({}) is True
        response = client.invoke("Reply with just 'OK'.")
        assert len(response) > 0

    def test_structured_output(self):
        """验证结构化输出功能。"""
        _load_api_env()
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY 未设置")

        from cuda_opt_agent.agent.llm_client import LLMClient
        client = LLMClient(provider="anthropic")
        result = client.invoke_json(
            '请只输出以下 JSON: {"status": "ok", "count": 42}'
        )
        assert result["status"] == "ok"
        assert result["count"] == 42
