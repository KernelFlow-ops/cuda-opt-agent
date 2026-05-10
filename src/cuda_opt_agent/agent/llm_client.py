"""
LLM 客户端抽象 —— 使用 LangChain 封装,支持 Anthropic / OpenAI。

[优化]:
  - ainvoke_tool_use: Tool Use (function calling) 替代自由 JSON 输出
  - ainvoke_structured 优先走 tool use 路径
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

# Prompt 模板目录
PROMPTS_DIR = Path(__file__).parent / "prompts"
DEFAULT_TEMPERATURE = 0.3


class _JSONToolPayload(BaseModel):
    payload: Any = Field(description="The exact JSON object or array requested by the prompt.")


class LLMStreamSink(Protocol):
    """Receives live LLM output for UI rendering."""

    def start_node(self, node_name: str) -> None:
        ...

    def on_token(self, chunk: str) -> None:
        ...

    def finish_node(self, summary: str = "") -> None:
        ...

    def on_error(self, error: BaseException | str) -> None:
        ...


async def _maybe_call(target: Any, method_name: str, *args: Any) -> None:
    if target is None:
        return
    method = getattr(target, method_name, None)
    if method is None:
        return
    result = method(*args)
    if inspect.isawaitable(result):
        await result


def _run_async_compat(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError("Cannot call synchronous LLM API from a running event loop; use the async API instead.")


def _normalize_openai_base_url(base_url: str | None) -> str | None:
    """Normalize root gateway URLs to the OpenAI-compatible /v1 API path."""
    if not base_url:
        return None

    normalized = base_url.rstrip("/")
    parsed = urlsplit(normalized)
    if parsed.path in ("", "/"):
        return f"{normalized}/v1"
    return normalized


# [优化] 可重试的异常类型
_RETRYABLE_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    OSError,
)


class LLMClient:
    """LLM 调用封装。"""

    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-sonnet-4-20250514",
        stream_sink: LLMStreamSink | None = None,
        use_tool_use: bool = True,
    ):
        self.provider = provider
        self.model = model
        self.stream_sink = stream_sink
        self.use_tool_use = use_tool_use
        self._llm_cache: dict[float, Any] = {}

    def _get_llm(self, temperature: float | None = None):
        """延迟初始化 LLM 实例。"""
        temperature = DEFAULT_TEMPERATURE if temperature is None else temperature
        if temperature in self._llm_cache:
            return self._llm_cache[temperature]

        if self.provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(
                model=self.model,
                temperature=temperature,
                max_tokens=100000,
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                base_url=os.getenv("ANTHROPIC_BASE_URL"),
            )
        elif self.provider == "openai":
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model=self.model,
                temperature=temperature,
                max_tokens=100000,
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=_normalize_openai_base_url(os.getenv("OPENAI_BASE_URL")),
                use_responses_api=True,
                output_version="responses/v1",
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

        self._llm_cache[temperature] = llm
        return llm

    def load_prompt(self, template_name: str) -> str:
        """加载 Prompt 模板。"""
        path = PROMPTS_DIR / template_name
        if not path.exists():
            raise FileNotFoundError(f"Prompt template not found: {path}")
        return path.read_text(encoding="utf-8")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
    def invoke(self, prompt: str, temperature: float | None = None) -> str:
        """调用 LLM,返回文本响应。"""
        if self.stream_sink is not None:
            return _run_async_compat(self.ainvoke(prompt, temperature=temperature))

        llm = self._get_llm(temperature)
        logger.debug("LLM request length: %d chars", len(prompt))
        response = llm.invoke(prompt)
        text = self._response_to_text(response)
        logger.debug("LLM response length: %d chars", len(text))
        return text

    async def ainvoke(
        self,
        prompt: str,
        temperature: float | None = None,
        *,
        node_name: str | None = None,
        stream_sink: LLMStreamSink | None = None,
    ) -> str:
        """Async LLM call. Uses streaming when a sink is available."""
        return await self.astream_text(
            prompt,
            temperature=temperature,
            node_name=node_name,
            stream_sink=stream_sink,
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
    async def astream_text(
        self,
        prompt: str,
        temperature: float | None = None,
        *,
        node_name: str | None = None,
        stream_sink: LLMStreamSink | None = None,
    ) -> str:
        """Stream an LLM response token-by-token and return the complete text."""
        llm = self._get_llm(temperature)
        sink = stream_sink if stream_sink is not None else self.stream_sink
        logger.debug("LLM request length: %d chars", len(prompt))

        if sink is not None and node_name:
            await _maybe_call(sink, "start_node", node_name)

        parts: list[str] = []
        try:
            async for chunk in llm.astream(prompt):
                text = self._chunk_to_text(chunk)
                if not text:
                    continue
                parts.append(text)
                await _maybe_call(sink, "on_token", text)
        except NotImplementedError:
            response = await llm.ainvoke(prompt)
            text = self._response_to_text(response)
            parts.append(text)
            await _maybe_call(sink, "on_token", text)
        except Exception as e:
            await _maybe_call(sink, "on_error", e)
            raise

        text = "".join(parts)
        logger.debug("LLM response length: %d chars", len(text))
        if sink is not None and node_name:
            await _maybe_call(sink, "finish_node", f"{len(text)} chars")
        return text

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
    def invoke_structured(self, prompt: str, schema: type, temperature: float | None = None) -> Any:
        """调用 LLM 并解析结构化输出。"""
        if not self.use_tool_use or self.provider == "openai":
            text = self.invoke(prompt) if temperature is None else self.invoke(prompt, temperature=temperature)
            return self._manual_parse(text, schema)

        llm = self._get_llm(temperature)
        try:
            structured_llm = llm.with_structured_output(schema)
            result = structured_llm.invoke(prompt)
            return result
        except Exception as e:
            logger.warning("Structured output parsing failed; falling back to manual parsing: %s", e)
            text = self.invoke(prompt) if temperature is None else self.invoke(prompt, temperature=temperature)
            return self._manual_parse(text, schema)

    def invoke_json(self, prompt: str, temperature: float | None = None) -> dict:
        """调用 LLM 并解析 JSON。"""
        text = self.invoke(prompt) if temperature is None else self.invoke(prompt, temperature=temperature)
        return self._extract_json(text)

    async def ainvoke_structured(
        self,
        prompt: str,
        schema: type,
        temperature: float | None = None,
        *,
        node_name: str | None = None,
        stream_sink: LLMStreamSink | None = None,
    ) -> Any:
        """Async structured output via streamed raw text followed by schema parsing."""
        if self.use_tool_use:
            return await self.ainvoke_tool_use(
                prompt,
                schema,
                temperature=temperature,
                node_name=node_name,
                stream_sink=stream_sink,
            )
        text = await self.ainvoke(
            prompt,
            temperature=temperature,
            node_name=node_name,
            stream_sink=stream_sink,
        )
        return self._manual_parse(text, schema)

    async def ainvoke_json(
        self,
        prompt: str,
        temperature: float | None = None,
        *,
        node_name: str | None = None,
        stream_sink: LLMStreamSink | None = None,
    ) -> dict:
        """Async LLM call with streamed raw output followed by JSON parsing."""
        if self.use_tool_use:
            try:
                llm = self._get_llm(temperature)
                sink = stream_sink if stream_sink is not None else self.stream_sink
                if sink is not None and node_name:
                    await _maybe_call(sink, "start_node", node_name)
                structured_llm = llm.with_structured_output(_JSONToolPayload)
                result = await structured_llm.ainvoke(prompt)
                if sink is not None and node_name:
                    await _maybe_call(sink, "finish_node", "tool_use JSON OK")
                if isinstance(result, _JSONToolPayload):
                    return result.payload
                if isinstance(result, dict) and "payload" in result:
                    return result["payload"]
                if hasattr(result, "payload"):
                    return result.payload
                return result
            except Exception as e:
                logger.warning(
                    "Tool Use JSON failed for %s; falling back to text JSON parsing: %s",
                    node_name or "unknown", e,
                )
        text = await self.ainvoke(
            prompt,
            temperature=temperature,
            node_name=node_name,
            stream_sink=stream_sink,
        )
        return self._extract_json(text)

    # ════════════════════════════════════════
    # [优化] Tool Use (function calling) 支持
    # ════════════════════════════════════════

    async def ainvoke_tool_use(
        self,
        prompt: str,
        schema: type,
        temperature: float | None = None,
        *,
        node_name: str | None = None,
        stream_sink: LLMStreamSink | None = None,
    ) -> Any:
        """
        [优化] 使用 Tool Use (function calling) 获取结构化输出。

        优先尝试 with_structured_output (tool use), 失败回退到手动 JSON 解析。
        """
        llm = self._get_llm(temperature)
        sink = stream_sink if stream_sink is not None else self.stream_sink

        if sink is not None and node_name:
            await _maybe_call(sink, "start_node", node_name)

        try:
            # 尝试使用 LangChain 的 with_structured_output (内部使用 tool use)
            structured_llm = llm.with_structured_output(schema)
            result = await structured_llm.ainvoke(prompt)

            if sink is not None and node_name:
                await _maybe_call(sink, "finish_node", "tool_use OK")
            return result

        except Exception as e:
            logger.warning("Tool Use failed for %s; falling back to JSON parsing: %s",
                           node_name or "unknown", e)

            # 回退到流式文本 + 手动解析
            text = await self.astream_text(
                prompt,
                temperature=temperature,
                node_name=node_name,
                stream_sink=stream_sink,
            )
            return self._manual_parse(text, schema)

    # ════════════════════════════════════════
    # 内部工具函数
    # ════════════════════════════════════════

    @staticmethod
    def _response_to_text(response: Any) -> str:
        """Extract plain text from LangChain messages across Chat and Responses APIs."""
        if isinstance(response, str):
            return response

        text_attr = getattr(response, "text", None)
        if isinstance(text_attr, str):
            return text_attr
        if callable(text_attr):
            text = text_attr()
            if isinstance(text, str):
                return text

        content_text = LLMClient._content_to_text(getattr(response, "content", None))
        if content_text:
            return content_text

        content_blocks_text = LLMClient._content_to_text(getattr(response, "content_blocks", None))
        if content_blocks_text:
            return content_blocks_text

        return str(response)

    @staticmethod
    def _chunk_to_text(chunk: Any) -> str:
        """Extract text from streaming chunks."""
        return LLMClient._response_to_text(chunk)

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""

        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                continue
            text = getattr(item, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)

    @staticmethod
    def _extract_json(text: str) -> dict:
        """从 LLM 输出中提取 JSON。"""
        import re

        json_match = re.search(r"```json\s*\n(.*?)```", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        try:
            start = -1
            for i, ch in enumerate(text):
                if ch in "{[":
                    start = i
                    break
            if start >= 0:
                depth = 0
                in_string = False
                escape = False
                for i in range(start, len(text)):
                    ch = text[i]
                    if escape:
                        escape = False
                        continue
                    if ch == "\\":
                        escape = True
                        continue
                    if ch == '"' and not escape:
                        in_string = not in_string
                        continue
                    if in_string:
                        continue
                    if ch in "{[":
                        depth += 1
                    elif ch in "}]":
                        depth -= 1
                        if depth == 0:
                            return json.loads(text[start:i + 1])
        except json.JSONDecodeError:
            pass

        raise ValueError(f"Could not extract JSON from LLM output: {text[:500]}")

    @staticmethod
    def _manual_parse(text: str, schema: type) -> Any:
        """手动解析 LLM 输出为指定 schema。"""
        data = LLMClient._extract_json(text)
        if hasattr(schema, "model_validate"):
            return schema.model_validate(data)
        return data

    def format_prompt(self, template_name: str, **kwargs) -> str:
        """加载模板并填充变量。"""
        template = self.load_prompt(template_name)
        import re
        placeholders = re.findall(r"\{(\w+)\}", template)
        for ph in placeholders:
            if ph not in kwargs:
                kwargs[ph] = ""
        return template.format(**kwargs)
