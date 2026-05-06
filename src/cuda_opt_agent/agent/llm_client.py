"""
LLM 客户端抽象 —— 使用 LangChain 封装,支持 Anthropic / OpenAI。
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# Prompt 模板目录
PROMPTS_DIR = Path(__file__).parent / "prompts"
DEFAULT_TEMPERATURE = 0.3


def _normalize_openai_base_url(base_url: str | None) -> str | None:
    """Normalize root gateway URLs to the OpenAI-compatible /v1 API path."""
    if not base_url:
        return None

    normalized = base_url.rstrip("/")
    parsed = urlsplit(normalized)
    if parsed.path in ("", "/"):
        return f"{normalized}/v1"
    return normalized


class LLMClient:
    """LLM 调用封装。"""

    def __init__(self, provider: str = "anthropic", model: str = "claude-sonnet-4-20250514"):
        self.provider = provider
        self.model = model
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
        """
        调用 LLM,返回文本响应。
        带自动重试。
        """
        llm = self._get_llm(temperature)
        logger.debug("LLM request length: %d chars", len(prompt))
        response = llm.invoke(prompt)
        text = self._response_to_text(response)
        logger.debug("LLM response length: %d chars", len(text))
        return text

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
    def invoke_structured(self, prompt: str, schema: type, temperature: float | None = None) -> Any:
        """
        调用 LLM 并解析结构化输出。
        使用 LangChain 的 with_structured_output。
        """
        if self.provider == "openai":
            text = self.invoke(prompt) if temperature is None else self.invoke(prompt, temperature=temperature)
            return self._manual_parse(text, schema)

        llm = self._get_llm(temperature)
        try:
            structured_llm = llm.with_structured_output(schema)
            result = structured_llm.invoke(prompt)
            return result
        except Exception as e:
            logger.warning("Structured output parsing failed; falling back to manual parsing: %s", e)
            # 降级: 调用普通 invoke 然后手动解析
            text = self.invoke(prompt) if temperature is None else self.invoke(prompt, temperature=temperature)
            return self._manual_parse(text, schema)

    def invoke_json(self, prompt: str, temperature: float | None = None) -> dict:
        """调用 LLM 并解析 JSON。"""
        text = self.invoke(prompt) if temperature is None else self.invoke(prompt, temperature=temperature)
        return self._extract_json(text)

    @staticmethod
    def _response_to_text(response: Any) -> str:
        """Extract plain text from LangChain messages across Chat and Responses APIs."""
        text_attr = getattr(response, "text", None)
        if isinstance(text_attr, str):
            return text_attr
        if callable(text_attr):
            text = text_attr()
            if isinstance(text, str):
                return text

        content = getattr(response, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                elif isinstance(item, str):
                    parts.append(item)
            if parts:
                return "".join(parts)

        return str(response)

    @staticmethod
    def _extract_json(text: str) -> dict:
        """从 LLM 输出中提取 JSON。"""
        import re

        # 尝试从 ```json ... ``` 块中提取
        json_match = re.search(r"```json\s*\n(.*?)```", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # 尝试直接解析
        try:
            # 找 { 或 [ 开头
            start = -1
            for i, ch in enumerate(text):
                if ch in "{[":
                    start = i
                    break
            if start >= 0:
                # 找到匹配的结束符
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
        # 对缺失的变量用空字符串填充
        import re
        placeholders = re.findall(r"\{(\w+)\}", template)
        for ph in placeholders:
            if ph not in kwargs:
                kwargs[ph] = ""
        return template.format(**kwargs)
