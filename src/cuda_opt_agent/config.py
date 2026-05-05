"""
配置加载 —— 从 .env 文件读取,构建 AgentConfig。
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from .models.data import AgentConfig


def load_config(env_path: str | Path | None = None) -> AgentConfig:
    """
    加载 .env 并返回 AgentConfig。
    优先级: 环境变量 > .env 文件 > 默认值。
    """
    if env_path is None:
        # 从项目根目录向上查找 .env
        candidates = [
            Path.cwd() / ".env",
            Path(__file__).resolve().parent.parent.parent / ".env",
        ]
        for c in candidates:
            if c.exists():
                env_path = c
                break

    if env_path and Path(env_path).exists():
        load_dotenv(env_path, override=False)

    llm_provider = os.getenv("LLM_PROVIDER", "anthropic")
    if llm_provider == "openai":
        llm_model = os.getenv("OPENAI_MODEL", os.getenv("LLM_MODEL", "gpt-4o"))
    else:
        llm_model = os.getenv("ANTHROPIC_MODEL", os.getenv("LLM_MODEL", "claude-sonnet-4-20250514"))

    return AgentConfig(
        llm_provider=llm_provider,
        llm_model=llm_model,
        default_dtype=os.getenv("DEFAULT_DTYPE", "fp16"),
        max_iterations=int(os.getenv("MAX_ITERATIONS", "30")),
        consecutive_reject_limit=int(os.getenv("CONSECUTIVE_REJECT_LIMIT", "5")),
        accept_epsilon=float(os.getenv("ACCEPT_EPSILON", "0.005")),
        compile_repair_max_retries=int(os.getenv("COMPILE_REPAIR_MAX_RETRIES", "3")),
        decide_reselect_max_retries=int(os.getenv("DECIDE_RESELECT_MAX_RETRIES", "3")),
        hp_candidate_count=int(os.getenv("HP_CANDIDATE_COUNT", "5")),
        hp_compile_workers=int(os.getenv("HP_COMPILE_WORKERS", "0")),
        benchmark_warmup_rounds=int(os.getenv("BENCHMARK_WARMUP_ROUNDS", "10")),
        benchmark_measure_rounds=int(os.getenv("BENCHMARK_MEASURE_ROUNDS", "100")),
        multi_shape_aggregator=os.getenv("MULTI_SHAPE_AGGREGATOR", "mean"),
        runs_dir=os.getenv("RUNS_DIR", "runs"),
        knowledge_base_dir=os.getenv("KNOWLEDGE_BASE_DIR", "knowledge_base"),
    )


def get_api_key(provider: str | None = None) -> str:
    """返回当前 LLM provider 的 API key。"""
    provider = provider or os.getenv("LLM_PROVIDER", "anthropic")
    if provider == "anthropic":
        key = os.getenv("ANTHROPIC_API_KEY", "")
    elif provider == "openai":
        key = os.getenv("OPENAI_API_KEY", "")
    else:
        key = os.getenv(f"{provider.upper()}_API_KEY", "")
    return key
