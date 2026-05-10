"""
配置加载 —— 从 .env 文件读取,构建 AgentConfig。

[优化] 新增字段加载: gpu_ids, correctness_max_parallel, nvcc_parallel_threads,
hp_llm_concurrency, use_code_diff, use_tool_use

[修复] 新增字段加载: hp_correctness_repair_max
"""

from __future__ import annotations

import json
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

    # [优化] 解析 GPU_IDS: 逗号分隔的 GPU 索引, 如 "0,1,2,3"
    gpu_ids_str = os.getenv("GPU_IDS", "")
    gpu_ids: list[int] = []
    if gpu_ids_str.strip():
        try:
            gpu_ids = [int(x.strip()) for x in gpu_ids_str.split(",") if x.strip()]
        except ValueError:
            gpu_ids = []

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
        ncu_launch_count=int(os.getenv("NCU_LAUNCH_COUNT", "3")),
        ncu_warmup_rounds=int(os.getenv("NCU_WARMUP_ROUNDS", "1")),
        ncu_profile_rounds=int(os.getenv("NCU_PROFILE_ROUNDS", "1")),
        multi_shape_aggregator=os.getenv("MULTI_SHAPE_AGGREGATOR", "mean"),
        runs_dir=os.getenv("RUNS_DIR", "runs"),
        knowledge_base_dir=os.getenv("KNOWLEDGE_BASE_DIR", "knowledge_base"),
        # [优化] 新增字段
        gpu_ids=gpu_ids,
        correctness_max_parallel=int(os.getenv("CORRECTNESS_MAX_PARALLEL", "2")),
        nvcc_parallel_threads=int(os.getenv("NVCC_PARALLEL_THREADS", "0")),
        hp_llm_concurrency=int(os.getenv("HP_LLM_CONCURRENCY", "3")),
        use_code_diff=os.getenv("USE_CODE_DIFF", "true").strip().lower() in {"1", "true", "yes", "on"},
        use_tool_use=os.getenv("USE_TOOL_USE", "true").strip().lower() in {"1", "true", "yes", "on"},
        # [修复] 新增字段
        hp_correctness_repair_max=int(os.getenv("HP_CORRECTNESS_REPAIR_MAX", "2")),
        enable_library_comparison=os.getenv("ENABLE_LIBRARY_COMPARISON", "true").strip().lower()
        in {"1", "true", "yes", "on"},
        enable_web_search_baseline=os.getenv("ENABLE_WEB_SEARCH_BASELINE", "true").strip().lower()
        in {"1", "true", "yes", "on"},
        bootstrap_web_search_max_calls=int(os.getenv("BOOTSTRAP_WEB_SEARCH_MAX_CALLS", "20")),
        bootstrap_web_search_max_results=int(os.getenv("BOOTSTRAP_WEB_SEARCH_MAX_RESULTS", "12")),
        bootstrap_web_search_per_query_results=int(os.getenv("BOOTSTRAP_WEB_SEARCH_PER_QUERY_RESULTS", "3")),
        web_search_on_failure_threshold=int(os.getenv("WEB_SEARCH_ON_FAILURE_THRESHOLD", "2")),
        launch_floor_ms=float(os.getenv("LAUNCH_FLOOR_MS", "0.005")),
        catastrophic_regression_threshold=float(os.getenv("CATASTROPHIC_REGRESSION_THRESHOLD", "3.0")),
        catastrophic_streak_limit=int(os.getenv("CATASTROPHIC_STREAK_LIMIT", "2")),
        tiny_kernel_reject_limit=int(os.getenv("TINY_KERNEL_REJECT_LIMIT", "3")),
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
