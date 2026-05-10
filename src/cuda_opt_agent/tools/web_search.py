"""
外部知识搜索工具 —— 集成 exa/web_search 获取高性能 CUDA 实现参考。

搜索优先级:
  1. NVIDIA 官方文档 (docs.nvidia.com)
  2. CUTLASS / CUB GitHub
  3. FlashAttention / Triton
  4. CUDA 博客和论文
"""
from __future__ import annotations
import asyncio
import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

NVIDIA_DOCS_DOMAINS = ["docs.nvidia.com", "developer.nvidia.com"]
CUDA_OSS_DOMAINS = [
    "github.com/NVIDIA/cutlass", "github.com/NVIDIA/cub",
    "github.com/Dao-AILab/flash-attention", "github.com/triton-lang/triton",
]
CUDA_BLOG_DOMAINS = ["developer.nvidia.com/blog", "arxiv.org", "siboehm.com"]
BOOTSTRAP_SEARCH_HARD_CALL_LIMIT = 20


class ExaSearchClient:
    """Exa AI 搜索客户端（可选依赖）。"""
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("EXA_API_KEY", "")
        self._client = None

    def _ensure_client(self):
        if self._client is None and self.api_key:
            try:
                from exa_py import Exa
                self._client = Exa(api_key=self.api_key)
            except ImportError:
                logger.warning("EXA_API_KEY is set but exa_py is not installed; pip install exa_py")
            except Exception as e:
                logger.warning("Exa init failed: %s", e)
        return self._client

    async def search(self, query: str, *, num_results: int = 5,
                     include_domains: list[str] | None = None) -> list[dict[str, str]]:
        client = self._ensure_client()
        if not client:
            return []
        try:
            logger.info("Exa search: %s", query)
            kw: dict[str, Any] = {"query": query, "num_results": num_results, "type": "auto"}
            if include_domains:
                kw["include_domains"] = include_domains
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, lambda: client.search_and_contents(**kw, text={"max_characters": 2000}))
            rows = [{"title": r.title or "", "url": r.url or "",
                     "text": (r.text or "")[:2000]} for r in (result.results if result else [])]
            logger.info("Exa search returned %d result(s)", len(rows))
            return rows
        except Exception as e:
            logger.warning("Exa search failed: %s", e)
            return []


class WebSearchFallback:
    """Tavily / SerpAPI 兜底。"""
    def __init__(self):
        self._tavily_key = os.getenv("TAVILY_API_KEY", "")
        self._serpapi_key = os.getenv("SERPAPI_API_KEY", "")

    async def search(self, query: str, num_results: int = 5) -> list[dict[str, str]]:
        if self._tavily_key:
            return await self._tavily(query, num_results)
        if self._serpapi_key:
            return await self._serpapi(query, num_results)
        return []

    async def _tavily(self, query: str, n: int) -> list[dict[str, str]]:
        try:
            logger.info("Tavily search: %s", query)
            from tavily import TavilyClient
            c = TavilyClient(api_key=self._tavily_key)
            loop = asyncio.get_running_loop()
            res = await loop.run_in_executor(None, lambda: c.search(query, max_results=n))
            rows = [{"title": r.get("title", ""), "url": r.get("url", ""),
                     "text": r.get("content", "")[:2000]} for r in res.get("results", [])]
            logger.info("Tavily search returned %d result(s)", len(rows))
            return rows
        except Exception as e:
            logger.warning("Tavily search failed: %s", e)
            return []

    async def _serpapi(self, query: str, n: int) -> list[dict[str, str]]:
        try:
            logger.info("SerpAPI search: %s", query)
            import httpx
            async with httpx.AsyncClient() as client:
                resp = await client.get("https://serpapi.com/search",
                    params={"q": query, "api_key": self._serpapi_key, "num": n, "engine": "google"},
                    timeout=15)
                data = resp.json()
                rows = [{"title": r.get("title", ""), "url": r.get("link", ""),
                         "text": r.get("snippet", "")}
                        for r in data.get("organic_results", [])[:n]]
                logger.info("SerpAPI search returned %d result(s)", len(rows))
                return rows
        except Exception as e:
            logger.warning("SerpAPI failed: %s", e)
            return []


SUBSPACE_KEYWORDS: dict[str, str] = {
    "memory-coalescing": "coalesced memory access global memory optimization",
    "shared-mem-tiling": "shared memory tiling blocking CUDA",
    "bank-conflict-resolution": "shared memory bank conflict padding swizzle",
    "vectorized-memory-access": "vectorized load store float4 __ldg CUDA",
    "async-memory-pipeline": "cp.async TMA asynchronous copy pipeline CUDA",
    "l2-cache-tuning": "L2 cache persistence policy cudaAccessPolicyWindow",
    "register-optimization": "register pressure launch_bounds maxrregcount CUDA",
    "texture-constant-memory": "texture memory constant memory CUDA",
    "occupancy-tuning": "occupancy block size CUDA optimization",
    "cta-redistribution": "persistent CTA thread block cluster split-K CUDA",
    "thread-coarsening": "thread coarsening grid-stride loop ILP CUDA",
    "warp-specialization": "warp specialization producer consumer pipeline Hopper",
    "instruction-optimization": "fast math intrinsics FMA rsqrt CUDA",
    "control-flow-divergence": "warp divergence branch predication CUDA",
    "precision-conversion": "mixed precision FP16 BF16 TF32 FP8 tensor core",
    "warp-primitive": "warp shuffle reduction cooperative groups CUDA",
    "reduction-restructure": "parallel reduction tree warp shuffle CUDA",
    "algorithm-replacement": "CUTLASS cuBLAS high performance CUDA",
    "fusion": "kernel fusion epilogue fusion vertical horizontal CUDA",
    "launch-overhead-mitigation": "CUDA graphs persistent kernel launch overhead",
}


async def search_cuda_knowledge(
    operator: str, subspace: str, *, context: str = "", max_results: int = 5,
) -> list[dict[str, str]]:
    """搜索 CUDA 优化相关外部知识。"""
    exa, web = ExaSearchClient(), WebSearchFallback()
    all_results: list[dict[str, str]] = []

    kw = SUBSPACE_KEYWORDS.get(subspace, f"{operator} {subspace} CUDA optimization")
    doc_query = f"{operator} {kw}"

    if exa.api_key:
        all_results.extend(await exa.search(doc_query, num_results=3, include_domains=NVIDIA_DOCS_DOMAINS))
        all_results.extend(await exa.search(f"CUDA {operator} kernel {subspace} implementation",
                                             num_results=3, include_domains=CUDA_OSS_DOMAINS))
    if not all_results:
        all_results.extend(await web.search(doc_query, num_results=max_results))
    if context:
        extra_q = f"CUDA {operator} {context[:100]}"
        r = await exa.search(extra_q, num_results=2) if exa.api_key else await web.search(extra_q, 2)
        all_results.extend(r)

    return _dedup(all_results)[:max_results]


def _compact_context(value: Any, max_chars: int = 180) -> str:
    if not value:
        return ""
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        text = str(value)
    return " ".join(text.split())[:max_chars]


def _unique_queries(queries: list[tuple[str, list[str] | None]]) -> list[tuple[str, list[str] | None]]:
    seen: set[str] = set()
    out: list[tuple[str, list[str] | None]] = []
    for query, domains in queries:
        normalized = " ".join(query.split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append((normalized, domains))
    return out


def _baseline_reference_queries(
    operator: str,
    dtype: str,
    *,
    task_description: str = "",
    shapes: Any = None,
    shape_profiles: Any = None,
    hardware_context: str = "",
) -> list[tuple[str, list[str] | None]]:
    task = _compact_context(task_description, 120)
    shape_text = _compact_context(shape_profiles or shapes, 180)
    hw = _compact_context(hardware_context, 120)
    context = " ".join(part for part in (task, shape_text, hw) if part)
    context_suffix = f" {context}" if context else ""
    oss_domains = CUDA_OSS_DOMAINS + CUDA_BLOG_DOMAINS
    all_cuda_domains = NVIDIA_DOCS_DOMAINS + CUDA_OSS_DOMAINS + CUDA_BLOG_DOMAINS

    queries: list[tuple[str, list[str] | None]] = [
        (f"CUTLASS {operator} kernel {dtype} implementation example", CUDA_OSS_DOMAINS),
        (f"CUDA {operator} high performance optimized kernel {dtype}", oss_domains),
        (f"NVIDIA {operator} best practices optimization CUDA", NVIDIA_DOCS_DOMAINS + CUDA_BLOG_DOMAINS),
        (f"CUDA {operator} {dtype} optimization methods{context_suffix}", all_cuda_domains),
        (f"{operator} CUDA kernel optimization shared memory vectorized warp reduction {dtype}", oss_domains),
        (f"{operator} CUDA common pitfalls numerical accuracy performance {dtype}", all_cuda_domains),
        (f"CUB CUDA {operator} reduction implementation {dtype}", CUDA_OSS_DOMAINS),
        (f"Triton {operator} kernel optimized {dtype} implementation", ["github.com/triton-lang/triton", "triton-lang.org"]),
        (f"CUDA {operator} warp shuffle reduction optimized kernel", oss_domains),
        (f"CUDA {operator} vectorized memory access half2 {dtype}", all_cuda_domains),
        (f"CUDA {operator} shared memory tiling bank conflict optimization", all_cuda_domains),
        (f"CUDA {operator} occupancy tuning register pressure launch bounds", all_cuda_domains),
        (f"CUDA {operator} persistent CTA grid stride loop optimization", all_cuda_domains),
        (f"CUDA {operator} launch overhead mitigation CUDA graphs", all_cuda_domains),
        (f"CUDA {operator} async memory pipeline cp.async optimization", all_cuda_domains),
        (f"CUDA {operator} instruction optimization fast math intrinsics {dtype}", all_cuda_domains),
        (f"GitHub CUDA {operator} optimized kernel {dtype}", CUDA_OSS_DOMAINS),
        (f"NVIDIA developer blog CUDA {operator} optimization", NVIDIA_DOCS_DOMAINS + CUDA_BLOG_DOMAINS),
        (f"arxiv CUDA {operator} kernel optimization {dtype}", CUDA_BLOG_DOMAINS),
        (f"CUDA {operator} benchmark optimized implementation source code", oss_domains),
    ]
    return _unique_queries(queries)


async def search_for_baseline_reference(
    operator: str,
    *,
    dtype: str = "fp16",
    max_results: int = 12,
    max_calls: int = 20,
    per_query_results: int = 3,
    task_description: str = "",
    shapes: Any = None,
    shape_profiles: Any = None,
    hardware_context: str = "",
) -> list[dict[str, str]]:
    """搜索算子参考实现。"""
    exa, web = ExaSearchClient(), WebSearchFallback()
    call_budget = min(max(0, max_calls), BOOTSTRAP_SEARCH_HARD_CALL_LIMIT)
    if call_budget <= 0 or max_results <= 0:
        logger.info("Baseline web search skipped by budget (max_calls=%d, max_results=%d)", max_calls, max_results)
        return []

    queries = _baseline_reference_queries(
        operator,
        dtype,
        task_description=task_description,
        shapes=shapes,
        shape_profiles=shape_profiles,
        hardware_context=hardware_context,
    )[:call_budget]
    all_results: list[dict[str, str]] = []
    calls = 0
    phase_targets = {min(6, call_budget), min(12, call_budget), call_budget}
    per_query_results = max(1, per_query_results)

    for q, domains in queries:
        calls += 1
        logger.info("Baseline web search query %d/%d: %s", calls, call_budget, q)
        r = (
            await exa.search(q, num_results=per_query_results, include_domains=domains)
            if exa.api_key else await web.search(q, per_query_results)
        )
        all_results.extend(r)
        deduped = _dedup(all_results)
        if calls in phase_targets and len(deduped) >= max_results:
            break

    deduped = _dedup(all_results)
    logger.info(
        "Baseline web search issued %d call(s), got %d raw/%d deduped result(s), injecting %d",
        calls, len(all_results), len(deduped), min(len(deduped), max_results),
    )
    return deduped[:max_results]


async def search_on_consecutive_failure(
    operator: str, subspace: str, failure_history: list[str], *, max_results: int = 5,
) -> list[dict[str, str]]:
    """连续失败时深度搜索。"""
    exa, web = ExaSearchClient(), WebSearchFallback()
    queries = [
        f"CUDA {operator} {subspace} common pitfalls solutions",
        f"CUDA {operator} optimization beyond {subspace} alternative approach",
    ]
    ctx = "; ".join(failure_history[-3:]) if failure_history else ""
    if ctx:
        queries.append(f"CUDA kernel {ctx[:80]} fix optimization")

    all_results: list[dict[str, str]] = []
    domains = NVIDIA_DOCS_DOMAINS + CUDA_OSS_DOMAINS + CUDA_BLOG_DOMAINS
    for q in queries:
        r = (await exa.search(q, num_results=2, include_domains=domains)
             if exa.api_key else await web.search(q, 2))
        all_results.extend(r)
    return _dedup(all_results)[:max_results]


def format_search_results_for_prompt(results: list[dict[str, str]]) -> str:
    """格式化搜索结果为 prompt 注入文本。"""
    if not results:
        return ""
    lines = ["### 外部知识参考（来自 Web 搜索）\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"**[{i}] {r.get('title', 'Untitled')}**")
        if r.get("url"):
            lines.append(f"URL: {r['url']}")
        if r.get("text"):
            lines.append(f"摘要: {r['text'][:800]}")
        lines.append("")
    return "\n".join(lines)


def _dedup(results: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    out = []
    for r in results:
        url = r.get("url", "")
        if url and url not in seen:
            seen.add(url)
            out.append(r)
    return out
