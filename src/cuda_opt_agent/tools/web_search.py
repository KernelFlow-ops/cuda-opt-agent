"""
(可选) Web 检索工具 —— 当 LLM 需要查阅 CUDA 文档时使用。
可接入 Tavily / SerpAPI / 或自定义接口。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    title: str = ""
    url: str = ""
    snippet: str = ""


def web_search(query: str, max_results: int = 5) -> list[SearchResult]:
    """
    执行 Web 搜索。当前为占位实现。

    可扩展接入:
    - Tavily API
    - SerpAPI
    - Google Custom Search
    """
    logger.info("Web search placeholder: %s", query)

    # TODO: 接入真实搜索 API
    # api_key = os.getenv("TAVILY_API_KEY")
    # if api_key:
    #     from tavily import TavilyClient
    #     client = TavilyClient(api_key=api_key)
    #     results = client.search(query, max_results=max_results)
    #     return [SearchResult(title=r["title"], url=r["url"], snippet=r["content"]) for r in results["results"]]

    return []