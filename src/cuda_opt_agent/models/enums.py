"""
辅助枚举与归一化工具。
"""

from __future__ import annotations

import re


def normalize_method_name(name: str) -> str:
    """
    将 LLM 输出的自由字符串方法名归一化,
    用作黑名单 key。
    """
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = s.strip("_")
    return s


def make_blacklist_key(method_name: str, hp_constraint: dict | None = None) -> str:
    """
    生成黑名单唯一键: method_norm 或 method_norm::hp_json
    """
    norm = normalize_method_name(method_name)
    if hp_constraint:
        import orjson
        hp_str = orjson.dumps(hp_constraint, option=orjson.OPT_SORT_KEYS).decode()
        return f"{norm}::{hp_str}"
    return norm


