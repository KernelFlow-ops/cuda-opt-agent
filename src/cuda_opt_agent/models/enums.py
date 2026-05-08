"""
辅助枚举与归一化工具。

[改进] 新增:
  - OPTIMIZATION_SUBSPACES: 优化子空间常量
  - SUBSPACE_KEYWORDS: 子空间关键词映射，用于模糊匹配
  - infer_subspace_from_method_name: 从方法名推断子空间
"""

from __future__ import annotations

import re


# ────────────────────────────────────────────
# 优化子空间定义
# ────────────────────────────────────────────
OPTIMIZATION_SUBSPACES = [
    "reduction-restructure",
    "cta-redistribution",
    "fusion",
    "warp-primitive",
    "vectorization",
    "shared-mem-tiling",
    "register-blocking",
    "algorithm-replacement",
    "launch-overhead-mitigation",
    "precision-conversion",
]

# 用于从 method_name 自动推断 subspace 的关键词映射
SUBSPACE_KEYWORDS: dict[str, list[str]] = {
    "reduction-restructure": [
        "reduction", "reduce", "归约", "cta_per_channel", "blocks_per_channel",
        "partial_sum", "partial_reduction", "multi_cta", "split_reduce",
        "cross_warp", "warp_per_channel", "channel_per_warp",
    ],
    "cta-redistribution": [
        "cta_redistrib", "block_mapping", "channels_per_block",
        "grid_parallel", "cta_packing", "打包", "redistrib",
    ],
    "fusion": [
        "fusion", "fuse", "fused", "merge", "融合", "single_kernel",
        "combine", "inline",
    ],
    "warp-primitive": [
        "warp_shuffle", "shfl", "shuffle", "warp_level", "warp_vote",
        "warp_match", "ballot", "warp_sync",
    ],
    "vectorization": [
        "vector", "half2", "float4", "float2", "向量化", "ldg",
        "vectorized_load", "vectorized_store", "coalesced",
    ],
    "shared-mem-tiling": [
        "shared_mem", "tiling", "tile", "double_buffer", "async_copy",
        "smem", "共享内存",
    ],
    "register-blocking": [
        "register_block", "register_tile", "ilp", "unroll",
        "寄存器", "reg_block",
    ],
    "algorithm-replacement": [
        "algorithm", "cudnn", "cublas", "cutlass", "library",
        "welford", "two_pass", "online", "formula",
    ],
    "launch-overhead-mitigation": [
        "launch", "cuda_graph", "persistent", "kernel_count",
        "graph_capture",
    ],
    "precision-conversion": [
        "precision", "mixed", "tf32", "fp8", "bf16",
        "quantiz", "精度",
    ],
}


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


def infer_subspace_from_method_name(method_name: str) -> str | None:
    """
    [改进] 从 method_name 推断可能的优化子空间。

    使用关键词匹配,返回匹配度最高的子空间。
    如果无法推断,返回 None。
    """
    norm = normalize_method_name(method_name)

    best_subspace: str | None = None
    best_score = 0

    for subspace, keywords in SUBSPACE_KEYWORDS.items():
        score = 0
        for kw in keywords:
            kw_norm = kw.lower().replace("-", "_").replace(" ", "_")
            if kw_norm in norm:
                score += len(kw_norm)  # 更长的匹配权重更高
        if score > best_score:
            best_score = score
            best_subspace = subspace

    return best_subspace


def subspaces_overlap(subspace_a: str | None, subspace_b: str | None) -> bool:
    """
    [改进] 判断两个子空间是否语义重叠。

    直接相同 → 重叠。
    某些子空间对也视为重叠（例如 reduction-restructure 和 cta-redistribution
    在小 kernel 上往往是同一件事）。
    """
    if not subspace_a or not subspace_b:
        return False
    if subspace_a == subspace_b:
        return True

    # 定义语义相近的子空间对
    overlap_pairs = {
        frozenset({"reduction-restructure", "cta-redistribution"}),
        frozenset({"reduction-restructure", "warp-primitive"}),
    }
    return frozenset({subspace_a, subspace_b}) in overlap_pairs
