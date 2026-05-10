"""
CUDA 优化子空间枚举 —— 完整 20 子空间体系。

基于 NVIDIA CUDA C++ Best Practices Guide 组织，分为 4 个层级:
  Layer 1 — 内存层级优化 (8 subspaces)
  Layer 2 — 执行配置优化 (4 subspaces)
  Layer 3 — 指令级优化   (4 subspaces)
  Layer 4 — 算法与架构级优化 (4 subspaces)
"""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any


class OptimizationSubspace(str, Enum):
    """20 个 CUDA 优化子空间。"""

    # Layer 1: 内存层级优化
    MEMORY_COALESCING = "memory-coalescing"
    SHARED_MEM_TILING = "shared-mem-tiling"
    BANK_CONFLICT_RESOLUTION = "bank-conflict-resolution"
    VECTORIZED_MEMORY_ACCESS = "vectorized-memory-access"
    ASYNC_MEMORY_PIPELINE = "async-memory-pipeline"
    L2_CACHE_TUNING = "l2-cache-tuning"
    REGISTER_OPTIMIZATION = "register-optimization"
    TEXTURE_CONSTANT_MEMORY = "texture-constant-memory"

    # Layer 2: 执行配置优化
    OCCUPANCY_TUNING = "occupancy-tuning"
    CTA_REDISTRIBUTION = "cta-redistribution"
    THREAD_COARSENING = "thread-coarsening"
    WARP_SPECIALIZATION = "warp-specialization"

    # Layer 3: 指令级优化
    INSTRUCTION_OPTIMIZATION = "instruction-optimization"
    CONTROL_FLOW_DIVERGENCE = "control-flow-divergence"
    PRECISION_CONVERSION = "precision-conversion"
    WARP_PRIMITIVE = "warp-primitive"

    # Layer 4: 算法与架构级优化
    REDUCTION_RESTRUCTURE = "reduction-restructure"
    ALGORITHM_REPLACEMENT = "algorithm-replacement"
    FUSION = "fusion"
    LAUNCH_OVERHEAD_MITIGATION = "launch-overhead-mitigation"


class SubspaceLayer(str, Enum):
    MEMORY = "memory"
    EXECUTION = "execution"
    INSTRUCTION = "instruction"
    ALGORITHM = "algorithm"


class BottleneckType(str, Enum):
    MEMORY_BOUND = "memory_bound"
    COMPUTE_BOUND = "compute_bound"
    LATENCY_BOUND = "latency_bound"
    BALANCED = "balanced"


class RunStatus(str, Enum):
    RUNNING = "running"
    PAUSED = "paused"
    DONE = "done"
    FAILED = "failed"


# ══════════════════════════════════════════════
# 子空间元数据
# ══════════════════════════════════════════════

SUBSPACE_METADATA: dict[str, dict[str, Any]] = {
    "memory-coalescing": {
        "layer": SubspaceLayer.MEMORY,
        "has_hyperparams": False,
        "priority": "highest",
        "min_sm": 0,
        "description": "全局内存合并访问: AoS→SoA, corner turning, alignment padding",
        "ncu_indicators": [
            "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
            "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
            "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct",
        ],
        "hyperparams": [],
    },
    "shared-mem-tiling": {
        "layer": SubspaceLayer.MEMORY,
        "has_hyperparams": True,
        "priority": "high",
        "min_sm": 0,
        "description": "共享内存分块与重用: tiling, double/multi-stage buffering",
        "ncu_indicators": [
            "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
            "sm__warps_active.avg.pct_of_peak_sustained_active",
        ],
        "hyperparams": [
            {"name": "tile_m", "type": "int", "range": [16, 32, 64, 128, 256]},
            {"name": "tile_n", "type": "int", "range": [16, 32, 64, 128, 256]},
            {"name": "tile_k", "type": "int", "range": [8, 16, 32, 64]},
            {"name": "num_stages", "type": "int", "range": [2, 3, 4, 5, 6, 7, 8]},
            {"name": "smem_swizzle_mode", "type": "enum", "range": ["None", "B32", "B64", "B128"]},
        ],
    },
    "bank-conflict-resolution": {
        "layer": SubspaceLayer.MEMORY,
        "has_hyperparams": True,
        "priority": "medium",
        "min_sm": 0,
        "description": "共享内存 Bank 冲突消除: padding, swizzling",
        "ncu_indicators": [
            "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
            "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
        ],
        "hyperparams": [
            {"name": "smem_padding", "type": "int", "range": [0, 1, 2, 4]},
            {"name": "swizzle_bits", "type": "int", "range": [0, 3, 4, 5]},
        ],
    },
    "vectorized-memory-access": {
        "layer": SubspaceLayer.MEMORY,
        "has_hyperparams": True,
        "priority": "medium-high",
        "min_sm": 0,
        "description": "向量化加载/存储: float4/int4/double2, __ldg, half2 打包",
        "ncu_indicators": [
            "smsp__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active",
        ],
        "hyperparams": [
            {"name": "vector_width", "type": "int", "range": [1, 2, 4, 8]},
            {"name": "use_ldg", "type": "bool", "range": [True, False]},
        ],
    },
    "async-memory-pipeline": {
        "layer": SubspaceLayer.MEMORY,
        "has_hyperparams": True,
        "priority": "high",
        "min_sm": 80,
        "description": "异步内存流水线: cp.async (SM80+), TMA (SM90+), software pipelining",
        "ncu_indicators": [
            "sm__sass_inst_executed_op_global_ld.sum",
            "smsp__warp_issue_stalled_lg_throttle.avg.pct_of_peak_sustained_active",
        ],
        "hyperparams": [
            {"name": "num_pipeline_stages", "type": "int", "range": [2, 3, 4, 5, 6, 7, 8]},
            {"name": "use_tma", "type": "bool", "range": [True, False]},
            {"name": "cp_async_size", "type": "int", "range": [4, 8, 16]},
            {"name": "prefetch_distance", "type": "int", "range": [1, 2, 3]},
        ],
    },
    "l2-cache-tuning": {
        "layer": SubspaceLayer.MEMORY,
        "has_hyperparams": True,
        "priority": "medium",
        "min_sm": 80,
        "description": "L2 缓存窗口调优: persistence policy, hit ratio",
        "ncu_indicators": [
            "lts__t_sectors_op_read.sum",
            "lts__t_sector_hit_rate.pct",
        ],
        "hyperparams": [
            {"name": "persisting_l2_size_mb", "type": "float", "range": [0, 4, 8, 16, 32]},
            {"name": "hit_ratio", "type": "float", "range": [0.0, 0.25, 0.5, 0.75, 1.0]},
            {"name": "access_policy", "type": "enum", "range": ["streaming", "persisting"]},
        ],
    },
    "register-optimization": {
        "layer": SubspaceLayer.MEMORY,
        "has_hyperparams": True,
        "priority": "high",
        "min_sm": 0,
        "description": "寄存器压力管理: __launch_bounds__, -maxrregcount, 变量复用",
        "ncu_indicators": [
            "launch__registers_per_thread",
            "sm__warps_active.avg.pct_of_peak_sustained_active",
        ],
        "hyperparams": [
            {"name": "max_registers_per_thread", "type": "int", "range": [32, 64, 96, 128, 255]},
            {"name": "launch_bounds_max_threads", "type": "int", "range": [64, 128, 256, 512, 1024]},
            {"name": "launch_bounds_min_blocks", "type": "int", "range": [1, 2, 3, 4, 6, 8]},
        ],
    },
    "texture-constant-memory": {
        "layer": SubspaceLayer.MEMORY,
        "has_hyperparams": False,
        "priority": "low-medium",
        "min_sm": 0,
        "description": "纹理/常量内存利用: 空间局部性只读数据, 64KB constant memory broadcast",
        "ncu_indicators": [],
        "hyperparams": [],
    },
    "occupancy-tuning": {
        "layer": SubspaceLayer.EXECUTION,
        "has_hyperparams": True,
        "priority": "high",
        "min_sm": 0,
        "description": "占用率与 Launch 配置调优: block size, dynamic smem, SM 利用率",
        "ncu_indicators": [
            "sm__warps_active.avg.pct_of_peak_sustained_active",
            "launch__occupancy_limit_warps",
            "launch__occupancy_limit_registers",
            "launch__occupancy_limit_shared_mem",
        ],
        "hyperparams": [
            {"name": "block_dim_x", "type": "int", "range": [32, 64, 128, 256, 512, 1024]},
            {"name": "block_dim_y", "type": "int", "range": [1, 2, 4, 8, 16, 32]},
            {"name": "block_dim_z", "type": "int", "range": [1, 2, 4]},
            {"name": "dynamic_smem_bytes", "type": "int", "range": [0, 4096, 8192, 16384, 32768, 49152]},
        ],
    },
    "cta-redistribution": {
        "layer": SubspaceLayer.EXECUTION,
        "has_hyperparams": True,
        "priority": "medium",
        "min_sm": 0,
        "description": "CTA-数据映射重组: persistent CTA, thread block cluster, split-K, tile scheduling",
        "ncu_indicators": [
            "sm__warps_active.avg.pct_of_peak_sustained_active",
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
        ],
        "hyperparams": [
            {"name": "persistent_cta_count", "type": "int", "range": "SM_count_multiples"},
            {"name": "cluster_shape_m", "type": "int", "range": [1, 2, 4]},
            {"name": "cluster_shape_n", "type": "int", "range": [1, 2, 4]},
            {"name": "split_k_slices", "type": "int", "range": [1, 2, 4, 8, 16]},
            {"name": "tile_schedule", "type": "enum", "range": ["row", "col", "swizzle", "stream_k"]},
        ],
    },
    "thread-coarsening": {
        "layer": SubspaceLayer.EXECUTION,
        "has_hyperparams": True,
        "priority": "medium",
        "min_sm": 0,
        "description": "线程粗化: 每线程处理多元素, grid-stride loop, 提高 ILP",
        "ncu_indicators": [
            "smsp__inst_executed.avg.pct_of_peak_sustained_active",
        ],
        "hyperparams": [
            {"name": "elements_per_thread", "type": "int", "range": [1, 2, 4, 8, 16]},
            {"name": "use_grid_stride", "type": "bool", "range": [True, False]},
        ],
    },
    "warp-specialization": {
        "layer": SubspaceLayer.EXECUTION,
        "has_hyperparams": True,
        "priority": "high",
        "min_sm": 90,
        "description": "Warp 特化: producer/consumer warp 分工, ping-pong 深流水线重叠 (Hopper+)",
        "ncu_indicators": [
            "sm__warps_active.avg.pct_of_peak_sustained_active",
        ],
        "hyperparams": [
            {"name": "num_producer_warps", "type": "int", "range": [1, 2]},
            {"name": "num_consumer_warp_groups", "type": "int", "range": [1, 2]},
            {"name": "pipeline_mode", "type": "enum", "range": ["cooperative", "pingpong"]},
        ],
    },
    "instruction-optimization": {
        "layer": SubspaceLayer.INSTRUCTION,
        "has_hyperparams": False,
        "priority": "medium",
        "min_sm": 0,
        "description": "算术指令级优化: fast math intrinsics, FMA, rsqrt, pragma unroll",
        "ncu_indicators": [
            "smsp__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active",
        ],
        "compiler_flags": [
            {"name": "--use_fast_math", "type": "bool"},
            {"name": "--fmad", "type": "bool"},
            {"name": "--prec-div", "type": "bool"},
            {"name": "--prec-sqrt", "type": "bool"},
        ],
        "hyperparams": [],
    },
    "control-flow-divergence": {
        "layer": SubspaceLayer.INSTRUCTION,
        "has_hyperparams": False,
        "priority": "high",
        "min_sm": 0,
        "description": "控制流分歧消除: warp divergence, branch predication, 循环统一化",
        "ncu_indicators": [
            "smsp__thread_inst_executed_pred_on.avg.pct_of_peak_sustained_active",
        ],
        "hyperparams": [],
    },
    "precision-conversion": {
        "layer": SubspaceLayer.INSTRUCTION,
        "has_hyperparams": True,
        "priority": "medium-high",
        "min_sm": 0,
        "description": "精度与数值格式优化: 混合精度, TF32, FP8, half2 打包运算",
        "ncu_indicators": [
            "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active",
        ],
        "hyperparams": [
            {"name": "compute_dtype", "type": "enum",
             "range": ["fp16", "bf16", "tf32", "fp32", "fp8_e4m3", "fp8_e5m2"]},
            {"name": "accumulator_dtype", "type": "enum", "range": ["fp16", "fp32"]},
            {"name": "output_dtype", "type": "enum", "range": ["fp16", "bf16", "fp32"]},
            {"name": "use_half2_pack", "type": "bool", "range": [True, False]},
        ],
    },
    "warp-primitive": {
        "layer": SubspaceLayer.INSTRUCTION,
        "has_hyperparams": False,
        "priority": "medium-high",
        "min_sm": 0,
        "description": "Warp 级原语替代: shuffle, vote, cooperative groups, 零共享内存归约",
        "ncu_indicators": [
            "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
        ],
        "hyperparams": [],
    },
    "reduction-restructure": {
        "layer": SubspaceLayer.ALGORITHM,
        "has_hyperparams": True,
        "priority": "medium",
        "min_sm": 0,
        "description": "归约拓扑重组: tree/warp-shuffle/multi-CTA, Welford, segmented reduction",
        "ncu_indicators": [
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
        ],
        "hyperparams": [
            {"name": "reduction_algo", "type": "enum",
             "range": ["sequential", "tree", "warp_shuffle", "cooperative"]},
            {"name": "warps_per_reduction", "type": "int", "range": [1, 2, 4, 8, 16, 32]},
            {"name": "use_atomic_reduction", "type": "bool", "range": [True, False]},
        ],
    },
    "algorithm-replacement": {
        "layer": SubspaceLayer.ALGORITHM,
        "has_hyperparams": False,
        "priority": "high",
        "min_sm": 0,
        "description": "算法级替换: cuBLAS/cuDNN/CUTLASS, Online Softmax, FlashAttention, Tensor Core 路径",
        "ncu_indicators": [],
        "hyperparams": [],
    },
    "fusion": {
        "layer": SubspaceLayer.ALGORITHM,
        "has_hyperparams": False,
        "priority": "high",
        "min_sm": 0,
        "description": "Kernel 合并: vertical/horizontal/epilogue fusion, 消除中间全局内存读写",
        "ncu_indicators": [],
        "hyperparams": [],
    },
    "launch-overhead-mitigation": {
        "layer": SubspaceLayer.ALGORITHM,
        "has_hyperparams": True,
        "priority": "high",
        "min_sm": 0,
        "description": "Launch 开销消减: CUDA Graphs, persistent kernel, cudaMemPool",
        "ncu_indicators": [],
        "hyperparams": [
            {"name": "use_cuda_graph", "type": "bool", "range": [True, False]},
            {"name": "persistent_grid_size", "type": "int", "range": "SM_count_multiples"},
        ],
    },
}

SUBSPACE_SYNERGIES: list[tuple[str, str, str]] = [
    ("memory-coalescing", "vectorized-memory-access", "合并访问后用向量加载进一步减少指令数"),
    ("shared-mem-tiling", "bank-conflict-resolution", "tiling 引入共享内存后必须检查 bank conflict"),
    ("shared-mem-tiling", "async-memory-pipeline", "异步拷贝是 tiling 的自然升级路径"),
    ("occupancy-tuning", "register-optimization", "二者通过寄存器用量耦合"),
    ("thread-coarsening", "vectorized-memory-access", "粗化后每线程可用向量加载处理更多数据"),
    ("warp-specialization", "async-memory-pipeline", "warp 特化的基础是异步内存流水线"),
]

SUBSPACE_CONFLICTS: list[tuple[str, str, str]] = [
    ("occupancy-tuning", "register-optimization", "高占用率 vs 大量寄存器降低占用率"),
    ("thread-coarsening", "occupancy-tuning", "减少线程数 vs 增加活跃线程数"),
    ("warp-primitive", "shared-mem-tiling", "消除共享内存 vs 增加共享内存使用"),
    ("cta-redistribution", "launch-overhead-mitigation", "split-k 增加 kernel 数量/atomic"),
]

ARCH_GATES: dict[str, int] = {
    "async-memory-pipeline": 80,
    "warp-specialization": 90,
    "l2-cache-tuning": 80,
}

PRECISION_ARCH_GATES: dict[str, int] = {
    "fp8_e4m3": 89, "fp8_e5m2": 89, "tf32": 80,
}


def get_subspace_meta(name: str) -> dict[str, Any]:
    return SUBSPACE_METADATA.get(name, {})

def has_hyperparams(name: str) -> bool:
    return get_subspace_meta(name).get("has_hyperparams", False)

def get_layer(name: str) -> str:
    meta = get_subspace_meta(name)
    layer = meta.get("layer")
    return layer.value if layer else "unknown"

def filter_by_arch(sm_version: int) -> list[str]:
    return [n for n, m in SUBSPACE_METADATA.items() if m.get("min_sm", 0) <= sm_version]

def get_synergies_for(subspace: str) -> list[tuple[str, str]]:
    r = []
    for a, b, reason in SUBSPACE_SYNERGIES:
        if a == subspace:
            r.append((b, reason))
        elif b == subspace:
            r.append((a, reason))
    return r

def get_conflicts_for(subspace: str) -> list[tuple[str, str]]:
    r = []
    for a, b, reason in SUBSPACE_CONFLICTS:
        if a == subspace:
            r.append((b, reason))
        elif b == subspace:
            r.append((a, reason))
    return r

def all_subspace_names() -> list[str]:
    return [s.value for s in OptimizationSubspace]


OPTIMIZATION_SUBSPACES = all_subspace_names()

SUBSPACE_KEYWORDS: dict[str, list[str]] = {
    "memory-coalescing": ["coalesc", "global_memory", "aos", "soa", "alignment"],
    "shared-mem-tiling": ["shared", "smem", "tile", "tiling", "blocking", "double_buffer"],
    "bank-conflict-resolution": ["bank", "conflict", "padding", "swizzle"],
    "vectorized-memory-access": ["vector", "float4", "float2", "half2", "int4", "ldg"],
    "async-memory-pipeline": ["cp_async", "async", "pipeline", "tma", "multistage"],
    "l2-cache-tuning": ["l2", "cache", "persist", "access_policy"],
    "register-optimization": ["register", "launch_bounds", "maxrregcount", "reg"],
    "texture-constant-memory": ["texture", "constant", "readonly"],
    "occupancy-tuning": ["occupancy", "block_size", "threads_per_block"],
    "cta-redistribution": ["cta", "block_mapping", "split_k", "persistent", "cluster"],
    "thread-coarsening": ["coarsen", "elements_per_thread", "grid_stride", "ilp"],
    "warp-specialization": ["warp_special", "producer", "consumer", "specialization"],
    "instruction-optimization": ["fma", "fast_math", "intrinsic", "unroll", "instruction"],
    "control-flow-divergence": ["divergence", "branch", "predicate", "control_flow"],
    "precision-conversion": ["precision", "mixed", "tf32", "fp8", "bf16", "half2"],
    "warp-primitive": ["shuffle", "shfl", "warp", "ballot", "cooperative_groups"],
    "reduction-restructure": ["reduction", "reduce", "tree", "partial_sum", "welford"],
    "algorithm-replacement": ["algorithm", "cutlass", "cublas", "cudnn", "library"],
    "fusion": ["fusion", "fuse", "fused", "epilogue", "merge"],
    "launch-overhead-mitigation": ["launch", "cuda_graph", "persistent_kernel", "graph_capture"],
}


def normalize_method_name(name: str) -> str:
    """Normalize free-form method names for blacklist keys."""
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def make_blacklist_key(method_name: str, hp_constraint: dict | None = None) -> str:
    norm = normalize_method_name(method_name)
    if hp_constraint:
        hp_text = json.dumps(hp_constraint, sort_keys=True, separators=(",", ":"))
        return f"{norm}::{hp_text}"
    return norm


def infer_subspace_from_method_name(method_name: str) -> str | None:
    norm = normalize_method_name(method_name)
    best_subspace: str | None = None
    best_score = 0
    for subspace, keywords in SUBSPACE_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            key = normalize_method_name(keyword)
            if key and key in norm:
                score += len(key)
        if score > best_score:
            best_score = score
            best_subspace = subspace
    return best_subspace


def subspaces_overlap(subspace_a: str | None, subspace_b: str | None) -> bool:
    if not subspace_a or not subspace_b:
        return False
    if subspace_a == subspace_b:
        return True
    aliases = {
        "vectorization": "vectorized-memory-access",
        "register-blocking": "register-optimization",
    }
    a = aliases.get(subspace_a, subspace_a)
    b = aliases.get(subspace_b, subspace_b)
    if a == b:
        return True
    overlap_pairs = {
        frozenset({"reduction-restructure", "cta-redistribution"}),
        frozenset({"reduction-restructure", "warp-primitive"}),
        frozenset({"shared-mem-tiling", "bank-conflict-resolution"}),
        frozenset({"shared-mem-tiling", "async-memory-pipeline"}),
        frozenset({"memory-coalescing", "vectorized-memory-access"}),
    }
    return frozenset({a, b}) in overlap_pairs
