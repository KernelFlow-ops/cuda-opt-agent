"""
核心数据模型 —— 严格遵循技术总纲 §4 定义。
所有结构均可序列化为 JSON。
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# ────────────────────────────────────────────
# 枚举
# ────────────────────────────────────────────
class RunStatus(str, Enum):
    RUNNING = "running"
    PAUSED = "paused"
    DONE = "done"
    FAILED = "failed"


class NodeName(str, Enum):
    INIT = "init"
    BOOTSTRAP = "bootstrap"
    PROFILE_BEST = "profile_best"
    ANALYZE = "analyze"
    DECIDE = "decide"
    HP_SEARCH = "hp_search"
    APPLY_DIRECT = "apply_direct"
    EVALUATE = "evaluate"
    REFLECT = "reflect"
    TERMINATE = "terminate"


# ────────────────────────────────────────────
# 算子与硬件
# ────────────────────────────────────────────
class OperatorSpec(BaseModel):
    """算子规格描述。"""
    name: str = Field(..., description="算子名称, 如 gemm, softmax, conv2d")
    signature: str = Field(..., description="算子签名, 如 C = A @ B, A:[M,K] B:[K,N]")
    dtypes: dict[str, str] = Field(default_factory=dict, description="各张量数据类型")
    shapes: dict[str, list[int]] = Field(default_factory=dict, description="各张量形状")
    constraints: list[str] = Field(default_factory=list, description="用户自由文本约束")


class HardwareSpec(BaseModel):
    """GPU 硬件信息。"""
    gpu_name: str = ""
    compute_capability: str = ""          # "sm_90", "sm_80"
    sm_count: int = 0
    shared_mem_per_block_kb: int = 0
    l2_cache_mb: int = 0
    has_tensor_cores: bool = False
    cuda_version: str = ""
    driver_version: str = ""
    raw_dump: str = ""                    # 完整 deviceQuery 输出

    @property
    def signature(self) -> str:
        """用于知识库匹配的硬件签名。"""
        gpu_short = self.gpu_name.lower().replace(" ", "_").replace("nvidia_", "")
        return f"{self.compute_capability}_{gpu_short}"


# ────────────────────────────────────────────
# Benchmark / Profiling
# ────────────────────────────────────────────
class BenchmarkResult(BaseModel):
    """Benchmark 测速结果。"""
    latency_ms_median: float = 0.0
    latency_ms_p95: float = 0.0
    throughput_gflops: float | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class NcuMetrics(BaseModel):
    """ncu Profiling 解析后的结构化指标。"""
    sm_throughput_pct: float | None = None
    compute_memory_throughput_pct: float | None = None
    l1_throughput_pct: float | None = None
    l2_throughput_pct: float | None = None
    dram_throughput_pct: float | None = None
    shared_mem_per_block_bytes: int | None = None
    registers_per_thread: int | None = None
    occupancy_pct: float | None = None
    stall_reasons: dict[str, float] = Field(default_factory=dict)
    raw_text: str = ""
    extra: dict[str, Any] = Field(default_factory=dict)


# ────────────────────────────────────────────
# 迭代记录
# ────────────────────────────────────────────
class IterationRecord(BaseModel):
    """单次迭代的完整记录。"""
    version_id: str                       # "v0", "v3", "v3_hp_2"
    parent_id: str | None = None          # 基于哪个版本
    method_name: str | None = None
    has_hyperparams: bool = False
    hyperparams: dict[str, Any] | None = None
    code_path: str = ""                   # 相对 run 目录
    compile_ok: bool = False
    correctness_ok: bool = False
    benchmark: BenchmarkResult | None = None
    ncu_report_path: str | None = None
    ncu_metrics: NcuMetrics | None = None
    accepted: bool = False
    reasoning_path: str = ""
    timestamp: str = Field(default_factory=lambda: _now_iso())


# ────────────────────────────────────────────
# 黑名单
# ────────────────────────────────────────────
class BlacklistEntry(BaseModel):
    """已失败的 (方法+超参约束) 记录。"""
    method_name_normalized: str
    hyperparam_constraint: dict[str, Any] | None = None
    failed_at_version: str = ""
    reason: str = ""


# ────────────────────────────────────────────
# LLM 决策
# ────────────────────────────────────────────
class MethodDecision(BaseModel):
    """LLM 在 decide 节点产出的结构化决策。"""
    method_name: str
    has_hyperparams: bool = False
    hyperparams_schema: dict[str, Any] | None = None
    rationale: str = ""
    expected_impact: str = "medium"       # high / medium / low + 说明
    confidence: float = 0.5               # 0~1
    give_up: bool = False                 # LLM 认为没招了


class HyperparamCandidate(BaseModel):
    """一组超参候选。"""
    index: int
    hyperparams: dict[str, Any]
    rationale: str = ""


# ────────────────────────────────────────────
# Agent 配置
# ────────────────────────────────────────────
class AgentConfig(BaseModel):
    """Agent 运行时配置。"""
    llm_provider: str = "anthropic"
    llm_model: str = "claude-sonnet-4-20250514"
    max_iterations: int = 30
    consecutive_reject_limit: int = 5
    accept_epsilon: float = 0.005
    compile_repair_max_retries: int = 3
    hp_candidate_count: int = 5
    benchmark_warmup_rounds: int = 10
    benchmark_measure_rounds: int = 100
    runs_dir: str = "runs"
    knowledge_base_dir: str = "knowledge_base"


# ────────────────────────────────────────────
# 运行状态（续跑核心）
# ────────────────────────────────────────────
class RunState(BaseModel):
    """完整运行状态,用于续跑和检查点。"""
    run_id: str = ""
    operator_spec: OperatorSpec = Field(default_factory=lambda: OperatorSpec(name="", signature=""))
    hardware_spec: HardwareSpec = Field(default_factory=HardwareSpec)
    iterations: list[IterationRecord] = Field(default_factory=list)
    current_best_id: str = ""
    blacklist: list[BlacklistEntry] = Field(default_factory=list)
    status: RunStatus = RunStatus.RUNNING
    created_at: str = Field(default_factory=lambda: _now_iso())
    updated_at: str = Field(default_factory=lambda: _now_iso())
    config: AgentConfig = Field(default_factory=AgentConfig)

    # ── 便捷方法 ──
    def iter_by_id(self, version_id: str) -> IterationRecord | None:
        for it in self.iterations:
            if it.version_id == version_id:
                return it
        return None

    def accepted_iterations(self) -> list[IterationRecord]:
        return [it for it in self.iterations if it.accepted]

    def consecutive_rejects(self) -> int:
        count = 0
        for it in reversed(self.iterations):
            if it.version_id == "v0":
                break
            if not it.accepted:
                count += 1
            else:
                break
        return count

    def next_version_id(self, has_hp: bool = False) -> str:
        idx = len(self.iterations)
        suffix = "_hp" if has_hp else ""
        return f"v{idx}{suffix}"

    def is_method_blacklisted(self, method_name: str, hp_constraint: dict | None = None) -> bool:
        from .enums import normalize_method_name
        norm = normalize_method_name(method_name)
        for entry in self.blacklist:
            if entry.method_name_normalized == norm:
                if entry.hyperparam_constraint is None and hp_constraint is None:
                    return True
                if entry.hyperparam_constraint == hp_constraint:
                    return True
        return False

    def touch(self) -> None:
        self.updated_at = _now_iso()


# ────────────────────────────────────────────
# 知识库
# ────────────────────────────────────────────
class Outcome(BaseModel):
    run_id: str = ""
    version_id: str = ""
    speedup_vs_parent: float = 1.0
    operator_shape_signature: str = ""
    timestamp: str = Field(default_factory=lambda: _now_iso())


class KnowledgeEntry(BaseModel):
    operator_class: str
    hardware_signature: str
    method_name: str
    hyperparams_pattern: dict[str, Any] | None = None
    observed_outcomes: list[Outcome] = Field(default_factory=list)
    aggregate_speedup: float = 1.0
    confidence: float = 0.0
    notes: str = ""
    last_updated: str = Field(default_factory=lambda: _now_iso())


# ────────────────────────────────────────────
# 辅助
# ────────────────────────────────────────────
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
