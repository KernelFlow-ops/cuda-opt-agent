"""
核心数据模型 —— 严格遵循技术总纲 §4 定义。
所有结构均可序列化为 JSON。

[优化] 新增字段:
  - AgentConfig.gpu_ids: 多 GPU 候选分发
  - AgentConfig.correctness_max_parallel: 跨 shape 并行校验
  - AgentConfig.nvcc_parallel_threads: nvcc -t N
  - AgentConfig.hp_llm_concurrency: hp_search LLM 并发数
  - AgentConfig.use_code_diff: 代码 diff 模式

[改进] 新增字段:
  - BlacklistEntry.subspace / pattern_signature: 子空间级黑名单
  - MethodDecision.subspace / differentiation_from_failed / falsification_condition / meta_decision / give_up_reason_type
  - RunState.kernel_regime: 首次分析后缓存的 regime 信息
  - AgentConfig.launch_floor_ms / catastrophic_regression_threshold
  - HyperparamCandidate.predicted_regression_risk / risk_rationale
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


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
    COMPARE_LIBRARY = "compare_library"  # [改进] 新增
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
    shapes: dict[str, Any] = Field(default_factory=dict, description="各张量形状或维度参数")
    constraints: list[str] = Field(default_factory=list, description="用户自由文本约束")
    task_description: str = Field(default="", description="自由文本任务说明")
    seed_code_path: str | None = Field(default=None, description="已有 .cu 文件路径,作为 v0 起点")
    shape_profiles: list[dict[str, Any]] = Field(
        default_factory=list,
        description="多尺度 shape profiles; shapes 为空时默认使用第一个 profile",
    )

    @model_validator(mode="after")
    def fill_shapes_from_first_profile(self) -> "OperatorSpec":
        """Keep legacy single-shape flows working for multi-profile specs."""
        if not self.shapes and self.shape_profiles:
            self.shapes = dict(self.shape_profiles[0])
        return self


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


class GpuDevice(BaseModel):
    """[优化] 单块 GPU 设备信息, 用于多 GPU 分发。"""
    index: int = 0
    gpu_name: str = ""
    compute_capability: str = ""
    memory_total_mb: int = 0


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
    """已失败的 (方法+超参约束) 记录。

    [改进] 新增 subspace 和 pattern_signature 字段，
    支持子空间级黑名单匹配，防止 LLM 换名绕过。
    """
    method_name_normalized: str
    hyperparam_constraint: dict[str, Any] | None = None
    failed_at_version: str = ""
    reason: str = ""
    # [改进] 子空间级黑名单
    subspace: str | None = None
    pattern_signature: str | None = None
    regression_severity: str | None = None  # catastrophic / severe / mild / correctness
    trigger_conditions: dict[str, Any] | None = None


# ────────────────────────────────────────────
# LLM 决策
# ────────────────────────────────────────────
class MetaDecision(BaseModel):
    """[改进] 元决策: 先决定是否继续优化。"""
    should_continue: bool = True
    reasons: list[str] = Field(default_factory=list)
    regression_streak: int = 0
    exhausted_subspaces: list[str] = Field(default_factory=list)
    remaining_promising_subspaces: list[str] = Field(default_factory=list)
    near_library_baseline: bool = False


class MethodDecision(BaseModel):
    """LLM 在 decide 节点产出的结构化决策。

    [改进] 新增:
      - meta_decision: 元决策信息
      - subspace: 所属优化子空间
      - differentiation_from_failed: 与已失败方法的差异论述
      - falsification_condition: 否证条件
      - give_up_reason_type: 放弃原因分类
    """
    method_name: str
    has_hyperparams: bool = False
    hyperparams_schema: dict[str, Any] | None = None
    rationale: str = ""
    expected_impact: str = "medium"       # high / medium / low + 说明
    confidence: float = 0.5               # 0~1
    give_up: bool = False                 # LLM 认为没招了
    # [改进] 新增字段
    meta_decision: MetaDecision | None = None
    subspace: str | None = None
    differentiation_from_failed: str = ""
    falsification_condition: str = ""
    give_up_reason_type: str | None = None  # optimal_reached / exhausted_search / catastrophic_streak / near_library_baseline


class HyperparamCandidate(BaseModel):
    """一组超参候选。

    [改进] 新增 predicted_regression_risk 和 risk_rationale。
    """
    index: int
    hyperparams: dict[str, Any]
    rationale: str = ""
    predicted_regression_risk: str = "medium"  # low / medium / high
    risk_rationale: str = ""


# ────────────────────────────────────────────
# Agent 配置
# ────────────────────────────────────────────
class AgentConfig(BaseModel):
    """Agent 运行时配置。"""
    llm_provider: str = "anthropic"
    llm_model: str = "claude-sonnet-4-20250514"
    default_dtype: str = "fp16"
    max_iterations: int = 30
    consecutive_reject_limit: int = 5
    accept_epsilon: float = 0.005
    compile_repair_max_retries: int = 3
    decide_reselect_max_retries: int = 3
    hp_candidate_count: int = 5
    hp_compile_workers: int = 0
    benchmark_warmup_rounds: int = 10
    benchmark_measure_rounds: int = 100
    ncu_launch_count: int = 3
    ncu_warmup_rounds: int = 1
    ncu_profile_rounds: int = 1
    multi_shape_aggregator: Literal["mean", "worst", "weighted"] = "mean"
    runs_dir: str = "runs"
    knowledge_base_dir: str = "knowledge_base"

    # ── [优化] 新增字段 ──
    gpu_ids: list[int] = Field(
        default_factory=list,
        description="可用 GPU 索引列表, 空=自动检测; 用于 hp 候选多 GPU 分发",
    )
    correctness_max_parallel: int = Field(
        default=2,
        description="跨 shape 并行正确性校验的最大并发数",
    )
    nvcc_parallel_threads: int = Field(
        default=0,
        description="nvcc -t N 编译并行线程数; 0=auto(cpu_count), 1=禁用",
    )
    hp_llm_concurrency: int = Field(
        default=3,
        description="hp_search 中 LLM 代码生成的最大并发数",
    )
    use_code_diff: bool = Field(
        default=True,
        description="为 apply/analyze 节点使用代码 diff 而非完整代码注入",
    )
    use_tool_use: bool = Field(
        default=True,
        description="对 JSON 输出节点使用 Tool Use (function calling) 替代自由 JSON",
    )

    # ── [修复] HP correctness 修复机制 ──
    hp_correctness_repair_max: int = Field(
        default=2,
        description="HP 候选 correctness 失败时的最大修复尝试次数 (0=不修复, 直接丢弃)",
    )

    # ── [改进] 新增配置 ──
    launch_floor_ms: float = Field(
        default=0.005,
        description="估计的 kernel launch overhead floor (ms), 用于判断 tiny kernel regime",
    )
    catastrophic_regression_threshold: float = Field(
        default=3.0,
        description="回归倍数 >= 该值视为 catastrophic regression",
    )
    catastrophic_streak_limit: int = Field(
        default=2,
        description="连续 catastrophic regression 的提示阈值; 强制跑满模式下不触发早停",
    )
    tiny_kernel_reject_limit: int = Field(
        default=3,
        description="tiny kernel 下连续 reject 的提示阈值; 强制跑满模式下不触发早停",
    )
    enable_library_comparison: bool = Field(
        default=True,
        description="是否在 bootstrap 后执行 cuDNN/cuBLAS 基线对比",
    )
    enable_web_search_baseline: bool = Field(
        default=True,
        description="是否在 bootstrap 阶段搜索外部参考实现",
    )
    bootstrap_web_search_max_calls: int = Field(
        default=20,
        description="bootstrap 阶段外部搜索最大调用次数; 运行时硬上限为 20",
    )
    bootstrap_web_search_max_results: int = Field(
        default=12,
        description="bootstrap 阶段最多注入 prompt 的去重搜索结果数",
    )
    bootstrap_web_search_per_query_results: int = Field(
        default=3,
        description="bootstrap 阶段每个搜索 query 请求的结果数",
    )
    web_search_on_failure_threshold: int = Field(
        default=2,
        description="连续 reject 达到该次数后触发外部知识搜索",
    )


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
    # [改进] 新增
    kernel_regime: dict[str, Any] = Field(
        default_factory=dict,
        description="首次 analyze 后缓存的 regime 信息 (regime, near_launch_floor 等)",
    )
    library_baseline_ms: float | None = Field(
        default=None,
        description="cuDNN/cuBLAS 等价实现的 latency (ms), 若有",
    )

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

    def consecutive_correctness_failures(self) -> int:
        """[修复] 统计尾部连续的 correctness 失败次数 (不区分是否 accepted)。"""
        count = 0
        for it in reversed(self.iterations):
            if it.version_id == "v0":
                break
            if not it.correctness_ok:
                count += 1
            else:
                break
        return count

    def best_latency_ms(self) -> float | None:
        """[改进] 获取当前 best 的 latency。"""
        best = self.iter_by_id(self.current_best_id)
        if best and best.benchmark:
            return best.benchmark.latency_ms_median
        return None

    def catastrophic_regression_streak(self, threshold: float = 3.0) -> int:
        """[改进] 统计尾部连续 catastrophic regression 次数。"""
        best_lat = self.best_latency_ms()
        if not best_lat or best_lat <= 0:
            return 0
        count = 0
        for it in reversed(self.iterations):
            if it.version_id == self.current_best_id or it.version_id == "v0":
                break
            if it.benchmark and it.benchmark.latency_ms_median > 0:
                ratio = it.benchmark.latency_ms_median / best_lat
                if ratio >= threshold:
                    count += 1
                else:
                    break
            elif not it.compile_ok or not it.correctness_ok:
                count += 1  # 编译/正确性失败也算入 streak
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

    def is_subspace_blacklisted(self, subspace: str | None) -> tuple[bool, str]:
        """[改进] 检查某个子空间是否已有 ≥ 2 次失败（视为穷尽）。"""
        if not subspace:
            return False, ""
        from .enums import subspaces_overlap
        failure_count = 0
        reasons = []
        for entry in self.blacklist:
            if subspaces_overlap(entry.subspace, subspace):
                failure_count += 1
                reasons.append(f"{entry.method_name_normalized}({entry.subspace})")
        if failure_count >= 2:
            return True, f"subspace '{subspace}' has {failure_count} failures: {', '.join(reasons)}"
        return False, ""

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
    # [改进] 新增
    polarity: str = "positive"  # positive / negative


# ────────────────────────────────────────────
# 辅助
# ────────────────────────────────────────────
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
