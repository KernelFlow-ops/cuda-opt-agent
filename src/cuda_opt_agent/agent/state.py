"""
LangGraph 状态定义 —— 节点间通过 GraphState 通信。
"""

from __future__ import annotations

from typing import Any, TypedDict

from ..models.data import (
    BenchmarkResult,
    HardwareSpec,
    IterationRecord,
    MethodDecision,
    NcuMetrics,
    OperatorSpec,
    RunState,
)


class GraphState(TypedDict, total=False):
    """
    LangGraph 状态机的共享状态。
    所有节点通过此 dict 通信。
    """
    # ── 输入 ──
    operator_spec: OperatorSpec
    hardware_spec: HardwareSpec
    run_state: RunState

    # ── 流转数据 ──
    current_code: str                     # 当前 best 的代码
    current_ncu: NcuMetrics               # 当前 best 的 ncu 报告
    current_benchmark: BenchmarkResult    # 当前 best 的 benchmark

    analysis_result: dict                 # analyze 节点产出
    method_decision: MethodDecision       # decide 节点产出
    hp_candidates: list[dict]             # hp_search 节点产出的候选
    new_code: str                         # apply 节点产出的新代码
    new_version_id: str                   # 新版本 ID

    # ── 评估结果 ──
    trial_version_id: str
    trial_benchmark: BenchmarkResult
    trial_ncu: NcuMetrics
    trial_accepted: bool
    trial_compile_ok: bool
    trial_correctness_ok: bool

    # ── 反思 ──
    reflection: dict

    # ── 控制流 ──
    should_stop: bool
    stop_reason: str
    error: str | None

    # ── 内部路由 ──
    has_hyperparams: bool
    iteration_count: int
