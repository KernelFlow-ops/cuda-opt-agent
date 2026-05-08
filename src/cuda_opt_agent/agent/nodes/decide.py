"""
Decide 节点 —— LLM 决策下一个优化方法。

[优化]:
  - 支持 use_tool_use: 使用 Tool Use (function calling) 替代自由 JSON

[改进]:
  - 子空间级黑名单匹配,防止 LLM "换皮"绕过
  - 动态温度: 连续回归时升温,鼓励换思路
  - 传递 kernel_regime / library_baseline / regression_streak 信息
  - 从 LLM 输出中提取 subspace 并验证
"""

from __future__ import annotations

import json
import logging

from ...models.data import BenchmarkResult, MethodDecision
from ...models.enums import (
    infer_subspace_from_method_name,
    normalize_method_name,
    subspaces_overlap,
)
from ..temperatures import get_dynamic_decide_temperature

logger = logging.getLogger(__name__)


async def decide_node(self, state: dict) -> dict:
    """LLM 决策下一个优化方法。"""
    logger.info("=== DECIDE ===")
    run_state = state["run_state"]
    op = state["operator_spec"]
    hw = state["hardware_spec"]

    analysis = state.get("analysis_result", {})
    bm = state.get("current_benchmark", BenchmarkResult())

    # ── [改进] 构建增强的黑名单文本（含子空间信息）──
    bl_lines = []
    for entry in run_state.blacklist:
        hp_str = json.dumps(entry.hyperparam_constraint) if entry.hyperparam_constraint else "no hyperparams"
        subspace_str = f", subspace={entry.subspace}" if entry.subspace else ""
        pattern_str = f", pattern={entry.pattern_signature}" if entry.pattern_signature else ""
        severity_str = f", severity={entry.regression_severity}" if entry.regression_severity else ""
        bl_lines.append(
            f"  - {entry.method_name_normalized} ({hp_str}{subspace_str}{pattern_str}{severity_str}): {entry.reason}"
        )
    blacklist_text = "\n".join(bl_lines) or "(none)"

    kb_hints = self.kb.query(op.name, hw.signature)
    hints_text = self.kb.format_hints_for_prompt(kb_hints)

    # ── [改进] 增强的 benchmark 文本（含 regime 和 regression streak）──
    abs_lat = bm.latency_ms_median
    launch_floor = self.sm.config.launch_floor_ms
    regime = (
        "tiny (likely launch-overhead-bound)" if abs_lat < 0.01
        else "small" if abs_lat < 0.1
        else "medium" if abs_lat < 1.0
        else "large"
    )
    distance_from_floor = abs_lat / launch_floor if launch_floor > 0 else 999.0

    cat_streak = run_state.catastrophic_regression_streak(
        self.sm.config.catastrophic_regression_threshold
    )

    bm_text = (
        f"latency_median: {abs_lat:.4f} ms\n"
        f"latency_p95: {bm.latency_ms_p95:.4f} ms\n"
        f"regime: {regime}\n"
        f"distance_from_launch_floor: {distance_from_floor:.1f}x  "
        f"(若 < 2x, 大部分时间是 launch overhead, ncu % 指标不可信)\n"
        f"catastrophic_regression_streak: {cat_streak} "
        f"(连续 >= {self.sm.config.catastrophic_regression_threshold}x 回归的次数)\n"
        f"aggregator: {bm.extra.get('aggregator', 'single')}\n"
        f"per_shape: {self._per_shape_summary(bm).replace('<br>', '; ') or 'N/A'}"
    )

    # ── [改进] Library baseline 文本 ──
    lib_lat = run_state.library_baseline_ms
    if lib_lat is not None and lib_lat > 0:
        ratio_to_lib = abs_lat / lib_lat
        library_text = (
            f"cuDNN/cuBLAS 等价实现 latency: {lib_lat:.4f} ms\n"
            f"当前 best / library = {ratio_to_lib:.2f}x  "
            f"({'已接近库基线, 继续优化收益有限' if ratio_to_lib < 1.2 else '仍有空间'})"
        )
    else:
        library_text = "(未测量或不适用)"

    rejected_methods: list[tuple[str, str]] = []
    forced_continue_notes: list[str] = []
    current_iteration = len(run_state.iterations)
    max_iterations = self.sm.config.max_iterations
    if current_iteration >= max_iterations:
        return {
            "should_stop": True,
            "stop_reason": f"Reached maximum iterations ({max_iterations})",
        }

    def rejected_methods_text() -> str:
        if not rejected_methods:
            return "(none)"
        return "\n".join(
            f"  - {name} (normalized: {normalized})"
            for name, normalized in rejected_methods
        )

    def forced_continue_text() -> str:
        lines = [
            f"当前迭代记录数: {current_iteration}",
            f"最大迭代数: {max_iterations}",
            "强制跑满策略: 只要当前迭代记录数 < 最大迭代数, 就必须继续提出候选方法, 不允许 give_up。",
        ]
        if forced_continue_notes:
            lines.append("本轮已拒绝的提前放弃/无效决策:")
            lines.extend(f"  - {note}" for note in forced_continue_notes)
        return "\n".join(lines)

    def build_prompt() -> str:
        return self.llm.format_prompt(
            "decide_method.md",
            operator_name=op.name,
            operator_context=self._operator_context(op),
            best_id=run_state.current_best_id,
            benchmark_metrics=bm_text,
            analysis_summary=json.dumps(analysis, ensure_ascii=False, indent=2),
            blacklist=blacklist_text,
            method_history=self._method_history_text(run_state),
            rejected_methods=rejected_methods_text(),
            forced_continue=forced_continue_text(),
            current_iteration=current_iteration,
            max_iterations=max_iterations,
            kb_hints=hints_text,
            hardware_summary=self._hardware_summary(hw),
            library_baseline=library_text,
        )

    # ── [改进] 动态温度 ──
    decide_temp = get_dynamic_decide_temperature(cat_streak)
    if decide_temp != 0.1:
        logger.info(
            "Dynamic decide temperature: %.2f (regression_streak=%d)",
            decide_temp, cat_streak,
        )

    decision: MethodDecision | None = None
    max_reselects = max(0, self.sm.config.decide_reselect_max_retries)
    for attempt in range(max_reselects + 1):
        prompt = build_prompt()
        # [优化] 根据配置选择 Tool Use 或自由 JSON
        if self.sm.config.use_tool_use:
            try:
                tool_result = await self.llm.ainvoke_tool_use(
                    prompt,
                    MethodDecision,
                    temperature=decide_temp,
                    node_name="decide",
                )
                if isinstance(tool_result, MethodDecision):
                    decision = tool_result
                else:
                    decision = MethodDecision.model_validate(tool_result)
            except Exception as e:
                logger.warning("Tool Use failed for decide, falling back to JSON: %s", e)
                decision_data = await self.llm.ainvoke_json(
                    prompt,
                    temperature=decide_temp,
                    node_name="decide",
                )
                decision = MethodDecision.model_validate(decision_data)
        else:
            decision_data = await self.llm.ainvoke_json(
                prompt,
                temperature=decide_temp,
                node_name="decide",
            )
            decision = MethodDecision.model_validate(decision_data)

        if decision.give_up:
            if current_iteration >= max_iterations:
                break
            remaining = max_reselects - attempt
            forced_continue_notes.append(
                f"attempt {attempt + 1}: LLM returned give_up before max_iterations; "
                f"reason={decision.give_up_reason_type or 'unknown'}, method={decision.method_name}"
            )
            logger.warning(
                "LLM attempted give_up at iteration %d/%d; forcing reselection (%d remaining)",
                current_iteration,
                max_iterations,
                remaining,
            )
            if remaining > 0:
                continue
            break

        # ── [改进] 子空间推断和验证 ──
        if not decision.subspace:
            decision.subspace = infer_subspace_from_method_name(decision.method_name)
            logger.info(
                "Inferred subspace '%s' for method '%s'",
                decision.subspace, decision.method_name,
            )

        # 检查 method_name 级黑名单
        if run_state.is_method_blacklisted(decision.method_name):
            normalized = normalize_method_name(decision.method_name)
            rejected_methods.append((decision.method_name, normalized))
            remaining = max_reselects - attempt
            logger.warning(
                "LLM selected blacklisted method %s (attempt %d/%d); %d reselections remaining",
                decision.method_name, attempt + 1, max_reselects + 1, remaining,
            )
            if remaining <= 0:
                decision.give_up = True
                decision.rationale += (
                    " [framework exhausted decide reselection retries after repeated blacklisted methods: "
                    + ", ".join(name for name, _ in rejected_methods)
                    + "]"
                )
                break
            continue

        # ── [改进] 子空间级黑名单匹配 ──
        subspace_blocked, block_reason = run_state.is_subspace_blacklisted(decision.subspace)
        if subspace_blocked:
            normalized = normalize_method_name(decision.method_name)
            rejected_methods.append((decision.method_name, normalized))
            remaining = max_reselects - attempt
            logger.warning(
                "LLM selected method '%s' in exhausted subspace '%s' (%s); %d reselections remaining",
                decision.method_name, decision.subspace, block_reason, remaining,
            )
            if remaining <= 0:
                decision.give_up = True
                decision.give_up_reason_type = "exhausted_search"
                decision.rationale += f" [subspace '{decision.subspace}' exhausted: {block_reason}]"
                break
            continue

        # 通过所有检查
        break

    assert decision is not None

    if decision.give_up and current_iteration < max_iterations:
        fallback_subspace = "algorithm-replacement"
        for candidate_subspace in (
            "register-blocking",
            "algorithm-replacement",
            "launch-overhead-mitigation",
            "fusion",
            "precision-conversion",
        ):
            blocked, _ = run_state.is_subspace_blacklisted(candidate_subspace)
            if not blocked:
                fallback_subspace = candidate_subspace
                break

        decision.give_up = False
        decision.give_up_reason_type = None
        decision.method_name = f"forced_continue_{fallback_subspace.replace('-', '_')}_candidate"
        decision.subspace = fallback_subspace
        decision.has_hyperparams = False
        decision.hyperparams_schema = None
        decision.rationale = (
            "Forced full-iteration mode: max_iterations has not been reached, so an "
            "early give_up decision is converted into a low-risk candidate. Focus on one "
            "small, correctness-preserving change in the selected subspace rather than stopping."
        )
        decision.expected_impact = "low; forced exploration until max_iterations"
        decision.confidence = max(decision.confidence, 0.1)
        decision.differentiation_from_failed = (
            f"Forced continuation fallback in subspace {fallback_subspace}; selected only because "
            "the previous decision attempted to stop before max_iterations."
        )
        decision.falsification_condition = (
            "If this fails, the run still continues only until max_iterations; the failure is "
            "evidence for the next decide prompt, not a stop condition."
        )
        logger.warning(
            "Converted premature give_up into fallback method '%s' at iteration %d/%d",
            decision.method_name,
            current_iteration,
            max_iterations,
        )

    if decision.give_up:
        return {
            "method_decision": decision,
            "should_stop": True,
            "stop_reason": f"LLM gave up ({decision.give_up_reason_type or 'unknown'}): {decision.rationale}",
        }

    return {
        "method_decision": decision,
        "has_hyperparams": decision.has_hyperparams,
    }
