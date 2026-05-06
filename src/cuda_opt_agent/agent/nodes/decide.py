from __future__ import annotations

import json
import logging

from ...models.data import BenchmarkResult, MethodDecision
from ...models.enums import normalize_method_name
from ..temperatures import TEMP_DECIDE

logger = logging.getLogger(__name__)


def decide_node(self, state: dict) -> dict:
    """LLM 决策下一个优化方法。"""
    logger.info("=== DECIDE ===")
    run_state = state["run_state"]
    op = state["operator_spec"]
    hw = state["hardware_spec"]

    analysis = state.get("analysis_result", {})
    bm = state.get("current_benchmark", BenchmarkResult())

    bl_lines = []
    for entry in run_state.blacklist:
        hp_str = json.dumps(entry.hyperparam_constraint) if entry.hyperparam_constraint else "no hyperparams"
        bl_lines.append(f"  - {entry.method_name_normalized} ({hp_str}): {entry.reason}")
    blacklist_text = "\n".join(bl_lines) or "(none)"

    kb_hints = self.kb.query(op.name, hw.signature)
    hints_text = self.kb.format_hints_for_prompt(kb_hints)

    bm_text = (
        f"latency_median: {bm.latency_ms_median:.4f} ms\n"
        f"latency_p95: {bm.latency_ms_p95:.4f} ms\n"
        f"aggregator: {bm.extra.get('aggregator', 'single')}\n"
        f"per_shape: {self._per_shape_summary(bm).replace('<br>', '; ') or 'N/A'}"
    )

    rejected_methods: list[tuple[str, str]] = []

    def rejected_methods_text() -> str:
        if not rejected_methods:
            return "(none)"
        return "\n".join(
            f"  - {name} (normalized: {normalized})"
            for name, normalized in rejected_methods
        )

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
            kb_hints=hints_text,
            hardware_summary=self._hardware_summary(hw),
        )

    decision: MethodDecision | None = None
    max_reselects = max(0, self.sm.config.decide_reselect_max_retries)
    for attempt in range(max_reselects + 1):
        decision_data = self.llm.invoke_json(build_prompt(), temperature=TEMP_DECIDE)
        decision = MethodDecision.model_validate(decision_data)

        if decision.give_up:
            break

        if not run_state.is_method_blacklisted(decision.method_name):
            break

        normalized = normalize_method_name(decision.method_name)
        rejected_methods.append((decision.method_name, normalized))
        remaining = max_reselects - attempt
        logger.warning(
            "LLM selected blacklisted method %s (attempt %d/%d); %d reselections remaining",
            decision.method_name,
            attempt + 1,
            max_reselects + 1,
            remaining,
        )
        if remaining <= 0:
            decision.give_up = True
            decision.rationale += (
                " [framework exhausted decide reselection retries after repeated blacklisted methods: "
                + ", ".join(name for name, _ in rejected_methods)
                + "]"
            )
            break

    assert decision is not None

    if decision.give_up:
        return {
            "method_decision": decision,
            "should_stop": True,
            "stop_reason": f"LLM gave up: {decision.rationale}",
        }

    return {
        "method_decision": decision,
        "has_hyperparams": decision.has_hyperparams,
    }
