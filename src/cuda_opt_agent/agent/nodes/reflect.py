from __future__ import annotations

import json
import logging

from ...models.data import BenchmarkResult, IterationRecord, MethodDecision
from ..temperatures import TEMP_REFLECT

logger = logging.getLogger(__name__)


def reflect_node(self, state: dict) -> dict:
    """LLM 反思:为什么有效/无效。"""
    logger.info("=== REFLECT ===")
    decision = state.get("method_decision", MethodDecision(method_name="unknown"))
    accepted = state.get("trial_accepted", False)
    run_state = state["run_state"]
    op = state["operator_spec"]
    hw = state["hardware_spec"]

    trial_bm = state.get("trial_benchmark") or BenchmarkResult()
    best_bm = state.get("current_benchmark") or BenchmarkResult()
    selected_hyperparams = self._selected_hyperparams(state)
    selected_hyperparams_text = self._hyperparams_text(selected_hyperparams)

    if accepted:
        speedup = (best_bm.latency_ms_median / trial_bm.latency_ms_median
                   if trial_bm.latency_ms_median > 0 else 1.0)

        prompt = self.llm.format_prompt(
            "reflect_success.md",
            method_name=decision.method_name,
            hyperparams=selected_hyperparams_text if decision.has_hyperparams else "none",
            parent_id=run_state.current_best_id,
            parent_latency_ms=best_bm.latency_ms_median,
            new_id=state.get("new_version_id", "?"),
            new_latency_ms=trial_bm.latency_ms_median,
            speedup=speedup,
            ncu_diff="(see ncu report for details)",
        )
    else:
        prompt = self.llm.format_prompt(
            "reflect_failure.md",
            method_name=decision.method_name,
            hyperparams=selected_hyperparams_text if decision.has_hyperparams else "none",
            best_id=run_state.current_best_id,
            best_latency_ms=best_bm.latency_ms_median,
            trial_id=state.get("new_version_id", "?"),
            trial_latency_ms=trial_bm.latency_ms_median if trial_bm.latency_ms_median > 0 else "N/A",
            failure_reason="no speedup" if state.get("trial_compile_ok") else "compile/correctness failed",
            ncu_report="(see ncu report)",
        )

    reflection = self.llm.invoke_json(prompt, temperature=TEMP_REFLECT)

    version_id = state.get("new_version_id", "unknown")
    code_path = ""
    if self.sm.run_dir:
        candidate_code_path = self.sm.run_dir / f"iter{version_id}" / "code.cu"
        if candidate_code_path.exists():
            code_path = str(candidate_code_path.relative_to(self.sm.run_dir))
    record = IterationRecord(
        version_id=version_id,
        parent_id=run_state.current_best_id,
        method_name=decision.method_name,
        has_hyperparams=decision.has_hyperparams,
        hyperparams=selected_hyperparams,
        code_path=code_path,
        benchmark=trial_bm if trial_bm.latency_ms_median > 0 else None,
        compile_ok=state.get("trial_compile_ok", False),
        correctness_ok=state.get("trial_correctness_ok", False),
        accepted=accepted,
    )

    self.sm.add_iteration(record)

    if accepted:
        iter_dir = self.sm.run_dir / f"iter{version_id}"
        self.sm.update_best(version_id, iter_dir)

        kb_suggestion = reflection.get("kb_write_suggestion", {})
        if kb_suggestion.get("should_write", False):
            speedup = (best_bm.latency_ms_median / trial_bm.latency_ms_median
                       if trial_bm.latency_ms_median > 0 else 1.0)
            self.kb.write_entry(
                operator_class=op.name,
                hardware_signature=hw.signature,
                method_name=decision.method_name,
                run_id=run_state.run_id,
                version_id=version_id,
                speedup_vs_parent=speedup,
                notes=kb_suggestion.get("notes", ""),
            )
    else:
        self.sm.add_to_blacklist(
            method_name=decision.method_name,
            reason=reflection.get("why_ineffective", "unknown"),
            hp_constraint=selected_hyperparams if decision.has_hyperparams else None,
            failed_at_version=version_id,
        )

    if self.sm.run_dir:
        reasoning_text = (
            f"## {version_id} · {'ACCEPTED' if accepted else 'REJECTED'}\n\n"
            f"### Method: {decision.method_name}\n"
            f"### Rationale: {decision.rationale}\n\n"
            f"### Reflection\n```json\n{json.dumps(reflection, ensure_ascii=False, indent=2)}\n```\n"
        )
        self.sm.persistence.save_reasoning_log(reasoning_text, self.sm.run_dir)

    should_stop, stop_reason = self.sm.should_stop()

    return {
        "reflection": reflection,
        "should_stop": should_stop,
        "stop_reason": stop_reason,
        "run_state": self.sm.state,
    }
