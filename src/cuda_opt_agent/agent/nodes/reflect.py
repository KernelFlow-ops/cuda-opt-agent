"""
反思节点 —— LLM 分析本次迭代为什么有效/无效。

[修复]:
  - 区分 3 种失败类型: 编译失败 / correctness 失败 / 性能不够
  - 传递 correctness 失败的详细数值误差信息给 LLM
  - 让 LLM 能够针对具体的数值错误进行有效反思

[改进]:
  - 失败时也写入 KB (negative polarity),让跨 run 学习失败教训
  - 传递 regression_ratio 和 kernel_regime 给 reflect prompt
  - 从 reflection 中提取 anti_pattern 结构化信息,存入 blacklist
  - 从 reflection 中提取 subspace,用于子空间级黑名单
"""

from __future__ import annotations

import json
import logging

from ...models.data import BenchmarkResult, IterationRecord, MethodDecision
from ...models.enums import infer_subspace_from_method_name
from ..temperatures import TEMP_REFLECT

logger = logging.getLogger(__name__)


async def reflect_node(self, state: dict) -> dict:
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
    version_id = state.get("new_version_id") or state.get("trial_version_id")
    if not version_id:
        version_id = run_state.next_version_id(has_hp=decision.has_hyperparams)

    if accepted:
        speedup = (best_bm.latency_ms_median / trial_bm.latency_ms_median
                   if trial_bm.latency_ms_median > 0 else 1.0)

        prompt = self.llm.format_prompt(
            "reflect_success.md",
            method_name=decision.method_name,
            hyperparams=selected_hyperparams_text if decision.has_hyperparams else "none",
            parent_id=run_state.current_best_id,
            parent_latency_ms=best_bm.latency_ms_median,
            new_id=version_id,
            new_latency_ms=trial_bm.latency_ms_median,
            speedup=speedup,
            ncu_diff="(see ncu report for details)",
        )
    else:
        # ── [修复] 区分 3 种失败类型, 传递 correctness 失败详情 ──
        trial_compile_ok = state.get("trial_compile_ok", False)
        trial_correctness_ok = state.get("trial_correctness_ok", False)
        hp_all_compiled = state.get("hp_all_compiled_ok", False)
        hp_corr_failures = state.get("hp_correctness_failures", [])

        failure_reason, correctness_failure_detail = _build_failure_info(
            trial_compile_ok=trial_compile_ok,
            trial_correctness_ok=trial_correctness_ok,
            hp_all_compiled=hp_all_compiled,
            hp_corr_failures=hp_corr_failures,
            trial_bm=trial_bm,
            best_bm=best_bm,
        )

        # ── [改进] 计算回归倍数 ──
        regression_ratio = 0.0
        if trial_bm.latency_ms_median > 0 and best_bm.latency_ms_median > 0:
            regression_ratio = trial_bm.latency_ms_median / best_bm.latency_ms_median

        # ── [改进] kernel regime 信息 ──
        kernel_regime = run_state.kernel_regime
        regime_text = json.dumps(kernel_regime, ensure_ascii=False) if kernel_regime else "(未评估)"

        prompt = self.llm.format_prompt(
            "reflect_failure.md",
            method_name=decision.method_name,
            hyperparams=selected_hyperparams_text if decision.has_hyperparams else "none",
            best_id=run_state.current_best_id,
            best_latency_ms=best_bm.latency_ms_median,
            trial_id=version_id,
            trial_latency_ms=trial_bm.latency_ms_median if trial_bm.latency_ms_median > 0 else "N/A",
            regression_ratio=f"{regression_ratio:.2f}" if regression_ratio > 0 else "N/A",
            failure_reason=failure_reason,
            correctness_failure_detail=correctness_failure_detail,
            ncu_report="(see ncu report)",
            kernel_regime=regime_text,
        )

    reflection = await self.llm.ainvoke_json(prompt, temperature=TEMP_REFLECT, node_name="reflect")

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
                polarity="positive",
            )
    else:
        # ── [改进] 从 reflection 提取 anti_pattern 的子空间和签名 ──
        anti_pattern = reflection.get("anti_pattern", {})
        subspace = anti_pattern.get("subspace") or getattr(decision, "subspace", None)
        if not subspace:
            subspace = infer_subspace_from_method_name(decision.method_name)
        pattern_signature = anti_pattern.get("pattern_signature")
        regression_severity = reflection.get("regression_severity")
        trigger_conditions = anti_pattern.get("trigger_conditions")

        self.sm.add_to_blacklist(
            method_name=decision.method_name,
            reason=reflection.get("why_ineffective", "unknown"),
            hp_constraint=selected_hyperparams if decision.has_hyperparams else None,
            failed_at_version=version_id,
            subspace=subspace,
            pattern_signature=pattern_signature,
            regression_severity=regression_severity,
            trigger_conditions=trigger_conditions,
        )

        # ── [改进] 失败教训也写入 KB (negative polarity) ──
        kb_suggestion = reflection.get("kb_write_suggestion", {})
        if kb_suggestion.get("should_write", False):
            regression_ratio_val = 0.0
            if trial_bm.latency_ms_median > 0 and best_bm.latency_ms_median > 0:
                regression_ratio_val = trial_bm.latency_ms_median / best_bm.latency_ms_median

            # speedup < 1.0 表示负面（回归）
            speedup_for_kb = (
                1.0 / regression_ratio_val
                if regression_ratio_val > 0
                else 0.5
            )

            anti_pattern_str = json.dumps(anti_pattern, ensure_ascii=False) if anti_pattern else ""
            notes = (
                f"[NEGATIVE] {kb_suggestion.get('notes', '')} | "
                f"anti_pattern={anti_pattern_str} | "
                f"regression={regression_ratio_val:.2f}x | "
                f"severity={regression_severity}"
            )

            polarity = kb_suggestion.get("polarity", "negative")
            self.kb.write_entry(
                operator_class=op.name,
                hardware_signature=hw.signature,
                method_name=decision.method_name,
                run_id=run_state.run_id,
                version_id=version_id,
                speedup_vs_parent=speedup_for_kb,
                hyperparams_pattern=selected_hyperparams,
                notes=notes,
                polarity=polarity,
            )
            logger.info(
                "Wrote negative KB entry for %s: regression=%.2fx, subspace=%s",
                decision.method_name, regression_ratio_val, subspace,
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


def _build_failure_info(
    *,
    trial_compile_ok: bool,
    trial_correctness_ok: bool,
    hp_all_compiled: bool,
    hp_corr_failures: list[dict],
    trial_bm: BenchmarkResult,
    best_bm: BenchmarkResult,
) -> tuple[str, str]:
    """
    [修复] 根据失败类型构建精确的 failure_reason 和 correctness_failure_detail。

    Returns:
        (failure_reason, correctness_failure_detail)
    """
    # Case 1: 编译都失败了
    if not trial_compile_ok and not hp_all_compiled:
        return "compilation failed", "(编译阶段失败, 无 correctness 数据)"

    # Case 2: 编译成功但 correctness 失败
    if hp_all_compiled and not trial_correctness_ok:
        failure_reason = (
            "all HP candidates compiled successfully but FAILED correctness check"
        )
        if hp_corr_failures:
            detail_lines = [
                f"共 {len(hp_corr_failures)} 个候选的 correctness 失败详情:"
            ]
            for f in hp_corr_failures[:5]:  # 最多展示 5 个候选
                cand_idx = f.get("candidate_index", "?")
                hp = json.dumps(f.get("hyperparams", {}), ensure_ascii=False)
                repair_n = f.get("repair_attempts", 0)
                detail_lines.append(
                    f"\n  候选 {cand_idx} (hp={hp}, 修复尝试={repair_n}次):"
                )
                for err in f.get("errors", [])[:3]:  # 每个候选最多 3 个 shape
                    detail_lines.append(
                        f"    - shape={err.get('shape', '?')}: "
                        f"max_abs_error={_fmt_error(err.get('max_abs_error'))}, "
                        f"max_rel_error={_fmt_error(err.get('max_rel_error'))}, "
                        f"msg={err.get('message', '?')}"
                    )
                # 展示累积的修复错误历史
                for ae in f.get("accumulated_errors", [])[:2]:
                    detail_lines.append(f"    修复历史: {ae[:200]}")
            correctness_detail = "\n".join(detail_lines)
        else:
            correctness_detail = "(编译成功但 correctness 失败, 无详细错误信息)"
        return failure_reason, correctness_detail

    # Case 3: 编译和 correctness 都通过但性能不够
    if trial_compile_ok and trial_correctness_ok:
        if trial_bm.latency_ms_median > 0 and best_bm.latency_ms_median > 0:
            ratio = trial_bm.latency_ms_median / best_bm.latency_ms_median
            return (
                f"no speedup (trial={trial_bm.latency_ms_median:.4f}ms vs "
                f"best={best_bm.latency_ms_median:.4f}ms, ratio={ratio:.4f})"
            ), "(性能未达改善阈值, correctness 通过)"
        return "no speedup", "(性能未达改善阈值)"

    # Case 4: 兜底
    return "compile/correctness failed", "(未知失败类型)"


def _fmt_error(val) -> str:
    """格式化错误值, 处理 inf 和 None。"""
    if val is None:
        return "?"
    try:
        fval = float(val)
        if fval == float("inf"):
            return "inf"
        return f"{fval:.6g}"
    except (TypeError, ValueError):
        return str(val)
