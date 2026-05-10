"""Reflect 节点 —— 分析结果，连续失败时注入外部知识。"""
from __future__ import annotations
import json
import logging
from typing import Any

from ...models.data import BenchmarkResult, IterationRecord, MethodDecision
from ...tools.web_search import (
    format_search_results_for_prompt,
    search_cuda_knowledge,
    search_on_consecutive_failure,
)
from ..temperatures import TEMP_REFLECT

logger = logging.getLogger(__name__)


def _count_consecutive_rejects(iterations: list) -> int:
    count = 0
    for it in reversed(iterations):
        acc = it.get("accepted", False) if isinstance(it, dict) else getattr(it, "accepted", False)
        if not acc:
            count += 1
        else:
            break
    return count


def _failure_reasons(iterations: list, n: int = 3) -> list[str]:
    reasons = []
    for it in reversed(iterations):
        if len(reasons) >= n:
            break
        acc = it.get("accepted", False) if isinstance(it, dict) else getattr(it, "accepted", False)
        if not acc:
            m = (it.get("method") or it.get("method_name") or "?") if isinstance(it, dict) else (getattr(it, "method_name", None) or getattr(it, "method", "?"))
            notes = (it.get("notes", "") if isinstance(it, dict) else getattr(it, "notes", "")) or ""
            reasons.append(f"{m}: {notes[:100]}")
    return reasons


def _selected_hyperparams(state: dict[str, Any], version_id: str) -> dict[str, Any] | None:
    for item in state.get("hp_candidates", []) or []:
        if isinstance(item, dict) and item.get("version_id") == version_id:
            hyperparams = item.get("hyperparams")
            return hyperparams if isinstance(hyperparams, dict) else None
    return None


def _risk_signals_text(
    run_state: Any,
    cfg: Any,
    accepted: bool,
    latency: float | None,
    best_latency: float,
    prior_iterations: list,
) -> str:
    if cfg is None:
        return "(none)"

    launch_floor_ms = getattr(cfg, "launch_floor_ms", 0.005)
    cat_threshold = getattr(cfg, "catastrophic_regression_threshold", 3.0)
    cat_limit = getattr(cfg, "catastrophic_streak_limit", 2)
    tiny_limit = getattr(cfg, "tiny_kernel_reject_limit", 3)
    regime = getattr(run_state, "kernel_regime", {}) or {}
    near_launch_floor = bool(regime.get("near_launch_floor"))
    if best_latency > 0 and launch_floor_ms > 0:
        near_launch_floor = near_launch_floor or best_latency <= launch_floor_ms

    ratio = latency / best_latency if latency and best_latency > 0 else None
    current_catastrophic = bool(not accepted and ratio is not None and ratio >= cat_threshold)
    prior_cat_streak = run_state.catastrophic_regression_streak(cat_threshold) if run_state else 0
    cat_streak = prior_cat_streak + (1 if current_catastrophic else 0)
    reject_streak = _count_consecutive_rejects(prior_iterations) + (0 if accepted else 1)

    lines = [
        f"- launch_floor_ms: {launch_floor_ms}",
        f"- near_launch_floor: {near_launch_floor}",
        f"- regression_ratio: {ratio:.2f}" if ratio is not None else "- regression_ratio: N/A",
        f"- catastrophic_regression_threshold: {cat_threshold}",
        f"- catastrophic_streak_after_trial: {cat_streak}/{cat_limit}",
        f"- tiny_kernel_reject_streak_after_trial: {reject_streak}/{tiny_limit}",
    ]
    if cat_limit > 0 and cat_streak >= cat_limit:
        lines.append("- guidance: catastrophic regression streak reached; recommend lower-risk next changes.")
    if near_launch_floor and tiny_limit > 0 and reject_streak >= tiny_limit:
        lines.append("- guidance: tiny-kernel reject limit reached; avoid extra launches and expensive setup.")
    return "\n".join(lines)


async def reflect_node(state: dict[str, Any], *, llm_client: Any = None,
                        state_manager: Any = None, config: Any = None) -> dict[str, Any]:
    """增强版 Reflect: 连续失败时触发 web search。"""
    run_state = state.get("run_state")
    iterations = list(getattr(run_state, "iterations", []) if run_state else state.get("iterations", []))
    accepted = state.get("trial_accepted", False)
    decision = state.get("method_decision")
    if not isinstance(decision, MethodDecision):
        decision = MethodDecision(method_name=state.get("chosen_method", "unknown"))
    method = decision.method_name
    trial_bm = state.get("trial_benchmark") or BenchmarkResult()
    best_bm = state.get("current_benchmark") or BenchmarkResult()
    latency = trial_bm.latency_ms_median if trial_bm.latency_ms_median > 0 else None
    speedup = 1.0
    if latency and best_bm.latency_ms_median > 0:
        speedup = best_bm.latency_ms_median / latency
    notes = "accepted" if accepted else "rejected"
    version_id = state.get("new_version_id") or state.get("trial_version_id")
    if not version_id and run_state:
        version_id = run_state.next_version_id(has_hp=decision.has_hyperparams)
    version_id = version_id or f"v{state.get('iteration_count', 0) + 1}"
    blacklist = list(state.get("blacklist", []))
    selected_hyperparams = _selected_hyperparams(state, version_id)
    hyperparams_text = json.dumps(selected_hyperparams, ensure_ascii=False, sort_keys=True) if selected_hyperparams else "none"

    cfg = config or state.get("config")
    max_iters = getattr(cfg, "max_iterations", 30) if cfg else 30
    reject_limit = getattr(cfg, "consecutive_reject_limit", 5) if cfg else 5
    search_threshold = getattr(cfg, "web_search_on_failure_threshold", 2) if cfg else 2

    code_path = ""
    iter_dir = None
    if state_manager and state_manager.run_dir:
        iter_dir = state_manager.run_dir / f"iter{version_id}"
        candidate_code_path = iter_dir / "code.cu"
        if candidate_code_path.exists():
            try:
                code_path = str(candidate_code_path.relative_to(state_manager.run_dir))
            except ValueError:
                code_path = str(candidate_code_path)

    record = IterationRecord(
        version_id=version_id,
        parent_id=getattr(run_state, "current_best_id", None) if run_state else None,
        method_name=method,
        has_hyperparams=decision.has_hyperparams,
        hyperparams=selected_hyperparams,
        code_path=code_path,
        accepted=accepted,
        benchmark=trial_bm if trial_bm.latency_ms_median > 0 else None,
        compile_ok=state.get("trial_compile_ok", False),
        correctness_ok=state.get("trial_correctness_ok", False),
    )

    reflection: dict[str, Any] = {}
    if llm_client is not None:
        prompt_name = "reflect_success.md" if accepted else "reflect_failure.md"
        prompt_kwargs = dict(
            iteration=len(iterations) + 1,
            method=method,
            method_name=method,
            hyperparams=hyperparams_text,
            speedup=f"{speedup:.4f}",
            latency_ms=latency if latency is not None else "N/A",
            code_diff_summary="(see generated code)",
            failure_reason=notes,
            failure_details=state.get("correctness_error") or state.get("compile_error") or "",
            consecutive_rejects=_count_consecutive_rejects(iterations),
            external_knowledge=state.get("external_knowledge") or "",
            best_id=getattr(run_state, "current_best_id", "") if run_state else "",
            best_latency_ms=best_bm.latency_ms_median,
            trial_id=version_id,
            trial_latency_ms=latency if latency is not None else "N/A",
            regression_ratio=f"{(latency / best_bm.latency_ms_median):.2f}" if latency and best_bm.latency_ms_median > 0 else "N/A",
            correctness_failure_detail=state.get("correctness_error") or "",
            ncu_report="(see ncu report)",
            kernel_regime=json.dumps(getattr(run_state, "kernel_regime", {}) or {}, ensure_ascii=False) if run_state else "{}",
            risk_signals=_risk_signals_text(run_state, cfg, accepted, latency, best_bm.latency_ms_median, iterations),
        )
        if hasattr(llm_client, "format_prompt"):
            prompt = llm_client.format_prompt(prompt_name, **prompt_kwargs)
        else:
            prompt = ""
        try:
            reflection = await llm_client.ainvoke_json(prompt, temperature=TEMP_REFLECT, node_name="reflect")
        except AttributeError:
            reflection = {}

    if state_manager:
        state_manager.add_iteration(record)
        if accepted and iter_dir is not None:
            state_manager.update_best(version_id, iter_dir)
        elif not accepted:
            try:
                state_manager.add_to_blacklist(
                    method_name=method,
                    reason=reflection.get("why_ineffective") or reflection.get("root_cause") or notes,
                    hp_constraint=record.hyperparams if decision.has_hyperparams else None,
                    failed_at_version=version_id,
                    subspace=decision.subspace or method,
                )
            except Exception as e:
                logger.warning("Could not add blacklist entry: %s", e)

    iterations = list(iterations) + [record]
    consecutive = _count_consecutive_rejects(iterations)

    if run_state and cfg and not accepted:
        cat_threshold = getattr(cfg, "catastrophic_regression_threshold", 3.0)
        cat_limit = getattr(cfg, "catastrophic_streak_limit", 2)
        tiny_limit = getattr(cfg, "tiny_kernel_reject_limit", 3)
        cat_streak = run_state.catastrophic_regression_streak(cat_threshold)
        if cat_limit > 0 and cat_streak >= cat_limit:
            logger.info(
                "Catastrophic regression streak (%d) >= limit (%d) at threshold %.2fx",
                cat_streak, cat_limit, cat_threshold,
            )
        regime = getattr(run_state, "kernel_regime", {}) or {}
        if regime.get("near_launch_floor") and tiny_limit > 0 and consecutive >= tiny_limit:
            logger.info(
                "Tiny-kernel reject streak (%d) >= limit (%d); next prompts will emphasize low-overhead changes.",
                consecutive, tiny_limit,
            )

    if not accepted and method not in blacklist:
        blacklist.append(method)

    ext_knowledge = state.get("external_knowledge")

    if not accepted and search_threshold > 0 and consecutive >= search_threshold:
        logger.info("Consecutive rejects (%d) >= threshold (%d). Triggering web search.", consecutive, search_threshold)
        op = ""
        if run_state:
            spec = getattr(run_state, "operator_spec", None)
            if spec:
                op = getattr(spec, "name", "")
        if not op:
            spec = state.get("operator_spec")
            op = getattr(spec, "name", "kernel") if spec else "kernel"

        try:
            reasons = _failure_reasons(iterations)
            results = await search_on_consecutive_failure(
                operator=op, subspace=method, failure_history=reasons, max_results=5)
            if results:
                ext_knowledge = format_search_results_for_prompt(results)
                logger.info("Injected %d external knowledge results", len(results))

            sub_results = await search_cuda_knowledge(
                operator=op, subspace=method,
                context=f"consecutive failures: {'; '.join(reasons[:2])}", max_results=3)
            if sub_results:
                extra = format_search_results_for_prompt(sub_results)
                ext_knowledge = (ext_knowledge or "") + "\n\n" + extra
        except Exception as e:
            logger.warning("Web search in reflect failed: %s", e)

    should_stop = False
    stop_reason = ""
    if state_manager:
        should_stop, stop_reason = state_manager.should_stop()
    if len(iterations) >= max_iters:
        should_stop = True
        stop_reason = f"Max iterations ({max_iters})"
    if consecutive >= reject_limit:
        should_stop = True
        stop_reason = f"Consecutive reject limit ({reject_limit})"

    # 有效方法列表
    effective = []
    for it in iterations:
        acc = it.get("accepted", False) if isinstance(it, dict) else getattr(it, "accepted", False)
        m = (it.get("method") or it.get("method_name") or "?") if isinstance(it, dict) else (getattr(it, "method_name", None) or getattr(it, "method", "?"))
        if acc:
            effective.append(m)

    return {
        **state,
        "iterations": iterations,
        "should_stop": should_stop,
        "stop_reason": stop_reason,
        "blacklist": blacklist,
        "consecutive_rejects": consecutive,
        "external_knowledge": ext_knowledge,
        "effective_methods_list": effective,
        "iteration_count": len(iterations),
        "run_state": state_manager.state if state_manager else run_state,
        "reflection": reflection or {"speedup": speedup, "notes": notes},
    }
