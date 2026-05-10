"""
Decide 节点 —— 选择下一个优化子空间。

增强上下文:
  - 当前 NCU 剖析数据
  - 优化历史表
  - 黑名单
  - 已有效使用的方法（避免重复）
  - NCU 对应的当前最优代码
  - 外部知识参考
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any

from ...models.data import MethodDecision
from ...models.enums import (
    SUBSPACE_METADATA,
    filter_by_arch,
    get_subspace_meta,
    has_hyperparams,
    infer_subspace_from_method_name,
    normalize_method_name,
)
from ...tools.profile import format_ncu_for_prompt
from ..temperatures import TEMP_DECIDE

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "decide_method.md"


def _load_prompt() -> str:
    if PROMPT_PATH.exists():
        return PROMPT_PATH.read_text(encoding="utf-8")
    return "Select the next optimization subspace."


def _format_history(iterations: list) -> str:
    if not iterations:
        return "No optimization history yet (first iteration)."
    hdr = "| Iter | Method | HP? | Accepted | Latency(ms) | Speedup | Notes |"
    sep = "|------|--------|-----|----------|-------------|---------|-------|"
    rows = [hdr, sep]
    for i, it in enumerate(iterations):
        if isinstance(it, dict):
            m, a, la, sp, n, hp = (it.get("method") or it.get("method_name") or "?", it.get("accepted",False),
                it.get("latency_ms","?"), it.get("speedup","?"),
                (it.get("notes","") or "")[:60], it.get("has_hyperparams",False))
        else:
            m = getattr(it,"method_name", None) or getattr(it,"method","?")
            a = getattr(it,"accepted",False)
            bm = getattr(it, "benchmark", None)
            la = getattr(it,"latency_ms", None)
            if la is None and bm is not None:
                la = getattr(bm, "latency_ms_median", "?")
            sp = getattr(it,"speedup","?")
            n, hp = (getattr(it,"notes","") or "")[:60], getattr(it,"has_hyperparams",False)
        rows.append(f"| {i+1} | {m} | {'Y' if hp else 'N'} | {'✓' if a else '✗'} | {la} | {sp} | {n} |")
    return "\n".join(rows)


def _format_blacklist(bl: list[str]) -> str:
    return "\n".join(f"- {n}" for n in bl) if bl else "No blacklisted subspaces."


def _format_effective_methods(iterations: list) -> str:
    effective = []
    for i, it in enumerate(iterations):
        acc = it.get("accepted",False) if isinstance(it,dict) else getattr(it,"accepted",False)
        m = (it.get("method") or it.get("method_name") or "?") if isinstance(it,dict) else (getattr(it,"method_name", None) or getattr(it,"method","?"))
        sp = it.get("speedup","?") if isinstance(it,dict) else getattr(it,"speedup","?")
        n = (it.get("notes","") if isinstance(it,dict) else getattr(it,"notes","")) or ""
        if acc:
            effective.append(f"- Iter {i+1}: **{m}** (speedup={sp}) - {n[:80]}")
    if not effective:
        return "No effective methods applied yet."
    return ("These methods have already been successfully applied. "
            "Do NOT repeat the same strategy; find a new angle.\n\n"
            + "\n".join(effective))


def _get_best_code(state: dict[str, Any]) -> str:
    code = state.get("current_code") or state.get("current_best_code", "")
    if code:
        return _truncate_code(code, 6000)
    run_state = state.get("run_state")
    if run_state:
        best_id = getattr(run_state, "current_best_id", None)
        run_dir = getattr(run_state, "run_dir", "")
        if best_id and run_dir:
            p = Path(run_dir) / f"iter{best_id}" / "code.cu"
            if p.exists():
                return _truncate_code(p.read_text(encoding="utf-8"), 6000)
        c = getattr(run_state, "current_best_code", "")
        if c:
            return _truncate_code(c, 6000)
    return "No code available."


def _truncate_code(code: str, max_chars: int = 8000) -> str:
    if len(code) <= max_chars:
        return code
    half = max_chars // 2
    return code[:half] + "\n\n... [truncated] ...\n\n" + code[-half:]


def _sm_version(hw: Any) -> int:
    cc = str(getattr(hw, "compute_capability", "") or "")
    digits = "".join(ch for ch in cc if ch.isdigit())
    return int(digits[:2]) if digits else 0


def _blacklist_items(state: dict[str, Any], run_state: Any) -> list[str]:
    items = [str(item) for item in (state.get("blacklist") or [])]
    if run_state:
        for entry in getattr(run_state, "blacklist", []) or []:
            subspace = getattr(entry, "subspace", None)
            if subspace:
                items.append(subspace)
            else:
                items.append(getattr(entry, "method_name_normalized", str(entry)))
    return items


def _operator_context(op: Any) -> str:
    return "\n".join([
        f"- Signature: {getattr(op, 'signature', '') or '(none)'}",
        f"- Dtypes: {json.dumps(getattr(op, 'dtypes', {}) or {}, ensure_ascii=False)}",
        f"- Shapes: {json.dumps(getattr(op, 'shapes', {}) or {}, ensure_ascii=False)}",
        f"- Shape profiles: {json.dumps(getattr(op, 'shape_profiles', []) or [], ensure_ascii=False)}",
        f"- Task description: {getattr(op, 'task_description', '') or '(none)'}",
    ])


def _hardware_summary(hw: Any) -> str:
    if hw is None:
        return "Unknown"
    return (
        f"GPU: {getattr(hw, 'gpu_name', '')}\n"
        f"Compute capability: {getattr(hw, 'compute_capability', '')}\n"
        f"SM count: {getattr(hw, 'sm_count', '')}\n"
        f"Shared memory/block: {getattr(hw, 'shared_mem_per_block_kb', '')} KB\n"
        f"L2 Cache: {getattr(hw, 'l2_cache_mb', '')} MB\n"
        f"Tensor Cores: {'yes' if getattr(hw, 'has_tensor_cores', False) else 'no'}\n"
        f"CUDA version: {getattr(hw, 'cuda_version', '')}"
    )


def _method_history_text(run_state: Any, limit: int = 20) -> str:
    rows = []
    for iteration in getattr(run_state, "iterations", []) or []:
        method = getattr(iteration, "method_name", None)
        if not method:
            continue
        bm = getattr(iteration, "benchmark", None)
        outcome = "no benchmark"
        if not getattr(iteration, "compile_ok", False):
            outcome = "compile failed"
        elif not getattr(iteration, "correctness_ok", False):
            outcome = "failed correctness"
        elif bm is not None:
            outcome = f"{bm.latency_ms_median:.4f} ms"
        hp = getattr(iteration, "hyperparams", None)
        hp_text = json.dumps(hp, ensure_ascii=False, sort_keys=True) if hp else "none"
        rows.append(
            f"| {iteration.version_id} | {method} | {hp_text} | {outcome} | "
            f"{'yes' if iteration.accepted else 'no'} |"
        )
    if not rows:
        return "(none)"
    rows = rows[-limit:]
    return "| version | method | hyperparams | outcome | accepted |\n|---|---|---|---|---|\n" + "\n".join(rows)


def _rejected_methods_text(rejected_methods: list[tuple[str, str]]) -> str:
    if not rejected_methods:
        return "(none)"
    return "\n".join(f"  - {name} (normalized: {normalized})" for name, normalized in rejected_methods)


def _forced_continue_text(current_iteration: int, max_iterations: int, notes: list[str]) -> str:
    lines = [
        f"当前迭代记录数: {current_iteration}",
        f"最大迭代数: {max_iterations}",
        "强制跑满策略: 未达到 max_iterations 前必须继续提出候选方法, 不允许提前停止。",
    ]
    if notes:
        lines.append("本轮已拒绝的提前放弃/无效决策:")
        lines.extend(f"  - {note}" for note in notes)
    return "\n".join(lines)


def _runtime_signals_text(run_state: Any, config: Any) -> str:
    if run_state is None or config is None:
        return "(none)"

    launch_floor_ms = getattr(config, "launch_floor_ms", 0.005)
    cat_threshold = getattr(config, "catastrophic_regression_threshold", 3.0)
    cat_limit = getattr(config, "catastrophic_streak_limit", 2)
    tiny_limit = getattr(config, "tiny_kernel_reject_limit", 3)
    best_latency = run_state.best_latency_ms() if hasattr(run_state, "best_latency_ms") else None
    regime = getattr(run_state, "kernel_regime", {}) or {}
    near_launch_floor = bool(regime.get("near_launch_floor"))
    if best_latency is not None and launch_floor_ms > 0:
        near_launch_floor = near_launch_floor or best_latency <= launch_floor_ms
    cat_streak = run_state.catastrophic_regression_streak(cat_threshold)
    reject_streak = run_state.consecutive_rejects()

    lines = [
        f"- launch_floor_ms: {launch_floor_ms}",
        f"- best_latency_ms: {best_latency if best_latency is not None else 'N/A'}",
        f"- near_launch_floor: {near_launch_floor}",
        f"- catastrophic_regression_threshold: {cat_threshold}",
        f"- catastrophic_streak: {cat_streak}/{cat_limit}",
        f"- tiny_kernel_reject_streak: {reject_streak}/{tiny_limit}",
    ]
    if cat_limit > 0 and cat_streak >= cat_limit:
        lines.append("- guidance: recent attempts caused catastrophic regressions; prefer lower-risk subspaces.")
    if near_launch_floor and tiny_limit > 0 and reject_streak >= tiny_limit:
        lines.append("- guidance: tiny-kernel regime with repeated rejects; avoid extra launches and heavyweight tiling.")
    return "\n".join(lines)


def _coerce_decision(data: Any, fallback: str) -> MethodDecision:
    if isinstance(data, MethodDecision):
        return data
    if not isinstance(data, dict):
        try:
            data = json.loads(str(data))
        except (json.JSONDecodeError, TypeError):
            data = {}
    method_name = data.get("method_name") or data.get("chosen_subspace") or fallback
    subspace = data.get("chosen_subspace") or data.get("subspace") or infer_subspace_from_method_name(method_name)
    hp = data.get("has_hyperparams")
    if hp is None:
        hp = has_hyperparams(subspace or method_name)
    schema = data.get("hyperparams_schema")
    if schema is None and subspace and has_hyperparams(subspace):
        schema = {"subspace": subspace, "hyperparams": get_subspace_meta(subspace).get("hyperparams", [])}
    return MethodDecision(
        method_name=method_name,
        has_hyperparams=bool(hp),
        hyperparams_schema=schema,
        rationale=data.get("rationale", ""),
        expected_impact=data.get("expected_impact") or data.get("expected_improvement", "medium"),
        confidence=data.get("confidence", 0.5),
        give_up=data.get("give_up", False),
        subspace=subspace,
        give_up_reason_type=data.get("give_up_reason_type"),
    )


async def decide_node(state: dict[str, Any], *, llm_client: Any,
                       state_manager: Any = None, kb: Any = None) -> dict[str, Any]:
    """增强版 Decide: 含有效方法 + 代码上下文 + 20 子空间。"""
    run_state = state.get("run_state")
    if run_state is None and state_manager is not None:
        run_state = state_manager.state
    hw = state.get("hardware_spec")
    op = state.get("operator_spec") or getattr(run_state, "operator_spec", None)
    ncu = state.get("ncu_profile") or state.get("current_ncu")
    iterations = state.get("iterations", [])
    blacklist = _blacklist_items(state, run_state)
    ext_knowledge = state.get("external_knowledge")
    config = state.get("config") or getattr(state_manager, "config", None) or getattr(run_state, "config", None)

    if not iterations and run_state:
        iterations = getattr(run_state, "iterations", [])

    max_iters = getattr(config, "max_iterations", 30) if config else state.get("max_iterations", 30)
    iter_count = len(getattr(run_state, "iterations", []) or []) if run_state else state.get("iteration_count", 0)
    if iter_count >= max_iters:
        return {"should_stop": True, "stop_reason": f"Max iterations ({max_iters})"}

    sm = _sm_version(hw)
    available = [s for s in filter_by_arch(sm) if s not in blacklist]
    fallback = available[0] if available else "memory-coalescing"

    ncu_str = format_ncu_for_prompt(ncu) if isinstance(ncu, dict) else str(ncu or "No NCU data")
    rejected_methods: list[tuple[str, str]] = []
    forced_continue_notes: list[str] = []
    max_reselects = max(0, getattr(config, "decide_reselect_max_retries", 0) if config else 0)

    def build_prompt() -> str:
        kwargs = dict(
            operator_name=getattr(op, "name", "kernel") if op else "kernel",
            operator_context=_operator_context(op) if op else "",
            hardware_spec=str(hw) if hw else "Unknown",
            hardware_summary=_hardware_summary(hw),
            current_best_code=_get_best_code(state),
            ncu_profile=ncu_str,
            optimization_history=_format_history(iterations),
            blacklist=_format_blacklist(blacklist),
            effective_methods=_format_effective_methods(iterations),
            external_knowledge=ext_knowledge or "No external knowledge available.",
            method_history=_method_history_text(run_state),
            rejected_methods=_rejected_methods_text(rejected_methods),
            forced_continue=_forced_continue_text(iter_count, max_iters, forced_continue_notes),
            runtime_signals=_runtime_signals_text(run_state, config),
            current_iteration=iter_count,
            max_iterations=max_iters,
            benchmark_metrics=str(state.get("current_benchmark", "")),
            analysis_summary=json.dumps(state.get("analysis_result", {}), ensure_ascii=False, indent=2),
            kb_hints="",
            library_baseline=str(getattr(run_state, "library_baseline_ms", None)) if run_state else "",
        )
        if hasattr(llm_client, "format_prompt"):
            return llm_client.format_prompt("decide_method.md", **kwargs)
        return _load_prompt().format(**kwargs)

    logger.info("Decide: calling LLM with enhanced context (code + effective methods)")
    decision_obj: MethodDecision | None = None
    for attempt in range(max_reselects + 1):
        prompt = build_prompt()
        try:
            response = await llm_client.ainvoke_json(prompt, temperature=TEMP_DECIDE, node_name="decide")
        except AttributeError:
            response = await llm_client.ainvoke_structured(prompt, temperature=TEMP_DECIDE)
        decision_obj = _coerce_decision(response, fallback)

        if decision_obj.give_up and iter_count < max_iters:
            forced_continue_notes.append(
                f"attempt {attempt + 1}: LLM returned give_up before max_iterations; "
                f"method={decision_obj.method_name}"
            )
            if attempt < max_reselects:
                continue
            break

        normalized = normalize_method_name(decision_obj.method_name)
        if normalized in blacklist:
            rejected_methods.append((decision_obj.method_name, normalized))
            if attempt < max_reselects:
                continue
            break

        if decision_obj.subspace in SUBSPACE_METADATA and decision_obj.subspace not in available:
            rejected_methods.append((decision_obj.method_name, normalize_method_name(decision_obj.subspace)))
            if attempt < max_reselects:
                continue
            break

        break

    assert decision_obj is not None

    normalized = normalize_method_name(decision_obj.method_name)
    blocked = normalized in blacklist or (
        decision_obj.subspace in SUBSPACE_METADATA and decision_obj.subspace not in available
    )
    if (decision_obj.give_up or blocked) and iter_count < max_iters:
        fallback_subspace = "algorithm-replacement"
        for candidate in ("register-optimization", "algorithm-replacement", "launch-overhead-mitigation", "fusion"):
            if candidate in available:
                fallback_subspace = candidate
                break
        decision_obj = MethodDecision(
            method_name=f"forced_continue_{fallback_subspace.replace('-', '_')}_candidate",
            has_hyperparams=False,
            rationale="Forced full-iteration mode: max_iterations has not been reached.",
            expected_impact="low; forced exploration until max_iterations",
            confidence=0.1,
            subspace=fallback_subspace,
        )

    if decision_obj.give_up:
        return {
            "method_decision": decision_obj,
            "should_stop": True,
            "stop_reason": f"LLM gave up: {decision_obj.rationale}",
        }

    logger.info("Decide: chose '%s' (HP=%s)", decision_obj.method_name, decision_obj.has_hyperparams)
    return {
        "chosen_method": decision_obj.method_name,
        "method_decision": decision_obj,
        "has_hyperparams": decision_obj.has_hyperparams,
        "decide_rationale": decision_obj.rationale,
        "bottleneck_analysis": state.get("bottleneck_analysis", ""),
    }
