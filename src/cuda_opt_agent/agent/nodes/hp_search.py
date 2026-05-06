from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from ...codegen.normalizer import extract_cuda_code
from ...models.data import HyperparamCandidate, NcuMetrics
from ...tools.correctness import check_correctness_multi, summarize_correctness_results
from ...tools.profile import format_ncu_for_prompt
from ..temperatures import TEMP_APPLY_METHOD, TEMP_PROPOSE_HP

logger = logging.getLogger(__name__)


async def hp_search_node(self, state: dict) -> dict:
    """LLM 提出多组超参候选,逐一编译+测速。"""
    logger.info("=== HP SEARCH ===")
    decision = state["method_decision"]
    hw = state["hardware_spec"]
    op = state["operator_spec"]
    ncu = state.get("current_ncu", NcuMetrics())

    prompt = self.llm.format_prompt(
        "propose_hp.md",
        operator_name=op.name,
        operator_context=self._operator_context(op),
        method_name=decision.method_name,
        method_rationale=decision.rationale,
        hyperparams_schema=json.dumps(decision.hyperparams_schema or {}, indent=2),
        known_hp_trials=self._method_history_text(state["run_state"], decision.method_name),
        ncu_key_metrics=format_ncu_for_prompt(ncu)[:3000],
        hardware_summary=self._hardware_summary(hw),
        hp_count=self.sm.config.hp_candidate_count,
    )

    candidates_raw = await self.llm.ainvoke_json(
        prompt,
        temperature=TEMP_PROPOSE_HP,
        node_name="hp_search",
    )
    if isinstance(candidates_raw, dict):
        candidates_raw = candidates_raw.get("candidates", [candidates_raw])

    candidates = []
    for item in candidates_raw:
        if isinstance(item, dict):
            candidates.append(HyperparamCandidate.model_validate(item))

    candidate_records: dict[str, dict[str, Any]] = {}
    compile_jobs: list[dict[str, Any]] = []
    version_base = state["run_state"].next_version_id(has_hp=True)

    for cand in candidates:
        version_id = f"{version_base}_cand{cand.index}"
        iter_dir = self.sm.create_iteration_dir(version_id)

        hp_section = f"- Hyperparams: {json.dumps(cand.hyperparams)}\n- Rationale: {cand.rationale}"

        apply_prompt = self.llm.format_prompt(
            "apply_method.md",
            operator_name=op.name,
            operator_context=self._operator_context(op),
            method_name=decision.method_name,
            method_rationale=decision.rationale,
            hyperparams_section=hp_section,
            hardware_summary=self._hardware_summary(hw),
            best_id=state["run_state"].current_best_id,
            best_code=state.get("current_code", "")[:8000],
            ncu_key_metrics=format_ncu_for_prompt(ncu)[:2000],
        )

        response = await self.llm.ainvoke(
            apply_prompt,
            temperature=TEMP_APPLY_METHOD,
            node_name=f"hp_search:cand{cand.index}",
        )
        code = extract_cuda_code(response)
        code_path = await asyncio.to_thread(self.sm.persistence.save_code, code, iter_dir)

        candidate_records[version_id] = {
            "candidate": cand,
            "code": code,
            "iter_dir": iter_dir,
        }
        compile_jobs.append({
            "index": cand.index,
            "version_id": version_id,
            "iter_dir": str(iter_dir),
            "code_path": str(code_path),
            "output_path": str(iter_dir / "kernel"),
            "compute_capability": hw.compute_capability,
        })

    results = []
    compiled_candidates = await asyncio.to_thread(self._compile_hp_candidates, compile_jobs)

    for compiled in compiled_candidates:
        version_id = compiled["version_id"]
        record = candidate_records[version_id]
        cand = record["candidate"]
        iter_dir = Path(record["iter_dir"])

        compile_output = (compiled.get("stdout", "") + "\n" + compiled.get("stderr", "")).strip()
        (iter_dir / "compile.log").write_text(compile_output, encoding="utf-8")

        if not compiled.get("success"):
            logger.warning("Candidate %d compilation failed: %s", cand.index, compiled.get("stderr", "")[:500])
            continue
        exe_path = Path(compiled["output_path"])

        dtype = list(op.dtypes.values())[0] if op.dtypes else "fp32"
        correctness_results = await asyncio.to_thread(
            check_correctness_multi,
            exe_path,
            self._active_shape_profiles(op),
            dtype=dtype,
        )
        if not all(r.get("correct") for r in correctness_results):
            logger.warning("Candidate %d correctness check failed: %s", cand.index, summarize_correctness_results(correctness_results))
            continue

        bm = await asyncio.to_thread(self._benchmark_multi, exe_path, op)
        results.append({
            "index": cand.index,
            "version_id": version_id,
            "hyperparams": cand.hyperparams,
            "benchmark": bm,
            "code": record["code"],
            "iter_dir": str(iter_dir),
        })

    if not results:
        return {
            "new_code": "",
            "new_version_id": "",
            "trial_version_id": "",
            "trial_benchmark": None,
            "trial_compile_ok": False,
            "trial_correctness_ok": False,
            "trial_accepted": False,
            "hp_candidates": [],
        }

    best_cand = min(results, key=lambda r: r["benchmark"].latency_ms_median)

    return {
        "new_code": best_cand["code"],
        "new_version_id": best_cand["version_id"],
        "trial_version_id": best_cand["version_id"],
        "trial_benchmark": best_cand["benchmark"],
        "trial_compile_ok": True,
        "trial_correctness_ok": True,
        "hp_candidates": results,
    }
