"""
评估节点 —— 比较 trial 与 best 的性能。

[修复]:
  - 透传 hp_correctness_failures 和 hp_all_compiled_ok 给 reflect 节点
"""

from __future__ import annotations

import asyncio
import logging

from ...models.data import BenchmarkResult

logger = logging.getLogger(__name__)


async def evaluate_node(self, state: dict) -> dict:
    """评估 v★ 是否优于 best。"""
    logger.info("=== EVALUATE ===")
    run_state = state["run_state"]
    epsilon = self.sm.config.accept_epsilon

    version_id = state.get("new_version_id", "")
    if not version_id:
        return {
            "trial_version_id": "",
            "trial_benchmark": None,
            "trial_accepted": False,
            "trial_compile_ok": state.get("trial_compile_ok", False),
            "trial_correctness_ok": state.get("trial_correctness_ok", False),
            # [修复] 透传 correctness 失败详情
            "hp_correctness_failures": state.get("hp_correctness_failures", []),
            "hp_all_compiled_ok": state.get("hp_all_compiled_ok", False),
        }

    if state.get("trial_benchmark") and state.get("trial_version_id") == version_id:
        trial_bm = state["trial_benchmark"]
        trial_compile_ok = state.get("trial_compile_ok", True)
        trial_correctness_ok = state.get("trial_correctness_ok", True)
    else:
        result = await self.compile_and_validate_node(state)
        if not result.get("trial_compile_ok") or not result.get("trial_correctness_ok"):
            return {
                "trial_version_id": version_id,
                "trial_benchmark": None,
                "trial_accepted": False,
                "trial_compile_ok": result.get("trial_compile_ok", False),
                "trial_correctness_ok": result.get("trial_correctness_ok", False),
                # [修复] 透传 correctness 失败详情
                "hp_correctness_failures": state.get("hp_correctness_failures", []),
                "hp_all_compiled_ok": state.get("hp_all_compiled_ok", False),
            }

        iter_dir = self.sm.run_dir / f"iter{version_id}"
        exe_path = self._kernel_executable(iter_dir)
        trial_bm = await asyncio.to_thread(self._benchmark_multi, exe_path, run_state.operator_spec)
        trial_compile_ok = result.get("trial_compile_ok", False)
        trial_correctness_ok = result.get("trial_correctness_ok", False)

    best_bm = state.get("current_benchmark", BenchmarkResult())

    accepted = False
    if best_bm.latency_ms_median > 0 and trial_bm.latency_ms_median > 0:
        threshold = best_bm.latency_ms_median * (1 - epsilon)
        accepted = trial_bm.latency_ms_median < threshold

    logger.info(
        "Evaluation: best=%.4fms, trial=%.4fms, threshold=%.4fms -> %s",
        best_bm.latency_ms_median, trial_bm.latency_ms_median,
        best_bm.latency_ms_median * (1 - epsilon),
        "ACCEPTED" if accepted else "REJECTED",
    )

    return {
        "trial_version_id": version_id,
        "trial_benchmark": trial_bm,
        "trial_accepted": accepted,
        "trial_compile_ok": trial_compile_ok,
        "trial_correctness_ok": trial_correctness_ok,
        # [修复] 透传 correctness 失败详情
        "hp_correctness_failures": state.get("hp_correctness_failures", []),
        "hp_all_compiled_ok": state.get("hp_all_compiled_ok", False),
    }
