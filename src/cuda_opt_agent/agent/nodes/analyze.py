from __future__ import annotations

import logging

from ...models.data import BenchmarkResult, NcuMetrics
from ...tools.profile import format_ncu_for_prompt
from ..temperatures import TEMP_ANALYZE

logger = logging.getLogger(__name__)


async def analyze_node(self, state: dict) -> dict:
    """LLM 分析瓶颈。"""
    logger.info("=== ANALYZE ===")
    run_state = state["run_state"]
    op = state["operator_spec"]
    hw = state["hardware_spec"]

    history_lines = []
    for it in run_state.iterations:
        bm_str = f"{it.benchmark.latency_ms_median:.4f}ms" if it.benchmark else "N/A"
        status = "accepted" if it.accepted else "rejected"
        history_lines.append(
            f"  {it.version_id}: {it.method_name or 'baseline'} -> {bm_str} {status}"
        )
    history_text = "\n".join(history_lines) or "(no history)"

    kb_hints = self.kb.query(op.name, hw.signature)
    hints_text = self.kb.format_hints_for_prompt(kb_hints)

    ncu = state.get("current_ncu", NcuMetrics())
    ncu_text = format_ncu_for_prompt(ncu)

    bm = state.get("current_benchmark", BenchmarkResult())
    bm_text = (
        f"latency_median: {bm.latency_ms_median:.4f} ms\n"
        f"latency_p95: {bm.latency_ms_p95:.4f} ms\n"
        f"throughput: {bm.throughput_gflops or 'N/A'} GFLOPS\n"
        f"aggregator: {bm.extra.get('aggregator', 'single')}\n"
        f"per_shape: {self._per_shape_summary(bm).replace('<br>', '; ') or 'N/A'}"
    )

    prompt = self.llm.format_prompt(
        "analyze.md",
        operator_name=op.name,
        operator_context=self._operator_context(op),
        hardware_summary=self._hardware_summary(hw),
        best_id=run_state.current_best_id,
        best_code=state.get("current_code", "")[:8000],
        ncu_report=ncu_text,
        benchmark_metrics=bm_text,
        iteration_history=history_text,
        kb_hints=hints_text,
    )

    analysis = await self.llm.ainvoke_json(prompt, temperature=TEMP_ANALYZE, node_name="analyze")
    return {"analysis_result": analysis}
