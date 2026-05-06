from __future__ import annotations

import logging

from ...codegen.normalizer import extract_cuda_code
from ...models.data import NcuMetrics
from ...tools.profile import format_ncu_for_prompt
from ..temperatures import TEMP_APPLY_METHOD

logger = logging.getLogger(__name__)


def apply_direct_node(self, state: dict) -> dict:
    """LLM 在 best 基础上应用方法 M(不含超参)。"""
    logger.info("=== APPLY DIRECT ===")
    decision = state["method_decision"]
    hw = state["hardware_spec"]
    op = state["operator_spec"]
    ncu = state.get("current_ncu", NcuMetrics())

    prompt = self.llm.format_prompt(
        "apply_method.md",
        operator_name=op.name,
        operator_context=self._operator_context(op),
        method_name=decision.method_name,
        method_rationale=decision.rationale,
        hyperparams_section="(no hyperparams in this step)",
        hardware_summary=self._hardware_summary(hw),
        best_id=state["run_state"].current_best_id,
        best_code=state.get("current_code", "")[:8000],
        ncu_key_metrics=format_ncu_for_prompt(ncu)[:2000],
    )

    response = self.llm.invoke(prompt, temperature=TEMP_APPLY_METHOD)
    code = extract_cuda_code(response)
    version_id = state["run_state"].next_version_id(has_hp=False)

    return {
        "new_code": code,
        "new_version_id": version_id,
        "trial_version_id": "",
        "trial_benchmark": None,
        "trial_ncu": None,
        "trial_accepted": False,
        "trial_compile_ok": False,
        "trial_correctness_ok": False,
        "hp_candidates": [],
    }
