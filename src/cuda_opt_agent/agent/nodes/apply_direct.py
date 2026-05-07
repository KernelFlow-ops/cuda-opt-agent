"""
Apply Direct 节点 —— LLM 在 best 基础上应用无超参的优化方法。

[修复]:
  - 注入近期 correctness 失败历史
  - 使用智能截断保留 main() 校验框架
  - 支持 use_code_diff: 当代码过长时发送骨架而非完整代码
"""

from __future__ import annotations

import logging

from ...codegen.normalizer import extract_cuda_code
from ...models.data import NcuMetrics
from ...tools.profile import format_ncu_for_prompt
from ..temperatures import TEMP_APPLY_METHOD
from .hp_search import _build_correctness_failure_history, _smart_truncate_code

logger = logging.getLogger(__name__)


async def apply_direct_node(self, state: dict) -> dict:
    """LLM 在 best 基础上应用方法 M(不含超参)。"""
    logger.info("=== APPLY DIRECT ===")
    decision = state["method_decision"]
    hw = state["hardware_spec"]
    op = state["operator_spec"]
    ncu = state.get("current_ncu", NcuMetrics())
    run_state = state["run_state"]

    # [修复] 智能代码截断
    best_code = state.get("current_code", "")
    if self.sm.config.use_code_diff and len(best_code) > 8000:
        from ._helpers import _build_code_diff_context
        ctx = _build_code_diff_context(best_code)
        if ctx["mode"] == "skeleton":
            code_for_prompt = _smart_truncate_code(best_code, max_chars=8000)
        else:
            code_for_prompt = ctx["code"]
    else:
        code_for_prompt = best_code

    # [修复] 构建 correctness 失败历史上下文
    correctness_history = _build_correctness_failure_history(run_state)

    prompt = self.llm.format_prompt(
        "apply_method.md",
        operator_name=op.name,
        operator_context=self._operator_context(op),
        method_name=decision.method_name,
        method_rationale=decision.rationale,
        hyperparams_section="(no hyperparams in this step)",
        hardware_summary=self._hardware_summary(hw),
        best_id=run_state.current_best_id,
        best_code=code_for_prompt,
        ncu_key_metrics=format_ncu_for_prompt(ncu)[:2000],
        correctness_failure_history=correctness_history,
    )

    response = await self.llm.ainvoke(prompt, temperature=TEMP_APPLY_METHOD, node_name="apply_direct")
    code = extract_cuda_code(response)
    version_id = run_state.next_version_id(has_hp=False)

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
        "hp_correctness_failures": [],
        "hp_all_compiled_ok": False,
    }
