from __future__ import annotations

import json
import logging

from ...codegen.normalizer import extract_cuda_code
from ..temperatures import TEMP_BOOTSTRAP

logger = logging.getLogger(__name__)


async def bootstrap_node(self, state: dict) -> dict:
    """LLM 生成 v0 baseline。"""
    logger.info("=== BOOTSTRAP: generating v0 baseline ===")
    op = state["operator_spec"]
    hw = state["hardware_spec"]

    kb_hints = self.kb.query(op.name, hw.signature)
    hints_text = self.kb.format_hints_for_prompt(kb_hints)
    kb_section = f"## 历史经验（仅供参考）\n{hints_text}" if kb_hints else ""

    seed_code_section = ""
    bootstrap_mode_instruction = "当前没有已有实现,请从零生成一个正确性优先的 v0 baseline。"
    if op.seed_code_path:
        seed_code = self._read_seed_code(op.seed_code_path)
        seed_code_section = (
            f"## 已有 v0 种子代码\n"
            f"路径: {op.seed_code_path}\n"
            f"```cuda\n{seed_code}\n```"
        )
        bootstrap_mode_instruction = (
            "以下代码已经实现该算子,请将其作为 v0 baseline。"
            "如果缺少正确性检查或 benchmark 框架,请补齐;不要修改算法逻辑,"
            "只做必要的封装、命令行参数和输出格式适配。"
        )

    prompt = self.llm.format_prompt(
        "bootstrap.md",
        operator_name=op.name,
        signature=op.signature,
        dtypes=json.dumps(op.dtypes, ensure_ascii=False),
        shapes=json.dumps(op.shapes, ensure_ascii=False),
        shape_profiles=json.dumps(op.shape_profiles, ensure_ascii=False),
        task_description=op.task_description or "无",
        constraints="\n".join(op.constraints) or "无",
        bootstrap_mode_instruction=bootstrap_mode_instruction,
        seed_code_section=seed_code_section,
        gpu_name=hw.gpu_name,
        compute_capability=hw.compute_capability,
        sm_count=hw.sm_count,
        shared_mem_per_block_kb=hw.shared_mem_per_block_kb,
        l2_cache_mb=hw.l2_cache_mb,
        has_tensor_cores=hw.has_tensor_cores,
        cuda_version=hw.cuda_version,
        kb_hints_section=kb_section,
    )

    response = await self.llm.ainvoke(prompt, temperature=TEMP_BOOTSTRAP, node_name="bootstrap")
    code = extract_cuda_code(response)
    return {"current_code": code, "new_version_id": "v0"}


