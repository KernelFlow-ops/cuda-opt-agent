from __future__ import annotations

import logging
from pathlib import Path

from ...codegen.normalizer import extract_cuda_code
from ...models.data import IterationRecord
from ...tools.compile import compile_cuda
from ...tools.correctness import check_correctness_multi, summarize_correctness_results
from ..temperatures import TEMP_REPAIR

logger = logging.getLogger(__name__)


def compile_and_validate_node(self, state: dict) -> dict:
    """编译 + 数值校验。失败时触发修复循环。"""
    logger.info("=== COMPILE & VALIDATE ===")
    code = state.get("new_code") or state.get("current_code", "")
    version_id = state.get("new_version_id", "v0")
    hw = state["hardware_spec"]
    op = state["operator_spec"]
    max_retries = self.sm.config.compile_repair_max_retries

    iter_dir = self.sm.create_iteration_dir(version_id)
    code_path = self.sm.persistence.save_code(code, iter_dir)

    compile_ok = False
    correctness_ok = False

    for attempt in range(max_retries + 1):
        cr = compile_cuda(
            code_path,
            output_path=iter_dir / "kernel",
            compute_capability=hw.compute_capability,
        )

        if cr.success:
            compile_ok = True
            exe_path = Path(cr.output_path)
            (iter_dir / "compile.log").write_text(cr.stdout + "\n" + cr.stderr, encoding="utf-8")

            dtype = list(op.dtypes.values())[0] if op.dtypes else "fp32"
            correctness_results = check_correctness_multi(
                exe_path,
                self._active_shape_profiles(op),
                dtype=dtype,
            )
            if all(r.get("correct") for r in correctness_results):
                correctness_ok = True
                break

            message = summarize_correctness_results(correctness_results)
            logger.warning("Correctness failed (attempt %d): %s", attempt, message)
            if attempt < max_retries:
                code = self._repair_code(code, f"Correctness failed: {message}", hw)
                code_path = self.sm.persistence.save_code(code, iter_dir, f"code_fix{attempt+1}.cu")
        else:
            compile_output = (cr.stdout + "\n" + cr.stderr).strip()
            (iter_dir / "compile.log").write_text(compile_output, encoding="utf-8")
            logger.warning("Compilation failed (attempt %d); details written to %s", attempt, iter_dir / "compile.log")
            if attempt < max_retries:
                code = self._repair_code(code, compile_output, hw)
                code_path = self.sm.persistence.save_code(code, iter_dir, f"code_fix{attempt+1}.cu")

    if compile_ok and correctness_ok and version_id == "v0" and self.sm.state:
        if self.sm.state.iter_by_id(version_id) is None:
            try:
                relative_code_path = str(code_path.relative_to(self.sm.run_dir))
            except ValueError:
                relative_code_path = str(code_path)
            self.sm.add_iteration(IterationRecord(
                version_id=version_id,
                parent_id=None,
                method_name=None,
                has_hyperparams=False,
                code_path=relative_code_path,
                compile_ok=True,
                correctness_ok=True,
                accepted=True,
            ))

    return {
        "current_code": code,
        "trial_version_id": version_id,
        "trial_compile_ok": compile_ok,
        "trial_correctness_ok": correctness_ok,
        "run_state": self.sm.state,
    }


def _repair_code(self, code: str, error: str, hw) -> str:
    """LLM 修复编译/校验错误。"""
    logger.info("Attempting LLM repair...")
    prompt = self.llm.format_prompt(
        "repair_compile.md",
        compile_error=error[:3000],
        code=code,
        compute_capability=hw.compute_capability,
        cuda_version=hw.cuda_version,
    )
    response = self.llm.invoke(prompt, temperature=TEMP_REPAIR)
    return extract_cuda_code(response)
