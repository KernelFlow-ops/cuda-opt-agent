"""
编译 + 数值校验节点。

[优化]:
  - _repair_code: 累积前几轮的错误信息, 帮助 LLM 跳出连续失败循环
  - compile_and_validate_node: 使用 check_correctness_multi_async 跨 shape 并行校验
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from ...codegen.normalizer import extract_cuda_code
from ...models.data import IterationRecord
from ...tools.compile import compile_cuda
from ...tools.correctness import check_correctness_multi_async, summarize_correctness_results
from ...tools.ref_eval import run_ref_correctness_multi
from ..temperatures import TEMP_REPAIR

logger = logging.getLogger(__name__)


def _format_correctness_log(results: list[dict]) -> str:
    lines = []
    for result in results:
        lines.append(
            f"{result.get('shape_label', 'shape')}: "
            f"correct={result.get('correct')} "
            f"compile_ok={result.get('compile_ok', result.get('correct'))} "
            f"max_abs={result.get('max_abs_error')} "
            f"max_rel={result.get('max_rel_error')} "
            f"message={result.get('message', '')}"
        )
    return "\n".join(lines)


async def compile_and_validate_node(self, state: dict) -> dict:
    """
    编译 + 数值校验。失败时触发修复循环。

    [优化]:
      - 累积所有历史错误信息传递给修复 LLM
      - 使用 check_correctness_multi_async 跨 shape 并行校验
      - 编译使用 nvcc_threads 参数
    """
    logger.info("=== COMPILE & VALIDATE ===")
    code = state.get("new_code") or state.get("current_code", "")
    version_id = state.get("new_version_id", "v0")
    hw = state["hardware_spec"]
    op = state["operator_spec"]
    max_retries = self.sm.config.compile_repair_max_retries

    iter_dir = self.sm.create_iteration_dir(version_id)
    code_path = await asyncio.to_thread(self.sm.persistence.save_code, code, iter_dir)

    compile_ok = False
    correctness_ok = False

    # [优化] 累积所有错误信息, 让 LLM 看到完整的失败历史
    accumulated_errors: list[str] = []
    compile_log_entries: list[str] = []
    ref_path = self._ref_py_path(state, self.sm.run_dir)
    use_ref_runner = bool(ref_path and ref_path.exists())

    for attempt in range(max_retries + 1):
        if use_ref_runner:
            dtype = list(op.dtypes.values())[0] if op.dtypes else "fp32"
            correctness_results = await asyncio.to_thread(
                run_ref_correctness_multi,
                ref_path,
                code_path,
                self._active_shape_profiles(op),
                func_name=self._kernel_function_name(op),
                compute_capability=hw.compute_capability,
                dtype=dtype,
            )
            compile_ok = all(r.get("compile_ok", r.get("correct")) for r in correctness_results)
            correctness_ok = all(r.get("correct") for r in correctness_results)
            compile_log_entries.append(
                f"[Attempt {attempt + 1} - ref.py {'passed' if correctness_ok else 'failed'}]\n"
                f"{_format_correctness_log(correctness_results)}"
            )
            await asyncio.to_thread(
                (iter_dir / "compile.log").write_text,
                "\n\n".join(compile_log_entries),
                encoding="utf-8",
            )
            if correctness_ok:
                break

            message = summarize_correctness_results(correctness_results)
            failure_kind = "Correctness failed" if compile_ok else "Compile failed"
            logger.warning("%s (attempt %d): %s", failure_kind, attempt, message)
            accumulated_errors.append(
                f"[Attempt {attempt + 1} - {failure_kind}]: {message}"
            )
            if attempt < max_retries:
                code = await self._repair_code(code, accumulated_errors, hw)
                code_path = await asyncio.to_thread(
                    self.sm.persistence.save_code,
                    code,
                    iter_dir,
                    f"code_fix{attempt + 1}.cu",
                )
            continue

        cr = await asyncio.to_thread(
            compile_cuda,
            code_path,
            iter_dir / "kernel",
            hw.compute_capability,
            nvcc_threads=self.sm.config.nvcc_parallel_threads,
        )
        compile_output = (cr.stdout + "\n" + cr.stderr).strip()
        compile_log_entries.append(
            f"[Attempt {attempt + 1} - "
            f"{'Compile succeeded' if cr.success else 'Compile failed'}]\n"
            f"{compile_output}"
        )
        await asyncio.to_thread(
            (iter_dir / "compile.log").write_text,
            "\n\n".join(compile_log_entries),
            encoding="utf-8",
        )

        if cr.success:
            compile_ok = True
            exe_path = Path(cr.output_path)

            dtype = list(op.dtypes.values())[0] if op.dtypes else "fp32"

            # [优化] 使用 check_correctness_multi_async 跨 shape 并行校验
            correctness_results = await check_correctness_multi_async(
                exe_path,
                self._active_shape_profiles(op),
                dtype=dtype,
                max_parallel=self.sm.config.correctness_max_parallel,
            )
            if all(r.get("correct") for r in correctness_results):
                correctness_ok = True
                break

            message = summarize_correctness_results(correctness_results)
            compile_log_entries.append(
                f"[Attempt {attempt + 1} - Correctness failed]\n{message}"
            )
            await asyncio.to_thread(
                (iter_dir / "compile.log").write_text,
                "\n\n".join(compile_log_entries),
                encoding="utf-8",
            )
            logger.warning("Correctness failed (attempt %d): %s", attempt, message)

            # [优化] 累积正确性错误
            accumulated_errors.append(
                f"[Attempt {attempt + 1} - Correctness failed]: {message}"
            )

            if attempt < max_retries:
                code = await self._repair_code(
                    code, accumulated_errors, hw
                )
                code_path = await asyncio.to_thread(
                    self.sm.persistence.save_code,
                    code,
                    iter_dir,
                    f"code_fix{attempt + 1}.cu",
                )
        else:
            logger.warning("Compilation failed (attempt %d); details written to %s",
                           attempt, iter_dir / "compile.log")

            # [优化] 累积编译错误
            accumulated_errors.append(
                f"[Attempt {attempt + 1} - Compile failed]: {compile_output[:2000]}"
            )

            if attempt < max_retries:
                code = await self._repair_code(
                    code, accumulated_errors, hw
                )
                code_path = await asyncio.to_thread(
                    self.sm.persistence.save_code,
                    code,
                    iter_dir,
                    f"code_fix{attempt + 1}.cu",
                )

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


async def _repair_code(
    self,
    code: str,
    accumulated_errors: list[str],
    hw,
) -> str:
    """
    LLM 修复编译/校验错误。

    [优化] 累积前几轮的错误信息, 帮助 LLM 跳出连续失败循环。
    """
    logger.info("Attempting LLM repair (with %d accumulated errors)...", len(accumulated_errors))

    # [优化] 构建累积错误上下文
    error_history = "\n\n".join(accumulated_errors)

    # 当有多次失败时, 添加显式提示
    cumulative_hint = ""
    if len(accumulated_errors) > 1:
        cumulative_hint = (
            f"\n\n## ⚠️ 重要: 这已经是第 {len(accumulated_errors)} 次修复尝试\n"
            f"前几次修复都未解决问题。请仔细分析**所有**历史错误信息, "
            f"找到根本原因, 而非只修表面症状。"
            f"如果之前的修复方向错误, 请尝试完全不同的修复策略。\n"
        )

    prompt = self.llm.format_prompt(
        "repair_compile.md",
        compile_error=error_history[:5000],
        code=code,
        compute_capability=hw.compute_capability,
        cuda_version=hw.cuda_version,
        cumulative_hint=cumulative_hint,
    )
    response = await self.llm.ainvoke(
        prompt,
        temperature=TEMP_REPAIR,
        node_name="compile_validate:repair",
    )
    return extract_cuda_code(response)
