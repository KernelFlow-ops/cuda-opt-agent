"""
HP Search 节点 —— LLM 提出多组超参候选, 并发生成+编译+测速。

[优化]:
  1. LLM 代码生成并发化: asyncio.gather + Semaphore
  2. 端到端流水线化: 生成→编译→校验→benchmark 各阶段重叠
  3. 多 GPU 分发: 候选分配到不同 GPU 并行 benchmark
  4. 编译使用 _compile_hp_candidates_async (基于 as_completed)
  5. 正确性校验使用 check_correctness_multi_async (跨 shape 并行)

[修复]:
  6. 添加 HP 编译失败修复循环 (对标 compile_validate 的 _repair_code)
  7. 添加 correctness 修复循环
  8. 收集 correctness 失败详情, 传递给 reflect 节点
  9. 智能代码截断, 保留 main() 校验框架
  10. 动态温度调节, 连续失败时降低随机性
  11. 向 LLM 注入近期 correctness 失败历史

[改进]:
  12. _prefilter_candidates: 在 tiny kernel 场景下预过滤明显有害的超参候选
  13. propose_hp.md 注入 kernel_regime 信息和安全约束
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from pathlib import Path
from typing import Any

from ...codegen.normalizer import extract_cuda_code
from ...models.data import BenchmarkResult, HyperparamCandidate, NcuMetrics
from ...tools.compile import compile_cuda
from ...tools.correctness import (
    check_correctness_multi,
    check_correctness_multi_async,
    summarize_correctness_results,
)
from ...tools.profile import format_ncu_for_prompt
from ..temperatures import TEMP_APPLY_METHOD, TEMP_PROPOSE_HP
from ._helpers import GpuPool, _compile_hp_candidates_async, _iter_compile_hp_candidates_async

logger = logging.getLogger(__name__)


# ════════════════════════════════════════
# [修复] 智能代码截断 —— 保留 main() 校验框架
# ════════════════════════════════════════

def _smart_truncate_code(code: str, max_chars: int = 8000) -> str:
    """
    智能截断: 保留 kernel 函数和 main 函数, 截断中间辅助函数。

    原来的 best_code[:8000] 可能丢失位于末尾的 main() 函数,
    导致 LLM 看不到 --check / --shape 参数解析和 JSON 输出格式,
    从而生成格式不兼容的代码。
    """
    if len(code) <= max_chars:
        return code

    # 找到 main 函数的起始位置
    main_start = -1
    for marker in ("int main(", "int main ("):
        pos = code.find(marker)
        if pos >= 0:
            main_start = pos
            break

    if main_start < 0:
        half = max_chars // 2
        return (
            code[:half]
            + "\n\n// ... [中间代码省略, 请保持原有的 main 函数、"
            "命令行参数解析和 JSON 输出格式不变] ...\n\n"
            + code[-half:]
        )

    main_section = code[main_start:]
    remaining_for_kernel = max_chars - len(main_section) - 100

    if remaining_for_kernel > 500:
        return (
            code[:remaining_for_kernel]
            + "\n\n// ... [辅助函数省略] ...\n\n"
            + main_section
        )
    else:
        kernel_budget = max(500, max_chars // 3)
        main_budget = max_chars - kernel_budget - 100
        return (
            code[:kernel_budget]
            + "\n\n// ... [中间代码省略] ...\n\n"
            + main_section[:main_budget]
            + ("\n// ... [main 函数末尾省略] ..." if len(main_section) > main_budget else "")
        )


# ════════════════════════════════════════
# [修复] 构建 correctness 失败历史上下文
# ════════════════════════════════════════

def _build_correctness_failure_history(run_state, limit: int = 5) -> str:
    """
    从最近的迭代中提取 correctness 失败的记录,
    让 LLM 在生成代码时知道之前哪些尝试因为正确性问题失败了。
    """
    lines = []
    for it in reversed(run_state.iterations[-20:]):
        if not it.correctness_ok and it.method_name:
            hp_text = json.dumps(it.hyperparams, ensure_ascii=False) if it.hyperparams else "none"
            lines.append(
                f"  - {it.version_id} ({it.method_name}, hp={hp_text}): "
                f"correctness 失败"
            )
            if len(lines) >= limit:
                break
    if not lines:
        return "(无近期 correctness 失败记录)"
    return "\n".join(lines)


# ════════════════════════════════════════
# [修复] 动态温度: 连续失败时降低随机性
# ════════════════════════════════════════

def _effective_apply_temperature(run_state) -> float:
    """
    根据连续 correctness 失败次数动态调整代码生成温度。
    连续失败越多, 温度越低, 生成更确定性的代码。
    """
    consecutive_fails = _count_consecutive_correctness_failures(run_state)
    if consecutive_fails >= 3:
        return max(0.05, TEMP_APPLY_METHOD - 0.03 * consecutive_fails)
    elif consecutive_fails >= 1:
        return max(0.08, TEMP_APPLY_METHOD - 0.02 * consecutive_fails)
    return TEMP_APPLY_METHOD


def _count_consecutive_correctness_failures(run_state) -> int:
    """统计尾部连续的 correctness 失败次数。"""
    count = 0
    for it in reversed(run_state.iterations):
        if it.version_id == "v0":
            break
        if not it.correctness_ok:
            count += 1
        else:
            break
    return count


# ════════════════════════════════════════
# [改进] 预过滤明显有害的超参候选
# ════════════════════════════════════════

def _prefilter_candidates(
    candidates: list[HyperparamCandidate],
    baseline_lat_ms: float,
    run_state: Any,
) -> list[HyperparamCandidate]:
    """
    [改进] 在 tiny kernel 场景下预过滤明显会导致回归的超参候选。

    当 baseline latency < 0.01 ms 时, 以下候选会被拦截:
    - blocks_per_channel > 1 (multi-CTA per output, 必须跨 CTA 合并)
    - threads_per_block < 64 (线程太少, 无法隐藏延迟)
    - 需要多个 kernel launch 的配置
    - 自声明 predicted_regression_risk = "high" 的候选

    返回过滤后的列表; 至少保留 1 个候选 (保底)。
    """
    if baseline_lat_ms >= 0.01:
        return candidates

    logger.info(
        "Prefiltering %d candidates for tiny kernel (baseline=%.4f ms)",
        len(candidates), baseline_lat_ms,
    )

    filtered = []
    dropped = []
    for c in candidates:
        hp = c.hyperparams or {}
        reason = None

        # 拦截已知的小 kernel 反模式
        if hp.get("blocks_per_channel", 1) > 1:
            reason = "blocks_per_channel > 1 on tiny kernel"
        elif hp.get("threads_per_block", 256) < 64:
            reason = "threads_per_block < 64 on tiny kernel"
        elif hp.get("channels_per_block", 1) > 16 and hp.get("elements_per_thread", 4) <= 1:
            reason = "high channels_per_block with low elements_per_thread"
        elif hp.get("num_kernels", 1) > 1 or hp.get("kernel_count", 1) > 1:
            reason = "multi-kernel launch on tiny kernel"
        elif getattr(c, "predicted_regression_risk", "medium") == "high":
            reason = "self-declared high regression risk"

        if reason:
            dropped.append((c.index, reason))
            logger.info("Prefilter DROP candidate %d: %s", c.index, reason)
        else:
            filtered.append(c)

    if not filtered:
        # 至少保留一个: 取 risk 最低的
        logger.warning("Prefilter dropped all candidates, keeping the least risky one")
        candidates.sort(
            key=lambda c: {"low": 0, "medium": 1, "high": 2}.get(
                getattr(c, "predicted_regression_risk", "medium"), 1
            )
        )
        filtered = [candidates[0]]

    logger.info(
        "Prefilter result: %d/%d candidates kept, %d dropped",
        len(filtered), len(filtered) + len(dropped), len(dropped),
    )
    return filtered


async def hp_search_node(self, state: dict) -> dict:
    """
    LLM 提出多组超参候选, 并发生成+编译+测速。
    """
    logger.info("=== HP SEARCH (optimized + correctness fix + prefilter) ===")
    decision = state["method_decision"]
    hw = state["hardware_spec"]
    op = state["operator_spec"]
    ncu = state.get("current_ncu", NcuMetrics())
    run_state = state["run_state"]

    # ── [改进] kernel regime 信息用于 propose_hp ──
    kernel_regime = run_state.kernel_regime
    regime_info = json.dumps(kernel_regime, ensure_ascii=False) if kernel_regime else "(未评估)"

    # ── Phase 0: LLM 提出超参候选 ──
    prompt = self.llm.format_prompt(
        "propose_hp.md",
        operator_name=op.name,
        operator_context=self._operator_context(op),
        method_name=decision.method_name,
        method_rationale=decision.rationale,
        hyperparams_schema=json.dumps(decision.hyperparams_schema or {}, indent=2),
        known_hp_trials=self._method_history_text(run_state, decision.method_name),
        ncu_key_metrics=format_ncu_for_prompt(ncu)[:3000],
        hardware_summary=self._hardware_summary(hw),
        hp_count=self.sm.config.hp_candidate_count,
        kernel_regime_info=regime_info,
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

    if not candidates:
        logger.warning("No HP candidates produced by LLM")
        return _empty_result()

    # ── [改进] 预过滤明显有害的候选 ──
    best_lat = run_state.best_latency_ms() or 0.0
    candidates = _prefilter_candidates(candidates, best_lat, run_state)

    version_base = run_state.next_version_id(has_hp=True)

    # [优化] 初始化 GPU 池
    gpu_pool = GpuPool(self.sm.config.gpu_ids if self.sm.config.gpu_ids else None)
    logger.info("HP search using %d GPU(s): %s", gpu_pool.count, gpu_pool.gpu_ids)

    # [修复] 智能代码截断, 保留 main() 校验框架
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

    # [修复] 动态温度
    effective_temp = _effective_apply_temperature(run_state)
    if effective_temp != TEMP_APPLY_METHOD:
        logger.info(
            "Dynamic temperature: %.3f (base=%.3f, consecutive_corr_fails=%d)",
            effective_temp, TEMP_APPLY_METHOD,
            _count_consecutive_correctness_failures(run_state),
        )

    # ── Phase 1: [优化] 并发 LLM 代码生成 ──
    llm_concurrency = max(1, self.sm.config.hp_llm_concurrency)
    semaphore = asyncio.Semaphore(llm_concurrency)
    logger.info("Generating %d candidate codes with concurrency=%d", len(candidates), llm_concurrency)

    async def _generate_code(cand: HyperparamCandidate) -> tuple[HyperparamCandidate, str, Path, Path]:
        """生成单个候选的代码 (受 semaphore 限制)。"""
        async with semaphore:
            version_id = f"{version_base}_cand{cand.index}"
            iter_dir = self.sm.create_iteration_dir(version_id)

            hp_section = f"- Hyperparams: {json.dumps(cand.hyperparams)}\n- Rationale: {cand.rationale}"

            # [修复] 注入 correctness 失败历史
            apply_prompt = self.llm.format_prompt(
                "apply_method.md",
                operator_name=op.name,
                operator_context=self._operator_context(op),
                method_name=decision.method_name,
                method_rationale=decision.rationale,
                hyperparams_section=hp_section,
                hardware_summary=self._hardware_summary(hw),
                best_id=run_state.current_best_id,
                best_code=code_for_prompt,
                ncu_key_metrics=format_ncu_for_prompt(ncu)[:2000],
                correctness_failure_history=correctness_history,
            )

            response = await self.llm.ainvoke(
                apply_prompt,
                temperature=effective_temp,
                node_name=f"hp_search:cand{cand.index}",
            )
            code = extract_cuda_code(response)
            code_path = await asyncio.to_thread(
                self.sm.persistence.save_code, code, iter_dir
            )
            return cand, code, code_path, iter_dir

    # 并发生成所有候选代码
    gen_tasks = [_generate_code(cand) for cand in candidates]
    gen_results = await asyncio.gather(*gen_tasks, return_exceptions=True)

    # 收集成功生成的候选
    candidate_records: dict[str, dict[str, Any]] = {}
    compile_jobs: list[dict[str, Any]] = []

    for i, result in enumerate(gen_results):
        if isinstance(result, Exception):
            logger.warning("Code generation for candidate %d failed: %s", i, result)
            continue
        cand, code, code_path, iter_dir = result
        version_id = f"{version_base}_cand{cand.index}"

        candidate_records[version_id] = {
            "candidate": cand,
            "code": code,
            "code_path": code_path,
            "iter_dir": iter_dir,
        }
        compile_jobs.append({
            "index": cand.index,
            "version_id": version_id,
            "iter_dir": str(iter_dir),
            "code_path": str(code_path),
            "output_path": str(iter_dir / "kernel"),
            "compute_capability": hw.compute_capability,
            "nvcc_threads": self.sm.config.nvcc_parallel_threads,
            "gpu_id": gpu_pool.assign_gpu(cand.index),
        })

    if not compile_jobs:
        logger.warning("All code generation failed")
        return _empty_result()

    # ── Phase 2-4: [优化] 编译完成即进入校验+benchmark 流水线 ──
    worker_count = self._hp_compile_worker_count(len(compile_jobs))
    dtype = list(op.dtypes.values())[0] if op.dtypes else "fp32"
    results: list[dict[str, Any]] = []
    max_correctness_parallel = self.sm.config.correctness_max_parallel

    # [修复] 收集所有候选的 correctness 失败详情
    correctness_failure_details: list[dict[str, Any]] = []

    # [修复] 获取 correctness 修复次数上限
    hp_correctness_repair_max = getattr(
        self.sm.config, "hp_correctness_repair_max", 2
    )

    async def _validate_and_benchmark(compiled: dict) -> dict[str, Any] | None:
        """单个候选的校验+benchmark流水线。"""
        version_id = compiled["version_id"]
        if version_id not in candidate_records:
            return None

        record = candidate_records[version_id]
        cand = record["candidate"]
        iter_dir = Path(record["iter_dir"])
        assigned_gpu = gpu_pool.assign_gpu(cand.index)

        # 写编译日志
        compile_output = (compiled.get("stdout", "") + "\n" + compiled.get("stderr", "")).strip()
        await asyncio.to_thread(
            (iter_dir / "compile.log").write_text, compile_output, encoding="utf-8"
        )

        current_code = record["code"]

        if not compiled.get("success"):
            compile_log_entries = [f"[Initial compile]\n{compile_output}"]
            compile_errors = [
                f"[HP candidate {cand.index} compile attempt 1]: "
                f"{compile_output[:2000]}"
            ]
            logger.warning("Candidate %d compilation failed: %s",
                           cand.index, compiled.get("stderr", "")[:500])

            exe_path: Path | None = None
            max_compile_repairs = max(
                0,
                getattr(self.sm.config, "compile_repair_max_retries", 0),
            )
            for repair_attempt in range(max_compile_repairs):
                logger.info(
                    "Attempting compile repair for candidate %d (%d/%d)",
                    cand.index, repair_attempt + 1, max_compile_repairs,
                )
                try:
                    current_code = await self._repair_code(
                        current_code, compile_errors, hw
                    )
                except Exception as e:
                    msg = f"Repair code generation failed: {e}"
                    compile_log_entries.append(
                        f"[Compile repair {repair_attempt + 1}]\n{msg}"
                    )
                    await asyncio.to_thread(
                        (iter_dir / "compile.log").write_text,
                        "\n\n".join(compile_log_entries),
                        encoding="utf-8",
                    )
                    logger.warning(
                        "Compile repair generation failed for candidate %d: %s",
                        cand.index, e,
                    )
                    break

                repair_code_path = await asyncio.to_thread(
                    self.sm.persistence.save_code,
                    current_code,
                    iter_dir,
                    f"code_compile_fix{repair_attempt + 1}.cu",
                )
                try:
                    repair_cr = await asyncio.to_thread(
                        compile_cuda,
                        repair_code_path,
                        iter_dir / f"kernel_compile_fix{repair_attempt + 1}",
                        hw.compute_capability,
                        nvcc_threads=self.sm.config.nvcc_parallel_threads,
                    )
                except TypeError as e:
                    if "unexpected keyword" not in str(e):
                        raise
                    repair_cr = await asyncio.to_thread(
                        compile_cuda,
                        repair_code_path,
                        iter_dir / f"kernel_compile_fix{repair_attempt + 1}",
                        hw.compute_capability,
                    )

                repair_output = (repair_cr.stdout + "\n" + repair_cr.stderr).strip()
                compile_log_entries.append(
                    f"[Compile repair {repair_attempt + 1}]\n{repair_output}"
                )
                await asyncio.to_thread(
                    (iter_dir / "compile.log").write_text,
                    "\n\n".join(compile_log_entries),
                    encoding="utf-8",
                )

                if repair_cr.success:
                    exe_path = Path(repair_cr.output_path)
                    record["code"] = current_code
                    logger.info(
                        "Candidate %d compilation PASSED after %d repair(s)",
                        cand.index, repair_attempt + 1,
                    )
                    break

                compile_errors.append(
                    f"[HP candidate {cand.index} compile repair "
                    f"{repair_attempt + 1} failed]: {repair_output[:2000]}"
                )

            if exe_path is None:
                return None
        else:
            exe_path = Path(compiled["output_path"])

        # ── [修复] correctness 修复循环 ──
        accumulated_errors: list[str] = []

        for repair_attempt in range(hp_correctness_repair_max + 1):
            shape_profiles = self._active_shape_profiles(op)
            if len(shape_profiles) <= 1 or max_correctness_parallel <= 1:
                try:
                    correctness_results = await asyncio.to_thread(
                        check_correctness_multi,
                        exe_path,
                        shape_profiles,
                        dtype=dtype,
                        gpu_id=assigned_gpu,
                    )
                except TypeError as e:
                    if "unexpected keyword" not in str(e):
                        raise
                    correctness_results = await asyncio.to_thread(
                        check_correctness_multi,
                        exe_path,
                        shape_profiles,
                        dtype=dtype,
                    )
            else:
                correctness_results = await check_correctness_multi_async(
                    exe_path,
                    shape_profiles,
                    dtype=dtype,
                    gpu_id=assigned_gpu,
                    max_parallel=max_correctness_parallel,
                )

            # ── 校验通过, 跳出修复循环 ──
            if all(r.get("correct") for r in correctness_results):
                if repair_attempt > 0:
                    logger.info(
                        "Candidate %d correctness PASSED after %d repair(s)",
                        cand.index, repair_attempt,
                    )
                break

            # ── [修复] 收集详细的 correctness 错误信息 ──
            error_details = []
            for r in correctness_results:
                if not r.get("correct"):
                    error_details.append(
                        f"shape={r.get('shape_label', '?')}: "
                        f"max_abs_error={r.get('max_abs_error', '?')}, "
                        f"max_rel_error={r.get('max_rel_error', '?')}, "
                        f"atol={r.get('atol_used', '?')}, rtol={r.get('rtol_used', '?')}, "
                        f"msg={r.get('message', '?')}"
                    )
            error_summary = "; ".join(error_details)
            accumulated_errors.append(
                f"[Correctness repair attempt {repair_attempt + 1}]: {error_summary}"
            )

            logger.warning(
                "Candidate %d correctness failed (attempt %d/%d): %s",
                cand.index, repair_attempt + 1, hp_correctness_repair_max + 1,
                error_summary[:300],
            )

            # ── 如果还有修复次数, 尝试修复 ──
            if repair_attempt < hp_correctness_repair_max:
                logger.info(
                    "Attempting correctness repair for candidate %d (%d/%d)",
                    cand.index, repair_attempt + 1, hp_correctness_repair_max,
                )
                try:
                    current_code = await self._repair_code(
                        current_code, accumulated_errors, hw
                    )
                except Exception as e:
                    logger.warning(
                        "Repair code generation failed for candidate %d: %s",
                        cand.index, e,
                    )
                    break

                # 保存修复后的代码
                fix_filename = f"code_fix{repair_attempt + 1}.cu"
                new_code_path = await asyncio.to_thread(
                    self.sm.persistence.save_code,
                    current_code,
                    iter_dir,
                    fix_filename,
                )

                # 重新编译
                try:
                    new_cr = await asyncio.to_thread(
                        compile_cuda,
                        new_code_path,
                        iter_dir / f"kernel_fix{repair_attempt + 1}",
                        hw.compute_capability,
                        nvcc_threads=self.sm.config.nvcc_parallel_threads,
                    )
                except TypeError as e:
                    if "unexpected keyword" not in str(e):
                        raise
                    new_cr = await asyncio.to_thread(
                        compile_cuda,
                        new_code_path,
                        iter_dir / f"kernel_fix{repair_attempt + 1}",
                        hw.compute_capability,
                    )

                if not new_cr.success:
                    logger.warning(
                        "Repair recompilation failed for candidate %d: %s",
                        cand.index, new_cr.stderr[:300],
                    )
                    accumulated_errors.append(
                        f"[Compile after repair {repair_attempt + 1}]: {new_cr.stderr[:500]}"
                    )
                    continue

                exe_path = Path(new_cr.output_path)
                record["code"] = current_code
            else:
                # 修复次数用尽, 收集失败详情
                correctness_failure_details.append({
                    "candidate_index": cand.index,
                    "hyperparams": cand.hyperparams,
                    "errors": [
                        {
                            "shape": r.get("shape_label", "?"),
                            "max_abs_error": r.get("max_abs_error"),
                            "max_rel_error": r.get("max_rel_error"),
                            "message": r.get("message", ""),
                        }
                        for r in correctness_results if not r.get("correct")
                    ],
                    "repair_attempts": hp_correctness_repair_max,
                    "accumulated_errors": accumulated_errors,
                })
                logger.warning(
                    "Candidate %d correctness failed after %d repair attempts",
                    cand.index, hp_correctness_repair_max,
                )
                return None
        else:
            correctness_failure_details.append({
                "candidate_index": cand.index,
                "hyperparams": cand.hyperparams,
                "errors": [],
                "repair_attempts": hp_correctness_repair_max,
                "accumulated_errors": accumulated_errors,
            })
            return None

        # 如果候选经历过修复, 将最终文件落回标准路径
        await asyncio.to_thread(
            self.sm.persistence.save_code,
            current_code,
            iter_dir,
        )
        canonical_exe = iter_dir / "kernel"
        if exe_path.resolve() != canonical_exe.resolve():
            await asyncio.to_thread(shutil.copy2, exe_path, canonical_exe)
        exe_path = canonical_exe

        # [优化] 多 GPU benchmark
        gpu_sem = gpu_pool.get_semaphore(assigned_gpu)
        async with gpu_sem:
            try:
                bm = await asyncio.to_thread(
                    self._benchmark_multi, exe_path, op, assigned_gpu
                )
            except TypeError as e:
                if "positional" not in str(e) and "argument" not in str(e):
                    raise
                bm = await asyncio.to_thread(self._benchmark_multi, exe_path, op)

        return {
            "index": cand.index,
            "version_id": version_id,
            "hyperparams": cand.hyperparams,
            "benchmark": bm,
            "code": record["code"],
            "iter_dir": str(iter_dir),
        }

    validate_tasks: list[asyncio.Task] = []
    if worker_count <= 1:
        compiled_candidates = await _compile_hp_candidates_async(compile_jobs, worker_count)
        for compiled in compiled_candidates:
            r = await _validate_and_benchmark(compiled)
            if r is not None:
                results.append(r)
    else:
        async for compiled in _iter_compile_hp_candidates_async(compile_jobs, worker_count):
            validate_tasks.append(asyncio.create_task(_validate_and_benchmark(compiled)))

    if validate_tasks:
        for task in asyncio.as_completed(validate_tasks):
            try:
                r = await task
            except Exception as e:
                logger.warning("Validate/benchmark pipeline error: %s", e)
                continue
            if r is not None:
                results.append(r)

    if not results:
        return _empty_result_with_details(correctness_failure_details)

    best_cand = min(results, key=lambda r: r["benchmark"].latency_ms_median)

    return {
        "new_code": best_cand["code"],
        "new_version_id": best_cand["version_id"],
        "trial_version_id": best_cand["version_id"],
        "trial_benchmark": best_cand["benchmark"],
        "trial_compile_ok": True,
        "trial_correctness_ok": True,
        "hp_candidates": results,
        "hp_correctness_failures": correctness_failure_details,
        "hp_all_compiled_ok": True,
    }


def _empty_result() -> dict:
    """返回空结果 (所有候选失败时使用, 向后兼容)。"""
    return {
        "new_code": "",
        "new_version_id": "",
        "trial_version_id": "",
        "trial_benchmark": None,
        "trial_compile_ok": False,
        "trial_correctness_ok": False,
        "trial_accepted": False,
        "hp_candidates": [],
        "hp_correctness_failures": [],
        "hp_all_compiled_ok": False,
    }


def _empty_result_with_details(
    correctness_failures: list[dict[str, Any]] | None = None,
) -> dict:
    """[修复] 返回包含 correctness 失败详情的空结果。"""
    has_compiled_candidates = bool(correctness_failures)
    return {
        "new_code": "",
        "new_version_id": "",
        "trial_version_id": "",
        "trial_benchmark": None,
        "trial_compile_ok": False,
        "trial_correctness_ok": False,
        "trial_accepted": False,
        "hp_candidates": [],
        "hp_correctness_failures": correctness_failures or [],
        "hp_all_compiled_ok": has_compiled_candidates,
    }
