"""
LangGraph 节点实现 —— 每个节点对应技术总纲 §3.1 的一个职责单元。
"""

from __future__ import annotations

import os
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from ..codegen.normalizer import extract_cuda_code
from ..codegen.verifier import generate_diff, verify_code_structure
from ..memory.knowledge import KnowledgeBase
from ..memory.run_state import RunStateManager
from ..models.data import (
    BenchmarkResult,
    BlacklistEntry,
    HyperparamCandidate,
    IterationRecord,
    MethodDecision,
    NcuMetrics,
)
from ..models.enums import normalize_method_name
from ..shape_profiles import shape_profile_to_args
from ..tools.benchmark import run_benchmark_multi
from ..tools.compile import compile_cuda
from ..tools.correctness import check_correctness_multi, summarize_correctness_results
from ..tools.hardware import collect_hardware_info
from ..tools.profile import format_ncu_for_prompt, run_ncu_profile
from .llm_client import LLMClient
from .state import GraphState

logger = logging.getLogger(__name__)


def _compile_hp_candidate_job(job: dict[str, Any]) -> dict[str, Any]:
    """Compile one HP candidate in a worker process."""
    result = {
        "index": job["index"],
        "version_id": job["version_id"],
        "iter_dir": job["iter_dir"],
        "code_path": job["code_path"],
        "success": False,
        "output_path": "",
        "stdout": "",
        "stderr": "",
        "return_code": -1,
    }
    try:
        cr = compile_cuda(
            job["code_path"],
            job["output_path"],
            job["compute_capability"],
        )
        result.update({
            "success": cr.success,
            "output_path": cr.output_path,
            "stdout": cr.stdout,
            "stderr": cr.stderr,
            "return_code": cr.return_code,
        })
    except Exception as e:
        result["stderr"] = f"Compilation worker error: {e}"
    return result


class AgentNodes:
    """
    所有 LangGraph 节点的集合。
    每个方法接受 GraphState,返回更新后的 partial dict。
    """

    def __init__(
        self,
        state_manager: RunStateManager,
        kb: KnowledgeBase,
        llm: LLMClient,
    ):
        self.sm = state_manager
        self.kb = kb
        self.llm = llm

    @staticmethod
    def _operator_context(op) -> str:
        """Format task semantics once so every optimization prompt sees the same context."""
        lines = [
            f"- Signature: {op.signature or '(none)'}",
            f"- Dtypes: {json.dumps(op.dtypes, ensure_ascii=False)}",
            f"- Shapes: {json.dumps(op.shapes, ensure_ascii=False)}",
        ]
        if op.shape_profiles:
            lines.append(f"- Shape profiles: {json.dumps(op.shape_profiles, ensure_ascii=False)}")
        if op.task_description:
            lines.append(f"- Task description: {op.task_description}")
        if op.constraints:
            lines.append("- Constraints:\n  " + "\n  ".join(op.constraints))
        if op.seed_code_path:
            lines.append(f"- Seed code path: {op.seed_code_path}")
        return "\n".join(lines)

    @staticmethod
    def _read_seed_code(seed_code_path: str) -> str:
        code = Path(seed_code_path).read_text(encoding="utf-8", errors="replace")
        max_chars = 60000
        if len(code) > max_chars:
            return code[:max_chars] + "\n/* ... seed code truncated for prompt length ... */"
        return code

    @staticmethod
    def _active_shape_profiles(op) -> list[dict]:
        if op.shape_profiles:
            return op.shape_profiles
        if op.shapes:
            return [op.shapes]
        return [{}]

    def _benchmark_multi(self, exe_path: Path, op) -> BenchmarkResult:
        return run_benchmark_multi(
            exe_path,
            self._active_shape_profiles(op),
            warmup_rounds=self.sm.config.benchmark_warmup_rounds,
            measure_rounds=self.sm.config.benchmark_measure_rounds,
            aggregator=self.sm.config.multi_shape_aggregator,
        )

    @staticmethod
    def _profile_args_from_benchmark(bm: BenchmarkResult) -> list[str]:
        shape = bm.extra.get("worst_shape") if bm and bm.extra else None
        args = shape_profile_to_args(shape or {})
        args.extend(["--warmup", "0", "--rounds", "1"])
        return args

    @staticmethod
    def _per_shape_summary(bm: BenchmarkResult | None, limit: int = 3) -> str:
        if not bm or not bm.extra.get("per_shape"):
            return ""
        parts = []
        for item in bm.extra["per_shape"][:limit]:
            parts.append(f"{item.get('shape_label', 'shape')}={item.get('latency_ms_median', 0.0):.4f}ms")
        if len(bm.extra["per_shape"]) > limit:
            parts.append(f"+{len(bm.extra['per_shape']) - limit} more")
        return "<br>".join(parts)

    def _hp_compile_worker_count(self, job_count: int) -> int:
        if job_count <= 1:
            return 1
        configured = self.sm.config.hp_compile_workers
        if configured == 1:
            return 1
        if configured and configured > 1:
            return min(job_count, configured)
        return min(job_count, os.cpu_count() or 1)

    def _compile_hp_candidates(self, jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        worker_count = self._hp_compile_worker_count(len(jobs))
        if worker_count <= 1:
            return [_compile_hp_candidate_job(job) for job in jobs]

        logger.info("Compiling %d HP candidates with %d workers", len(jobs), worker_count)
        try:
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                return list(executor.map(_compile_hp_candidate_job, jobs))
        except Exception as e:
            logger.warning("Parallel HP compilation failed; falling back to serial compile: %s", e)
            return [_compile_hp_candidate_job(job) for job in jobs]

    # ════════════════════════════════════════
    # INIT
    # ════════════════════════════════════════
    def init_node(self, state: GraphState) -> dict:
        """初始化节点:采集硬件信息。"""
        logger.info("=== INIT: collecting hardware info ===")
        hw = collect_hardware_info()
        logger.info("GPU: %s (%s), CUDA: %s", hw.gpu_name, hw.compute_capability, hw.cuda_version)
        return {"hardware_spec": hw}

    # ════════════════════════════════════════
    # BOOTSTRAP (生成 v0)
    # ════════════════════════════════════════
    def bootstrap_node(self, state: GraphState) -> dict:
        """LLM 生成 v0 baseline。"""
        logger.info("=== BOOTSTRAP: generating v0 baseline ===")
        op = state["operator_spec"]
        hw = state["hardware_spec"]

        # 查询 KB 软提示
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

        response = self.llm.invoke(prompt)
        code = extract_cuda_code(response)

        return {"current_code": code, "new_version_id": "v0"}

    # ════════════════════════════════════════
    # COMPILE & REPAIR LOOP
    # ════════════════════════════════════════
    def compile_and_validate_node(self, state: GraphState) -> dict:
        """编译 + 数值校验。失败时触发修复循环。"""
        logger.info("=== COMPILE & VALIDATE ===")
        code = state.get("new_code") or state.get("current_code", "")
        version_id = state.get("new_version_id", "v0")
        hw = state["hardware_spec"]
        op = state["operator_spec"]
        max_retries = self.sm.config.compile_repair_max_retries

        # 创建迭代目录
        iter_dir = self.sm.create_iteration_dir(version_id)
        code_path = self.sm.persistence.save_code(code, iter_dir)

        compile_ok = False
        correctness_ok = False

        for attempt in range(max_retries + 1):
            # 编译
            cr = compile_cuda(
                code_path,
                output_path=iter_dir / "kernel",
                compute_capability=hw.compute_capability,
            )

            if cr.success:
                compile_ok = True
                exe_path = Path(cr.output_path)
                (iter_dir / "compile.log").write_text(cr.stdout + "\n" + cr.stderr, encoding="utf-8")

                # 数值校验
                dtype = list(op.dtypes.values())[0] if op.dtypes else "fp32"
                correctness_results = check_correctness_multi(
                    exe_path,
                    self._active_shape_profiles(op),
                    dtype=dtype,
                )
                if all(r.get("correct") for r in correctness_results):
                    correctness_ok = True
                    break
                else:
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
        response = self.llm.invoke(prompt)
        return extract_cuda_code(response)

    # ════════════════════════════════════════
    # PROFILE BEST
    # ════════════════════════════════════════
    def profile_best_node(self, state: GraphState) -> dict:
        """对当前 best 进行 benchmark + ncu profiling。"""
        logger.info("=== PROFILE BEST ===")
        run_state = state["run_state"]
        best = run_state.iter_by_id(run_state.current_best_id)

        if best is None:
            return {"error": "Best version not found"}

        best_dir = self.sm.run_dir / f"iter{best.version_id}"
        exe_path = self._kernel_executable(best_dir)

        if not exe_path.exists():
            raise FileNotFoundError(f"Best executable not found: {exe_path}")

        # Benchmark
        bm = self._benchmark_multi(exe_path, run_state.operator_spec)

        # ncu Profile
        ncu = run_ncu_profile(
            exe_path,
            output_report_path=best_dir / "ncu_report.txt",
            executable_args=self._profile_args_from_benchmark(bm),
        )

        # 读取 best 代码
        code_path = self.sm.run_dir / best.code_path if best.code_path else best_dir / "code.cu"
        code = code_path.read_text(encoding="utf-8") if code_path.exists() else ""

        best.benchmark = bm
        best.ncu_metrics = ncu
        best.ncu_report_path = str((best_dir / "ncu_report.txt").relative_to(self.sm.run_dir))
        self.sm._save()

        return {
            "current_benchmark": bm,
            "current_ncu": ncu,
            "current_code": code,
            "run_state": self.sm.state,
        }

    # ════════════════════════════════════════
    # ANALYZE
    # ════════════════════════════════════════
    def analyze_node(self, state: GraphState) -> dict:
        """LLM 分析瓶颈。"""
        logger.info("=== ANALYZE ===")
        run_state = state["run_state"]
        op = state["operator_spec"]
        hw = state["hardware_spec"]

        # 构建迭代历史摘要(不塞全量代码)
        history_lines = []
        for it in run_state.iterations:
            bm_str = f"{it.benchmark.latency_ms_median:.4f}ms" if it.benchmark else "N/A"
            status = "accepted" if it.accepted else "rejected"
            history_lines.append(
                f"  {it.version_id}: {it.method_name or 'baseline'} -> {bm_str} {status}"
            )
        history_text = "\n".join(history_lines) or "(no history)"

        # KB 软提示
        kb_hints = self.kb.query(op.name, hw.signature)
        hints_text = self.kb.format_hints_for_prompt(kb_hints)

        # ncu 报告
        ncu = state.get("current_ncu", NcuMetrics())
        ncu_text = format_ncu_for_prompt(ncu)

        # benchmark
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

        analysis = self.llm.invoke_json(prompt)
        return {"analysis_result": analysis}

    # ════════════════════════════════════════
    # DECIDE
    # ════════════════════════════════════════
    def decide_node(self, state: GraphState) -> dict:
        """LLM 决策下一个优化方法。"""
        logger.info("=== DECIDE ===")
        run_state = state["run_state"]
        op = state["operator_spec"]
        hw = state["hardware_spec"]

        analysis = state.get("analysis_result", {})
        bm = state.get("current_benchmark", BenchmarkResult())

        # 黑名单格式化
        bl_lines = []
        for entry in run_state.blacklist:
            hp_str = json.dumps(entry.hyperparam_constraint) if entry.hyperparam_constraint else "no hyperparams"
            bl_lines.append(f"  - {entry.method_name_normalized} ({hp_str}): {entry.reason}")
        blacklist_text = "\n".join(bl_lines) or "(none)"

        # KB 软提示
        kb_hints = self.kb.query(op.name, hw.signature)
        hints_text = self.kb.format_hints_for_prompt(kb_hints)

        bm_text = (
            f"latency_median: {bm.latency_ms_median:.4f} ms\n"
            f"latency_p95: {bm.latency_ms_p95:.4f} ms\n"
            f"aggregator: {bm.extra.get('aggregator', 'single')}\n"
            f"per_shape: {self._per_shape_summary(bm).replace('<br>', '; ') or 'N/A'}"
        )

        rejected_methods: list[tuple[str, str]] = []

        def rejected_methods_text() -> str:
            if not rejected_methods:
                return "(none)"
            return "\n".join(
                f"  - {name} (normalized: {normalized})"
                for name, normalized in rejected_methods
            )

        def build_prompt() -> str:
            return self.llm.format_prompt(
                "decide_method.md",
                operator_name=op.name,
                operator_context=self._operator_context(op),
                best_id=run_state.current_best_id,
                benchmark_metrics=bm_text,
                analysis_summary=json.dumps(analysis, ensure_ascii=False, indent=2),
                blacklist=blacklist_text,
                rejected_methods=rejected_methods_text(),
                kb_hints=hints_text,
                hardware_summary=self._hardware_summary(hw),
            )

        decision: MethodDecision | None = None
        max_reselects = max(0, self.sm.config.decide_reselect_max_retries)
        for attempt in range(max_reselects + 1):
            decision_data = self.llm.invoke_json(build_prompt())
            decision = MethodDecision.model_validate(decision_data)

            if decision.give_up:
                break

            if not run_state.is_method_blacklisted(decision.method_name):
                break

            normalized = normalize_method_name(decision.method_name)
            rejected_methods.append((decision.method_name, normalized))
            remaining = max_reselects - attempt
            logger.warning(
                "LLM selected blacklisted method %s (attempt %d/%d); %d reselections remaining",
                decision.method_name,
                attempt + 1,
                max_reselects + 1,
                remaining,
            )
            if remaining <= 0:
                decision.give_up = True
                decision.rationale += (
                    " [framework exhausted decide reselection retries after repeated blacklisted methods: "
                    + ", ".join(name for name, _ in rejected_methods)
                    + "]"
                )
                break

        assert decision is not None

        if decision.give_up:
            return {
                "method_decision": decision,
                "should_stop": True,
                "stop_reason": f"LLM gave up: {decision.rationale}",
            }

        return {
            "method_decision": decision,
            "has_hyperparams": decision.has_hyperparams,
        }

    # ════════════════════════════════════════
    # HP_SEARCH
    # ════════════════════════════════════════
    def hp_search_node(self, state: GraphState) -> dict:
        """LLM 提出多组超参候选,逐一编译+测速。"""
        logger.info("=== HP SEARCH ===")
        decision = state["method_decision"]
        hw = state["hardware_spec"]
        op = state["operator_spec"]
        ncu = state.get("current_ncu", NcuMetrics())

        # 提出候选
        prompt = self.llm.format_prompt(
            "propose_hp.md",
            operator_name=op.name,
            operator_context=self._operator_context(op),
            method_name=decision.method_name,
            method_rationale=decision.rationale,
            hyperparams_schema=json.dumps(decision.hyperparams_schema or {}, indent=2),
            ncu_key_metrics=format_ncu_for_prompt(ncu)[:3000],
            hardware_summary=self._hardware_summary(hw),
            hp_count=self.sm.config.hp_candidate_count,
        )

        candidates_raw = self.llm.invoke_json(prompt)
        if isinstance(candidates_raw, dict):
            candidates_raw = candidates_raw.get("candidates", [candidates_raw])

        candidates = []
        for item in candidates_raw:
            if isinstance(item, dict):
                candidates.append(HyperparamCandidate.model_validate(item))

        # 先串行生成代码并落盘,再并行编译;GPU 校验和测速必须保持串行。
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

            response = self.llm.invoke(apply_prompt)
            code = extract_cuda_code(response)
            code_path = self.sm.persistence.save_code(code, iter_dir)

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
        compiled_candidates = self._compile_hp_candidates(compile_jobs)

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

            # 校验
            dtype = list(op.dtypes.values())[0] if op.dtypes else "fp32"
            correctness_results = check_correctness_multi(exe_path, self._active_shape_profiles(op), dtype=dtype)
            if not all(r.get("correct") for r in correctness_results):
                logger.warning("Candidate %d correctness check failed: %s", cand.index, summarize_correctness_results(correctness_results))
                continue

            # 测速
            bm = self._benchmark_multi(exe_path, op)
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

        # 选最快的
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

    # ════════════════════════════════════════
    # APPLY DIRECT (无超参)
    # ════════════════════════════════════════
    def apply_direct_node(self, state: GraphState) -> dict:
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

        response = self.llm.invoke(prompt)
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

    # ════════════════════════════════════════
    # EVALUATE
    # ════════════════════════════════════════
    def evaluate_node(self, state: GraphState) -> dict:
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
            }

        # 如果 hp_search 已经做过编译/测速，且结果属于当前版本，直接用结果。
        if state.get("trial_benchmark") and state.get("trial_version_id") == version_id:
            trial_bm = state["trial_benchmark"]
        else:
            # apply_direct 路径:还需要编译+校验+测速
            result = self.compile_and_validate_node(state)
            if not result.get("trial_compile_ok") or not result.get("trial_correctness_ok"):
                return {
                    "trial_version_id": version_id,
                    "trial_benchmark": None,
                    "trial_accepted": False,
                    "trial_compile_ok": result.get("trial_compile_ok", False),
                    "trial_correctness_ok": result.get("trial_correctness_ok", False),
                }

            iter_dir = self.sm.run_dir / f"iter{version_id}"
            exe_path = self._kernel_executable(iter_dir)
            trial_bm = self._benchmark_multi(exe_path, run_state.operator_spec)

        best_bm = state.get("current_benchmark", BenchmarkResult())

        # 接受准则: v★.latency < best.latency * (1 - ε)
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
        }

    # ════════════════════════════════════════
    # REFLECT
    # ════════════════════════════════════════
    def reflect_node(self, state: GraphState) -> dict:
        """LLM 反思:为什么有效/无效。"""
        logger.info("=== REFLECT ===")
        decision = state.get("method_decision", MethodDecision(method_name="unknown"))
        accepted = state.get("trial_accepted", False)
        run_state = state["run_state"]
        op = state["operator_spec"]
        hw = state["hardware_spec"]

        trial_bm = state.get("trial_benchmark") or BenchmarkResult()
        best_bm = state.get("current_benchmark") or BenchmarkResult()

        if accepted:
            speedup = (best_bm.latency_ms_median / trial_bm.latency_ms_median
                       if trial_bm.latency_ms_median > 0 else 1.0)

            prompt = self.llm.format_prompt(
                "reflect_success.md",
                method_name=decision.method_name,
                hyperparams=json.dumps(decision.hyperparams_schema) if decision.has_hyperparams else "none",
                parent_id=run_state.current_best_id,
                parent_latency_ms=best_bm.latency_ms_median,
                new_id=state.get("new_version_id", "?"),
                new_latency_ms=trial_bm.latency_ms_median,
                speedup=speedup,
                ncu_diff="(see ncu report for details)",
            )
        else:
            prompt = self.llm.format_prompt(
                "reflect_failure.md",
                method_name=decision.method_name,
                hyperparams=json.dumps(decision.hyperparams_schema) if decision.has_hyperparams else "none",
                best_id=run_state.current_best_id,
                best_latency_ms=best_bm.latency_ms_median,
                trial_id=state.get("new_version_id", "?"),
                trial_latency_ms=trial_bm.latency_ms_median if trial_bm.latency_ms_median > 0 else "N/A",
                failure_reason="no speedup" if state.get("trial_compile_ok") else "compile/correctness failed",
                ncu_report="(see ncu report)",
            )

        reflection = self.llm.invoke_json(prompt)

        # 根据结果更新状态
        version_id = state.get("new_version_id", "unknown")
        code_path = ""
        if self.sm.run_dir:
            candidate_code_path = self.sm.run_dir / f"iter{version_id}" / "code.cu"
            if candidate_code_path.exists():
                code_path = str(candidate_code_path.relative_to(self.sm.run_dir))
        record = IterationRecord(
            version_id=version_id,
            parent_id=run_state.current_best_id,
            method_name=decision.method_name,
            has_hyperparams=decision.has_hyperparams,
            code_path=code_path,
            benchmark=trial_bm if trial_bm.latency_ms_median > 0 else None,
            compile_ok=state.get("trial_compile_ok", False),
            correctness_ok=state.get("trial_correctness_ok", False),
            accepted=accepted,
        )

        self.sm.add_iteration(record)

        if accepted:
            iter_dir = self.sm.run_dir / f"iter{version_id}"
            self.sm.update_best(version_id, iter_dir)

            # 更新 KB
            kb_suggestion = reflection.get("kb_write_suggestion", {})
            if kb_suggestion.get("should_write", False):
                speedup = (best_bm.latency_ms_median / trial_bm.latency_ms_median
                           if trial_bm.latency_ms_median > 0 else 1.0)
                self.kb.write_entry(
                    operator_class=op.name,
                    hardware_signature=hw.signature,
                    method_name=decision.method_name,
                    run_id=run_state.run_id,
                    version_id=version_id,
                    speedup_vs_parent=speedup,
                    notes=kb_suggestion.get("notes", ""),
                )
        else:
            # 加入黑名单
            self.sm.add_to_blacklist(
                method_name=decision.method_name,
                reason=reflection.get("why_ineffective", "unknown"),
                failed_at_version=version_id,
            )

        # 保存推理日志
        if self.sm.run_dir:
            reasoning_text = (
                f"## {version_id} · {'ACCEPTED' if accepted else 'REJECTED'}\n\n"
                f"### Method: {decision.method_name}\n"
                f"### Rationale: {decision.rationale}\n\n"
                f"### Reflection\n```json\n{json.dumps(reflection, ensure_ascii=False, indent=2)}\n```\n"
            )
            self.sm.persistence.save_reasoning_log(reasoning_text, self.sm.run_dir)

        # 检查终止条件
        should_stop, stop_reason = self.sm.should_stop()

        return {
            "reflection": reflection,
            "should_stop": should_stop,
            "stop_reason": stop_reason,
            "run_state": self.sm.state,
        }

    # ════════════════════════════════════════
    # TERMINATE
    # ════════════════════════════════════════
    def terminate_node(self, state: GraphState) -> dict:
        """终止节点:生成最终报告。"""
        logger.info("=== TERMINATE ===")
        run_state = state["run_state"]
        self.sm.mark_done()

        # 生成最终报告
        if self.sm.run_dir:
            report = self._generate_final_report(run_state)
            (self.sm.run_dir / "final_report.md").write_text(report, encoding="utf-8")

        return {"should_stop": True, "stop_reason": state.get("stop_reason", "done")}

    # ════════════════════════════════════════
    # 辅助方法
    # ════════════════════════════════════════
    def _hardware_summary(self, hw) -> str:
        return (
            f"GPU: {hw.gpu_name}\n"
            f"Compute capability: {hw.compute_capability}\n"
            f"SM count: {hw.sm_count}\n"
            f"Shared memory/block: {hw.shared_mem_per_block_kb} KB\n"
            f"L2 Cache: {hw.l2_cache_mb} MB\n"
            f"Tensor Cores: {'yes' if hw.has_tensor_cores else 'no'}\n"
            f"CUDA version: {hw.cuda_version}"
        )

    @staticmethod
    def _kernel_executable(iter_dir: Path) -> Path:
        exe_path = iter_dir / "kernel"
        if exe_path.exists():
            return exe_path
        win_exe_path = exe_path.with_suffix(".exe")
        if win_exe_path.exists():
            return win_exe_path
        return exe_path

    def _generate_final_report(self, run_state) -> str:
        """生成运行结束时的总结报告。"""
        lines = [
            f"# CUDA 算子优化报告",
            f"",
            f"## 基本信息",
            f"- 运行 ID: {run_state.run_id}",
            f"- 算子: {run_state.operator_spec.name}",
            f"- 硬件: {run_state.hardware_spec.gpu_name} ({run_state.hardware_spec.compute_capability})",
            f"- 总迭代次数: {len(run_state.iterations)}",
            f"- 最终 best: {run_state.current_best_id}",
            f"",
            f"## 优化历程",
            "",
        ]

        v0 = run_state.iter_by_id("v0")
        best = run_state.iter_by_id(run_state.current_best_id)

        if v0 and v0.benchmark and best and best.benchmark:
            speedup = v0.benchmark.latency_ms_median / best.benchmark.latency_ms_median
            lines.append(f"| 指标 | v0 (baseline) | {best.version_id} (best) | 加速比 |")
            lines.append(f"|------|--------------|--------------|--------|")
            lines.append(
                f"| latency (ms) | {v0.benchmark.latency_ms_median:.4f} | "
                f"{best.benchmark.latency_ms_median:.4f} | {speedup:.2f}x |"
            )

        lines.append("")
        lines.append("## 各迭代详情")
        lines.append("")
        lines.append("| 版本 | 方法 | aggregate latency (ms) | per-shape latency | 状态 |")
        lines.append("|------|------|------------------------|-------------------|------|")
        for it in run_state.iterations:
            lat = f"{it.benchmark.latency_ms_median:.4f}" if it.benchmark else "N/A"
            per_shape = self._per_shape_summary(it.benchmark) or "-"
            status = "✓" if it.accepted else "✗"
            lines.append(f"| {it.version_id} | {it.method_name or 'baseline'} | {lat} | {per_shape} | {status} |")

        return "\n".join(lines)
