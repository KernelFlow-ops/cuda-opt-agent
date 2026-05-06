from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from ...models.data import BenchmarkResult, IterationRecord
from ...models.enums import normalize_method_name
from ...shape_profiles import shape_profile_to_args
from ...tools.benchmark import run_benchmark_multi
from ...tools.compile import compile_cuda

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


def _read_seed_code(seed_code_path: str) -> str:
    code = Path(seed_code_path).read_text(encoding="utf-8", errors="replace")
    max_chars = 60000
    if len(code) > max_chars:
        return code[:max_chars] + "\n/* ... seed code truncated for prompt length ... */"
    return code


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


def _profile_args_from_benchmark(bm: BenchmarkResult) -> list[str]:
    shape = bm.extra.get("worst_shape") if bm and bm.extra else None
    args = shape_profile_to_args(shape or {})
    args.extend(["--warmup", "0", "--rounds", "1"])
    return args


def _per_shape_summary(bm: BenchmarkResult | None, limit: int = 3) -> str:
    if not bm or not bm.extra.get("per_shape"):
        return ""
    parts = []
    for item in bm.extra["per_shape"][:limit]:
        parts.append(f"{item.get('shape_label', 'shape')}={item.get('latency_ms_median', 0.0):.4f}ms")
    if len(bm.extra["per_shape"]) > limit:
        parts.append(f"+{len(bm.extra['per_shape']) - limit} more")
    return "<br>".join(parts)


def _iteration_outcome_text(iteration: IterationRecord) -> str:
    if not iteration.compile_ok:
        return "compile failed"
    if not iteration.correctness_ok:
        return "failed correctness"
    if iteration.benchmark:
        return f"{iteration.benchmark.latency_ms_median:.4f} ms"
    return "no benchmark"


def _hyperparams_text(hyperparams: dict[str, Any] | None) -> str:
    if not hyperparams:
        return "none"
    return json.dumps(hyperparams, ensure_ascii=False, sort_keys=True)


def _method_history_text(self, run_state, method_name: str | None = None, limit: int = 20) -> str:
    target = normalize_method_name(method_name) if method_name else None
    rows = []
    for iteration in run_state.iterations:
        if not iteration.method_name:
            continue
        normalized = normalize_method_name(iteration.method_name)
        if target and normalized != target:
            continue
        rows.append(
            "| {version} | {method} | {hyperparams} | {outcome} | {accepted} |".format(
                version=iteration.version_id,
                method=iteration.method_name,
                hyperparams=self._hyperparams_text(iteration.hyperparams),
                outcome=self._iteration_outcome_text(iteration),
                accepted="yes" if iteration.accepted else "no",
            )
        )

    if not rows:
        return "(none)"
    rows = rows[-limit:]
    header = "| version | method | hyperparams | outcome | accepted |\n|---|---|---|---|---|"
    return header + "\n" + "\n".join(rows)


def _selected_hyperparams(state) -> dict[str, Any] | None:
    version_id = state.get("new_version_id") or state.get("trial_version_id")
    for item in state.get("hp_candidates", []) or []:
        if isinstance(item, dict) and item.get("version_id") == version_id:
            hyperparams = item.get("hyperparams")
            return hyperparams if isinstance(hyperparams, dict) else None
    return None


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
