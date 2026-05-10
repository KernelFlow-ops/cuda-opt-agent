"""
节点辅助函数。

[优化]:
  - _compile_hp_candidate_job: 支持 gpu_id 和 nvcc_threads
  - _compile_hp_candidates_async: 基于 asyncio + as_completed 的异步编译
  - _generate_code_diff / _build_code_diff_context: 代码 diff 工具
  - GpuPool: 多 GPU 资源池
"""

from __future__ import annotations

import asyncio
import difflib
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


# ════════════════════════════════════════
# [优化] 多 GPU 资源池
# ════════════════════════════════════════
class GpuPool:
    """简易 GPU 资源池, 用于 hp 候选多 GPU 分发。"""

    def __init__(self, gpu_ids: list[int] | None = None):
        if gpu_ids:
            self._ids = list(gpu_ids)
        else:
            # 自动检测
            self._ids = self._detect_gpus()
        self._semaphores: dict[int, asyncio.Semaphore] = {}

    def _detect_gpus(self) -> list[int]:
        """检测可用 GPU 数量。"""
        try:
            # Provided by the nvidia-ml-py package; import name remains pynvml.
            import pynvml
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            pynvml.nvmlShutdown()
            return list(range(count)) or [0]
        except Exception:
            return [0]

    @property
    def gpu_ids(self) -> list[int]:
        return self._ids

    @property
    def count(self) -> int:
        return len(self._ids)

    def get_semaphore(self, gpu_id: int) -> asyncio.Semaphore:
        """每个 GPU 一个 semaphore, 防止同时跑多个 benchmark。"""
        if gpu_id not in self._semaphores:
            self._semaphores[gpu_id] = asyncio.Semaphore(1)
        return self._semaphores[gpu_id]

    def assign_gpu(self, index: int) -> int:
        """根据候选索引分配 GPU (round-robin)。"""
        return self._ids[index % len(self._ids)]


# ════════════════════════════════════════
# HP 候选编译 (原始 + 优化)
# ════════════════════════════════════════

def _compile_hp_candidate_job(job: dict[str, Any]) -> dict[str, Any]:
    """
    Compile one HP candidate in a worker process.

    [优化] 支持 gpu_id 和 nvcc_threads 参数。
    """
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
        try:
            cr = compile_cuda(
                job["code_path"],
                job["output_path"],
                job["compute_capability"],
                nvcc_threads=job.get("nvcc_threads", 0),
                gpu_id=job.get("gpu_id"),
            )
        except TypeError as e:
            if "unexpected keyword" not in str(e):
                raise
            cr = compile_cuda(job["code_path"], job["output_path"], job["compute_capability"])
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


async def _compile_hp_candidates_async(
    jobs: list[dict[str, Any]],
    worker_count: int,
) -> list[dict[str, Any]]:
    """
    [优化] 基于 asyncio + ProcessPoolExecutor 的异步编译。

    使用 run_in_executor 逐个提交, 结果按完成顺序收集。
    """
    return [result async for result in _iter_compile_hp_candidates_async(jobs, worker_count)]


async def _iter_compile_hp_candidates_async(
    jobs: list[dict[str, Any]],
    worker_count: int,
):
    """Yield HP candidate compile results as soon as each job finishes."""
    if not jobs:
        return

    if worker_count <= 1:
        for job in jobs:
            yield await asyncio.to_thread(_compile_hp_candidate_job, job)
        return

    loop = asyncio.get_running_loop()
    completed: set[int] = set()

    async def _run_job(executor: ProcessPoolExecutor, job: dict[str, Any]) -> dict[str, Any]:
        try:
            return await loop.run_in_executor(executor, _compile_hp_candidate_job, job)
        except Exception as e:
            return {
                "index": job["index"],
                "version_id": job.get("version_id", ""),
                "iter_dir": job.get("iter_dir", ""),
                "code_path": job.get("code_path", ""),
                "success": False,
                "output_path": "",
                "stdout": "",
                "stderr": f"async compile error: {e}",
                "return_code": -1,
            }

    try:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            tasks = [_run_job(executor, job) for job in jobs]
            for coro in asyncio.as_completed(tasks):
                result = await coro
                completed.add(result["index"])
                status = "OK" if result["success"] else "FAIL"
                logger.info("Compiled candidate %d: %s", result["index"], status)
                yield result
    except Exception as e:
        logger.warning("Parallel HP compilation failed; falling back to serial: %s", e)
        for job in jobs:
            if job["index"] in completed:
                continue
            yield await asyncio.to_thread(_compile_hp_candidate_job, job)


# ════════════════════════════════════════
# 原有辅助函数 (保持不变 + 小优化)
# ════════════════════════════════════════

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


def _ref_py_path(state: dict | None, run_dir: Path | None) -> Path | None:
    """Resolve the generated ref.py path for unified CUDA evaluation."""
    if state and state.get("ref_py_path"):
        return Path(state["ref_py_path"])
    if run_dir is not None:
        return Path(run_dir) / "ref.py"
    return None


def _kernel_function_name(op) -> str:
    return f"{op.name}_kernel"


def _benchmark_multi(self, exe_path: Path, op, gpu_id: int | None = None) -> BenchmarkResult:
    """[优化] 支持 gpu_id 参数。"""
    return run_benchmark_multi(
        exe_path,
        self._active_shape_profiles(op),
        warmup_rounds=self.sm.config.benchmark_warmup_rounds,
        measure_rounds=self.sm.config.benchmark_measure_rounds,
        aggregator=self.sm.config.multi_shape_aggregator,
        gpu_id=gpu_id,
    )


def _profile_args_from_benchmark(self, bm: BenchmarkResult) -> list[str]:
    shape = bm.extra.get("worst_shape") if bm and bm.extra else None
    args = shape_profile_to_args(shape or {})
    measure_flag = "--iters" if bm and bm.extra.get("benchmark_arg_style") in {"iters", "iters_benchmark"} else "--rounds"
    if bm and bm.extra.get("benchmark_arg_style") == "iters_benchmark":
        args.insert(0, "--benchmark")
    args.extend(["--warmup", str(self.sm.config.ncu_warmup_rounds), measure_flag, str(self.sm.config.ncu_profile_rounds)])
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
    """原始串行/ProcessPoolExecutor 方式 (向后兼容, 仍可使用)。"""
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


# ════════════════════════════════════════
# [优化] 代码 diff 工具
# ════════════════════════════════════════

def _generate_code_diff(old_code: str, new_code: str, context_lines: int = 5) -> str:
    """
    [优化] 生成 unified diff 格式的代码差异。

    Args:
        old_code: 旧版本代码
        new_code: 新版本代码
        context_lines: diff 上下文行数

    Returns:
        unified diff 字符串
    """
    old_lines = old_code.splitlines(keepends=True)
    new_lines = new_code.splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines, new_lines,
        fromfile="best.cu", tofile="new.cu",
        n=context_lines,
    )
    return "".join(diff)


def _build_code_diff_context(
    best_code: str,
    max_skeleton_chars: int = 3000,
    max_full_chars: int = 8000,
) -> dict[str, str]:
    """
    [优化] 为 apply/analyze 节点构建代码上下文, 用 diff 模式。

    当代码较短时直接发送完整代码;
    当代码较长时发送代码骨架 (函数签名、kernel launch config 等)。

    Returns:
        {"mode": "full"|"skeleton", "code": ..., "skeleton": ...}
    """
    if len(best_code) <= max_full_chars:
        return {
            "mode": "full",
            "code": best_code,
            "skeleton": "",
        }

    # 提取骨架: kernel 函数签名、launch config、shared mem 声明
    import re
    skeleton_lines = []
    lines = best_code.split("\n")
    in_kernel = False
    brace_depth = 0

    for i, line in enumerate(lines):
        stripped = line.strip()

        # 保留 #include 和 #define
        if stripped.startswith("#include") or stripped.startswith("#define"):
            skeleton_lines.append(line)
            continue

        # 保留 __global__ / __device__ 函数签名
        if "__global__" in stripped or "__device__" in stripped:
            skeleton_lines.append(line)
            in_kernel = True
            continue

        # 保留 __shared__ 声明
        if "__shared__" in stripped:
            skeleton_lines.append(line)
            continue

        # 保留 kernel launch <<<...>>>
        if "<<<" in stripped and ">>>" in stripped:
            skeleton_lines.append(line)
            continue

        # 保留 main 函数签名
        if re.match(r"\s*(int|void)\s+main\s*\(", stripped):
            skeleton_lines.append(line)
            continue

        # 保留 cudaMalloc / cudaMemcpy
        if any(kw in stripped for kw in ["cudaMalloc", "cudaMemcpy", "cudaFree"]):
            skeleton_lines.append(line)
            continue

    skeleton = "\n".join(skeleton_lines)
    if len(skeleton) > max_skeleton_chars:
        skeleton = skeleton[:max_skeleton_chars] + "\n// ... skeleton truncated ..."

    return {
        "mode": "skeleton",
        "code": best_code[:max_full_chars] + "\n/* ... code truncated ... */",
        "skeleton": skeleton,
    }


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
