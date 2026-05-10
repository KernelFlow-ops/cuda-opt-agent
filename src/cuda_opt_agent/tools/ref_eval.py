"""Run generated ref.py as the unified CUDA correctness/benchmark harness."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from statistics import median
from typing import Literal

from ..models.data import BenchmarkResult
from ..shape_profiles import ShapeProfile, profile_weight, shape_profile_label, shape_profile_to_args
from .benchmark import _parse_benchmark_output
from .correctness import CorrectnessResult, _parse_correctness_output, get_tolerance

logger = logging.getLogger(__name__)


def _env_for_gpu(gpu_id: int | None) -> dict[str, str] | None:
    if gpu_id is None:
        return None
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return env


def _base_cmd(
    ref_py_path: str | Path,
    cuda_source_path: str | Path,
    *,
    func_name: str,
    compute_capability: str,
    dtype: str | None,
    compile_timeout: int,
) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(ref_py_path)),
        "--cuda", str(Path(cuda_source_path)),
        "--func", func_name,
        "--arch", compute_capability or "sm_80",
        "--compile-timeout", str(compile_timeout),
    ]
    if dtype:
        cmd.extend(["--dtype", dtype])
    return cmd


def _missing_path_result(path: Path, *, atol: float = 0.0, rtol: float = 0.0) -> CorrectnessResult:
    return CorrectnessResult(
        correct=False,
        atol_used=atol,
        rtol_used=rtol,
        message=f"Path not found: {path}",
        details={"compile_ok": False},
    )


def run_ref_correctness(
    ref_py_path: str | Path,
    cuda_source_path: str | Path,
    *,
    func_name: str,
    compute_capability: str = "sm_80",
    dtype: str = "fp32",
    atol: float | None = None,
    rtol: float | None = None,
    timeout: int = 180,
    compile_timeout: int = 120,
    extra_args: list[str] | None = None,
    gpu_id: int | None = None,
) -> CorrectnessResult:
    """Run one ref.py CUDA correctness check."""
    ref_path = Path(ref_py_path)
    code_path = Path(cuda_source_path)
    default_atol, default_rtol = get_tolerance(dtype)
    used_atol = atol if atol is not None else default_atol
    used_rtol = rtol if rtol is not None else default_rtol

    if not ref_path.exists():
        return _missing_path_result(ref_path, atol=used_atol, rtol=used_rtol)
    if not code_path.exists():
        return _missing_path_result(code_path, atol=used_atol, rtol=used_rtol)

    cmd = _base_cmd(
        ref_path,
        code_path,
        func_name=func_name,
        compute_capability=compute_capability,
        dtype=dtype,
        compile_timeout=compile_timeout,
    )
    cmd.extend(["--check", "--atol", str(used_atol), "--rtol", str(used_rtol)])
    if extra_args:
        cmd.extend(extra_args)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            env=_env_for_gpu(gpu_id),
        )
    except subprocess.TimeoutExpired:
        return CorrectnessResult(
            correct=False,
            atol_used=used_atol,
            rtol_used=used_rtol,
            message=f"ref.py correctness timed out ({timeout}s)",
            details={"compile_ok": False, "command": cmd},
        )

    parsed = _parse_correctness_output(result.stdout, result.returncode, used_atol, used_rtol)
    details = dict(parsed.details or {})
    details.update({
        "command": cmd,
        "returncode": result.returncode,
        "stdout": result.stdout[:4000],
        "stderr": result.stderr[:4000],
        "compile_ok": bool(details.get("compile_ok", parsed.correct)),
    })
    message = parsed.message
    if result.returncode != 0 and result.stderr and message in {"failed", f"returncode={result.returncode}"}:
        message = result.stderr[:500]
    parsed.details = details
    parsed.message = message
    return parsed


def run_ref_correctness_multi(
    ref_py_path: str | Path,
    cuda_source_path: str | Path,
    shape_profiles: list[ShapeProfile] | None,
    *,
    func_name: str,
    compute_capability: str = "sm_80",
    dtype: str = "fp32",
    atol: float | None = None,
    rtol: float | None = None,
    timeout: int = 180,
    compile_timeout: int = 120,
    extra_args: list[str] | None = None,
    gpu_id: int | None = None,
) -> list[dict]:
    """Run ref.py correctness once per shape profile."""
    profiles = shape_profiles or [{}]
    results: list[dict] = []
    for profile in profiles:
        args = shape_profile_to_args(profile)
        if extra_args:
            args.extend(extra_args)
        result = run_ref_correctness(
            ref_py_path,
            cuda_source_path,
            func_name=func_name,
            compute_capability=compute_capability,
            dtype=dtype,
            atol=atol,
            rtol=rtol,
            timeout=timeout,
            compile_timeout=compile_timeout,
            extra_args=args,
            gpu_id=gpu_id,
        )
        results.append({
            "shape": profile,
            "shape_label": shape_profile_label(profile),
            "correct": result.correct,
            "compile_ok": bool((result.details or {}).get("compile_ok", result.correct)),
            "max_abs_error": result.max_abs_error,
            "max_rel_error": result.max_rel_error,
            "atol_used": result.atol_used,
            "rtol_used": result.rtol_used,
            "message": result.message,
            "details": result.details,
        })
    return results


def run_ref_benchmark(
    ref_py_path: str | Path,
    cuda_source_path: str | Path,
    *,
    func_name: str,
    compute_capability: str = "sm_80",
    dtype: str = "fp32",
    warmup_rounds: int = 10,
    measure_rounds: int = 100,
    timeout: int = 300,
    compile_timeout: int = 120,
    extra_args: list[str] | None = None,
    gpu_id: int | None = None,
) -> BenchmarkResult:
    """Run one ref.py CUDA benchmark."""
    ref_path = Path(ref_py_path)
    code_path = Path(cuda_source_path)
    if not ref_path.exists() or not code_path.exists():
        missing = ref_path if not ref_path.exists() else code_path
        return BenchmarkResult(extra={"error": f"Path not found: {missing}", "compile_ok": False})

    cmd = _base_cmd(
        ref_path,
        code_path,
        func_name=func_name,
        compute_capability=compute_capability,
        dtype=dtype,
        compile_timeout=compile_timeout,
    )
    cmd.extend(["--benchmark", "--warmup", str(warmup_rounds), "--rounds", str(measure_rounds)])
    if extra_args:
        cmd.extend(extra_args)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            env=_env_for_gpu(gpu_id),
        )
    except subprocess.TimeoutExpired:
        return BenchmarkResult(extra={"error": f"ref.py benchmark timed out ({timeout}s)", "command": cmd, "compile_ok": False})

    bm = _parse_benchmark_output(result.stdout)
    bm.extra.update({
        "command": cmd,
        "returncode": result.returncode,
        "stdout": result.stdout[:4000],
        "stderr": result.stderr[:4000],
        "compile_ok": bool(bm.extra.get("compile_ok", result.returncode == 0)),
        "benchmark_arg_style": "ref_py",
    })
    if result.returncode != 0 and bm.latency_ms_median <= 0:
        bm.extra.setdefault("error", result.stderr[:500] or result.stdout[:500])
    return bm


def run_ref_benchmark_multi(
    ref_py_path: str | Path,
    cuda_source_path: str | Path,
    shape_profiles: list[ShapeProfile] | None,
    *,
    func_name: str,
    compute_capability: str = "sm_80",
    dtype: str = "fp32",
    warmup_rounds: int = 10,
    measure_rounds: int = 100,
    timeout: int = 300,
    compile_timeout: int = 120,
    extra_args: list[str] | None = None,
    aggregator: Literal["mean", "worst", "weighted"] = "mean",
    gpu_id: int | None = None,
) -> BenchmarkResult:
    """Run ref.py benchmark once per shape profile and aggregate latency."""
    profiles = shape_profiles or [{}]
    per_shape = []
    for profile in profiles:
        args = shape_profile_to_args(profile)
        if extra_args:
            args.extend(extra_args)
        result = run_ref_benchmark(
            ref_py_path,
            cuda_source_path,
            func_name=func_name,
            compute_capability=compute_capability,
            dtype=dtype,
            warmup_rounds=warmup_rounds,
            measure_rounds=measure_rounds,
            timeout=timeout,
            compile_timeout=compile_timeout,
            extra_args=args,
            gpu_id=gpu_id,
        )
        per_shape.append({
            "shape": profile,
            "shape_label": shape_profile_label(profile),
            "latency_ms_median": result.latency_ms_median,
            "latency_ms_p95": result.latency_ms_p95,
            "throughput_gflops": result.throughput_gflops,
            "extra": result.extra,
        })

    latencies = [float(item["latency_ms_median"]) for item in per_shape]
    p95s = [float(item["latency_ms_p95"]) for item in per_shape]
    if any(lat <= 0 for lat in latencies):
        aggregate_latency = 0.0
        aggregate_p95 = 0.0
    elif aggregator == "worst":
        aggregate_latency = max(latencies)
        aggregate_p95 = max(p95s)
    elif aggregator == "weighted":
        weights = [profile_weight(profile) for profile in profiles]
        weight_sum = sum(weights) or 1.0
        aggregate_latency = sum(lat * weight for lat, weight in zip(latencies, weights)) / weight_sum
        aggregate_p95 = sum(p95 * weight for p95, weight in zip(p95s, weights)) / weight_sum
    else:
        aggregate_latency = sum(latencies) / len(latencies)
        aggregate_p95 = sum(p95s) / len(p95s)

    worst_idx = max(range(len(per_shape)), key=lambda i: per_shape[i]["latency_ms_median"]) if per_shape else 0
    return BenchmarkResult(
        latency_ms_median=aggregate_latency,
        latency_ms_p95=aggregate_p95,
        throughput_gflops=None,
        extra={
            "aggregator": aggregator,
            "aggregate_latency_ms": aggregate_latency,
            "per_shape": per_shape,
            "shape_count": len(per_shape),
            "worst_shape": per_shape[worst_idx]["shape"] if per_shape else {},
            "worst_shape_label": per_shape[worst_idx]["shape_label"] if per_shape else "default",
            "benchmark_arg_style": "ref_py",
            "latency_ms_median_check": median(latencies) if latencies else 0.0,
        },
    )
