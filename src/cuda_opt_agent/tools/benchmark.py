"""
Benchmark 工具 —— 使用 cudaEvent 精确测量 kernel latency。

[优化]:
  - run_benchmark / run_benchmark_multi: 支持 gpu_id 参数
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Literal

from ..models.data import BenchmarkResult
from ..shape_profiles import ShapeProfile, profile_weight, shape_profile_label, shape_profile_to_args

logger = logging.getLogger(__name__)


def run_benchmark(
    executable_path: str | Path,
    warmup_rounds: int = 10,
    measure_rounds: int = 100,
    timeout: int = 300,
    extra_args: list[str] | None = None,
    gpu_id: int | None = None,
) -> BenchmarkResult:
    """
    运行已编译的 benchmark 可执行文件。

    [优化] 新增 gpu_id 参数, 通过 CUDA_VISIBLE_DEVICES 指定 GPU。
    """
    exe = Path(executable_path)
    if not exe.exists():
        logger.error("Executable not found: %s", exe)
        return BenchmarkResult()

    shape_args = list(extra_args or [])
    attempts = [
        ("rounds", [str(exe), *shape_args, "--warmup", str(warmup_rounds), "--rounds", str(measure_rounds)]),
        ("iters_benchmark", [str(exe), "--benchmark", *shape_args, "--warmup", str(warmup_rounds), "--iters", str(measure_rounds)]),
        ("iters", [str(exe), *shape_args, "--warmup", str(warmup_rounds), "--iters", str(measure_rounds)]),
    ]

    # [优化] 设置 CUDA_VISIBLE_DEVICES
    env = None
    if gpu_id is not None:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        failures = []
        for arg_style, cmd in attempts:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )

            if result.returncode == 0:
                bm = _parse_benchmark_output(result.stdout)
                bm.extra["benchmark_arg_style"] = arg_style
                return bm

            failures.append((arg_style, result.stdout, result.stderr))

        last_style, last_stdout, last_stderr = failures[-1]
        logger.error("Benchmark execution failed after %d argument styles (last=%s):\nstdout: %s\nstderr: %s",
                     len(failures), last_style, last_stdout[:2000], last_stderr[:2000])
        return BenchmarkResult()

    except subprocess.TimeoutExpired:
        logger.error("Benchmark timed out (%ds)", timeout)
        return BenchmarkResult()
    except Exception as e:
        logger.error("Benchmark error: %s", e)
        return BenchmarkResult()


def run_benchmark_multi(
    executable_path: str | Path,
    shape_profiles: list[ShapeProfile] | None,
    warmup_rounds: int = 10,
    measure_rounds: int = 100,
    timeout: int = 300,
    extra_args: list[str] | None = None,
    aggregator: Literal["mean", "worst", "weighted"] = "mean",
    gpu_id: int | None = None,
) -> BenchmarkResult:
    """
    Run benchmark once per shape profile and aggregate latency.

    [优化] 支持 gpu_id 参数。
    """
    profiles = shape_profiles or [{}]
    per_shape = []
    for profile in profiles:
        args = []
        args.extend(shape_profile_to_args(profile))
        if extra_args:
            args.extend(extra_args)
        result = run_benchmark(
            executable_path,
            warmup_rounds=warmup_rounds,
            measure_rounds=measure_rounds,
            timeout=timeout,
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
    arg_style = None
    if per_shape:
        arg_style = per_shape[worst_idx].get("extra", {}).get("benchmark_arg_style")
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
            "benchmark_arg_style": arg_style,
        },
    )


def _parse_benchmark_output(stdout: str) -> BenchmarkResult:
    """解析 benchmark 可执行文件的 JSON 输出。"""
    objects = _extract_json_objects(stdout)
    if objects:
        data = next(
            (
                obj for obj in reversed(objects)
                if any(k in obj for k in ("latencies_ms", "latency_ms_median", "avg_ms", "latency_ms"))
            ),
            objects[-1],
        )
        return _benchmark_result_from_dict(data)

    data = _parse_key_value_output(stdout)
    if data:
        return _benchmark_result_from_dict(data)

    logger.error("No parseable benchmark result found in output: %s", stdout[:500])
    return BenchmarkResult()


def _extract_json_objects(stdout: str) -> list[dict]:
    """Extract one or more JSON objects from mixed stdout."""
    decoder = json.JSONDecoder()
    objects = []
    idx = 0
    while idx < len(stdout):
        start = stdout.find("{", idx)
        if start < 0:
            break
        try:
            obj, end = decoder.raw_decode(stdout[start:])
        except json.JSONDecodeError:
            idx = start + 1
            continue
        if isinstance(obj, dict):
            objects.append(obj)
        idx = start + end
    return objects


def _parse_key_value_output(stdout: str) -> dict:
    """Parse simple key=value benchmark output."""
    data = {}
    for line in stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key == "latencies_ms":
            try:
                data[key] = [float(v) for v in value.split(",") if v.strip()]
            except ValueError:
                pass
        else:
            try:
                data[key] = float(value)
            except ValueError:
                data[key] = value
    return data


def _benchmark_result_from_dict(data: dict) -> BenchmarkResult:
    latencies = data.get("latencies_ms", [])

    if not latencies:
        latency = data.get("latency_ms_median", data.get("avg_ms", data.get("latency_ms", 0.0)))
        throughput = data.get("throughput_gflops")
        if throughput is None and data.get("tflops") is not None:
            throughput = data["tflops"] * 1000.0
        return BenchmarkResult(
            latency_ms_median=latency,
            latency_ms_p95=data.get("latency_ms_p95", latency),
            throughput_gflops=throughput,
            extra=data,
        )

    sorted_lat = sorted(latencies)
    med = median(sorted_lat)
    p95_idx = int(len(sorted_lat) * 0.95)
    p95 = sorted_lat[min(p95_idx, len(sorted_lat) - 1)]

    return BenchmarkResult(
        latency_ms_median=med,
        latency_ms_p95=p95,
        throughput_gflops=data.get("throughput_gflops"),
        extra={"latencies_count": len(latencies), "min_ms": sorted_lat[0], "max_ms": sorted_lat[-1]},
    )
