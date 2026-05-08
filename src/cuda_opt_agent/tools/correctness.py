"""
数值正确性校验 —— 将 kernel 输出与参考实现对比。

[优化]:
  - 新增 check_correctness_multi_async: 基于 asyncio 的跨 shape 并行校验
  - 新增 gpu_id 参数支持多 GPU 分发
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..shape_profiles import ShapeProfile, shape_profile_label, shape_profile_to_args

logger = logging.getLogger(__name__)


@dataclass
class CorrectnessResult:
    correct: bool = False
    max_abs_error: float = float("inf")
    max_rel_error: float = float("inf")
    atol_used: float = 0.0
    rtol_used: float = 0.0
    message: str = ""
    details: dict = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


# 不同 dtype 的默认容差
TOLERANCE_MAP = {
    "fp32": {"atol": 1e-4, "rtol": 1e-3},
    "float32": {"atol": 1e-4, "rtol": 1e-3},
    "fp16": {"atol": 1e-2, "rtol": 5e-2},
    "float16": {"atol": 1e-2, "rtol": 5e-2},
    "bf16": {"atol": 1e-2, "rtol": 5e-2},
    "bfloat16": {"atol": 1e-2, "rtol": 5e-2},
    "fp64": {"atol": 1e-10, "rtol": 1e-6},
    "float64": {"atol": 1e-10, "rtol": 1e-6},
}


def get_tolerance(dtype: str) -> tuple[float, float]:
    """返回 (atol, rtol)。"""
    tol = TOLERANCE_MAP.get(dtype.lower(), {"atol": 1e-4, "rtol": 1e-3})
    return tol["atol"], tol["rtol"]


def check_correctness(
    executable_path: str | Path,
    dtype: str = "fp32",
    atol: float | None = None,
    rtol: float | None = None,
    timeout: int = 120,
    extra_args: list[str] | None = None,
    gpu_id: int | None = None,
) -> CorrectnessResult:
    """
    运行正确性校验可执行文件。

    [优化] 新增 gpu_id 参数, 用于多 GPU 分发。
    """
    exe = Path(executable_path)
    if not exe.exists():
        return CorrectnessResult(correct=False, message=f"Executable not found: {exe}")

    default_atol, default_rtol = get_tolerance(dtype)
    used_atol = atol if atol is not None else default_atol
    used_rtol = rtol if rtol is not None else default_rtol

    cmd = [
        str(exe),
        "--check",
        "--atol", str(used_atol),
        "--rtol", str(used_rtol),
    ]
    if extra_args:
        cmd.extend(extra_args)

    # [优化] 设置 CUDA_VISIBLE_DEVICES 用于多 GPU 校验
    env = None
    if gpu_id is not None:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )

        return _parse_correctness_output(
            result.stdout, result.returncode, used_atol, used_rtol
        )

    except subprocess.TimeoutExpired:
        return CorrectnessResult(
            correct=False,
            atol_used=used_atol,
            rtol_used=used_rtol,
            message=f"Correctness check timed out ({timeout}s)",
        )
    except Exception as e:
        return CorrectnessResult(
            correct=False,
            atol_used=used_atol,
            rtol_used=used_rtol,
            message=f"Correctness check error: {e}",
        )


def check_correctness_multi(
    executable_path: str | Path,
    shape_profiles: list[ShapeProfile] | None,
    dtype: str = "fp32",
    atol: float | None = None,
    rtol: float | None = None,
    timeout: int = 120,
    extra_args: list[str] | None = None,
    gpu_id: int | None = None,
) -> list[dict[str, Any]]:
    """Run correctness once per shape profile (串行, 向后兼容)."""
    profiles = shape_profiles or [{}]
    results = []
    for profile in profiles:
        args = []
        args.extend(shape_profile_to_args(profile))
        if extra_args:
            args.extend(extra_args)
        result = check_correctness(
            executable_path,
            dtype=dtype,
            atol=atol,
            rtol=rtol,
            timeout=timeout,
            extra_args=args,
            gpu_id=gpu_id,
        )
        results.append({
            "shape": profile,
            "shape_label": shape_profile_label(profile),
            "correct": result.correct,
            "max_abs_error": result.max_abs_error,
            "max_rel_error": result.max_rel_error,
            "atol_used": result.atol_used,
            "rtol_used": result.rtol_used,
            "message": result.message,
            "details": result.details,
        })
    return results


async def check_correctness_multi_async(
    executable_path: str | Path,
    shape_profiles: list[ShapeProfile] | None,
    dtype: str = "fp32",
    atol: float | None = None,
    rtol: float | None = None,
    timeout: int = 120,
    extra_args: list[str] | None = None,
    gpu_id: int | None = None,
    max_parallel: int = 2,
) -> list[dict[str, Any]]:
    """
    [优化] 跨 shape 并行正确性校验。

    使用 asyncio.Semaphore 控制并发数, 避免 GPU OOM。
    对小 shape 的校验可安全并发; 大 shape 仍受 semaphore 限制。

    Args:
        max_parallel: 最大并行校验数 (默认 2, 避免 GPU 竞争)
    """
    profiles = shape_profiles or [{}]

    if len(profiles) <= 1 or max_parallel <= 1:
        # 单 shape 或禁用并行, 走串行路径
        return await asyncio.to_thread(
            check_correctness_multi,
            executable_path,
            shape_profiles,
            dtype=dtype,
            atol=atol,
            rtol=rtol,
            timeout=timeout,
            extra_args=extra_args,
            gpu_id=gpu_id,
        )

    semaphore = asyncio.Semaphore(max_parallel)

    async def _check_one(profile: ShapeProfile) -> dict[str, Any]:
        async with semaphore:
            args = list(shape_profile_to_args(profile))
            if extra_args:
                args.extend(extra_args)
            result = await asyncio.to_thread(
                check_correctness,
                executable_path,
                dtype=dtype,
                atol=atol,
                rtol=rtol,
                timeout=timeout,
                extra_args=args,
                gpu_id=gpu_id,
            )
            return {
                "shape": profile,
                "shape_label": shape_profile_label(profile),
                "correct": result.correct,
                "max_abs_error": result.max_abs_error,
                "max_rel_error": result.max_rel_error,
                "atol_used": result.atol_used,
                "rtol_used": result.rtol_used,
                "message": result.message,
                "details": result.details,
            }

    tasks = [_check_one(p) for p in profiles]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    final = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            logger.warning("Correctness check for shape %s failed: %s",
                           shape_profile_label(profiles[i]), r)
            final.append({
                "shape": profiles[i],
                "shape_label": shape_profile_label(profiles[i]),
                "correct": False,
                "max_abs_error": float("inf"),
                "max_rel_error": float("inf"),
                "atol_used": 0.0,
                "rtol_used": 0.0,
                "message": f"Exception: {r}",
                "details": {},
            })
        else:
            final.append(r)
    return final


def summarize_correctness_results(results: list[dict[str, Any]]) -> str:
    failed = [r for r in results if not r.get("correct")]
    if not failed:
        return "all shape profiles passed"
    return "; ".join(f"{r.get('shape_label')}: {r.get('message')}" for r in failed)


def _parse_correctness_output(
    stdout: str, returncode: int, atol: float, rtol: float,
) -> CorrectnessResult:
    """解析校验输出 JSON。"""
    try:
        json_start = stdout.find("{")
        json_end = stdout.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            data = json.loads(stdout[json_start:json_end])
            correct = data.get("correct", data.get("passed", data.get("success", False)))
            max_abs_error = data.get("max_abs_error", data.get("max_abs_err", float("inf")))
            max_rel_error = data.get("max_rel_error", data.get("max_rel_err", float("inf")))
            return CorrectnessResult(
                correct=bool(correct),
                max_abs_error=max_abs_error,
                max_rel_error=max_rel_error,
                atol_used=atol,
                rtol_used=rtol,
                message=data.get("message", "ok" if correct else "failed"),
                details=data,
            )
    except json.JSONDecodeError:
        pass

    return CorrectnessResult(
        correct=returncode == 0,
        atol_used=atol,
        rtol_used=rtol,
        message=stdout[:500] if stdout else f"returncode={returncode}",
    )


def save_correctness_result(result: CorrectnessResult, output_path: str | Path) -> None:
    """将校验结果保存为 JSON。"""
    output_path = Path(output_path)
    data = {
        "correct": result.correct,
        "max_abs_error": result.max_abs_error,
        "max_rel_error": result.max_rel_error,
        "atol_used": result.atol_used,
        "rtol_used": result.rtol_used,
        "message": result.message,
    }
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
