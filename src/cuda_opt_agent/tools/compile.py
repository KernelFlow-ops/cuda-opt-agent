"""
nvcc 编译工具 —— 编译 .cu 文件并返回结果。

[优化]:
  - 支持 -t N 多线程编译 (nvcc_threads 参数)
  - 支持 gpu_id 指定 CUDA_VISIBLE_DEVICES
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


def _actual_output_path(output_path: Path) -> Path:
    """Return the executable path nvcc/linker actually produced."""
    if output_path.exists():
        return output_path
    if os.name == "nt" and output_path.suffix == "":
        exe_path = output_path.with_suffix(".exe")
        if exe_path.exists():
            return exe_path
    return output_path


@dataclass
class CompileResult:
    success: bool = False
    output_path: str = ""
    stdout: str = ""
    stderr: str = ""
    return_code: int = -1


def _auto_nvcc_threads() -> int:
    """[优化] 自动确定 nvcc 并行线程数。"""
    cpu = os.cpu_count() or 1
    # nvcc -t 的合理上限, 避免在大核心机器上过度竞争
    return min(cpu, 8)


def _auto_link_flags(source_path: Path) -> list[str]:
    """Infer CUDA library link flags from common library headers."""
    try:
        text = source_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []

    flags: list[str] = []
    if "cublasLt.h" in text:
        flags.extend(["-lcublasLt", "-lcublas"])
    elif "cublas_v2.h" in text:
        flags.append("-lcublas")
    if "cudnn.h" in text:
        flags.append("-lcudnn")
    return flags


def _dedupe_flags(flags: list[str]) -> list[str]:
    """Preserve flag order while dropping exact duplicates."""
    result: list[str] = []
    seen: set[str] = set()
    for flag in flags:
        if flag in seen:
            continue
        seen.add(flag)
        result.append(flag)
    return result


def compile_cuda(
    source_path: str | Path,
    output_path: str | Path | None = None,
    compute_capability: str = "sm_80",
    extra_flags: list[str] | None = None,
    timeout: int = 120,
    nvcc_threads: int = 0,
    gpu_id: int | None = None,
) -> CompileResult:
    """
    使用 nvcc 编译 CUDA 源文件。

    [优化]:
      - nvcc_threads: 0=auto, 1=禁用, >1=指定线程数 (nvcc -t N)
      - gpu_id: 指定 CUDA_VISIBLE_DEVICES, 用于多 GPU 场景

    Args:
        source_path: .cu 源文件路径
        output_path: 输出可执行文件路径,默认同目录同名
        compute_capability: GPU 架构, 如 "sm_80"
        extra_flags: 额外的 nvcc 编译参数
        timeout: 编译超时秒数
        nvcc_threads: nvcc 并行线程数
        gpu_id: 目标 GPU 索引

    Returns:
        CompileResult
    """
    source_path = Path(source_path)
    if not source_path.exists():
        return CompileResult(success=False, stderr=f"Source file not found: {source_path}")
    source_path = source_path.resolve()

    if output_path is None:
        output_path = source_path.with_suffix("")
    output_path = Path(output_path).resolve()

    nvcc = shutil.which("nvcc")
    if not nvcc:
        return CompileResult(success=False, stderr="nvcc not found; ensure CUDA Toolkit is installed")

    # 构建编译命令
    arch = compute_capability.replace("sm_", "")
    cmd = [
        nvcc,
        "-o", str(output_path),
        str(source_path),
        f"-gencode=arch=compute_{arch},code=sm_{arch}",
        "-allow-unsupported-compiler",
        "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH",
        "-O3",
        "-lineinfo",
    ]

    # [优化] 添加 -t N 多线程编译
    effective_threads = nvcc_threads
    if effective_threads == 0:
        effective_threads = _auto_nvcc_threads()
    if effective_threads > 1:
        cmd.extend(["-t", str(effective_threads)])

    inferred_flags = _auto_link_flags(source_path)
    all_extra_flags = _dedupe_flags([*(extra_flags or []), *inferred_flags])
    if all_extra_flags:
        cmd.extend(all_extra_flags)

    logger.info("Compile command: %s", " ".join(cmd))

    # [优化] 设置 CUDA_VISIBLE_DEVICES 用于多 GPU 编译
    env = None
    if gpu_id is not None:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            cwd=str(source_path.parent),
            env=env,
        )
        actual_output_path = _actual_output_path(output_path)
        return CompileResult(
            success=result.returncode == 0,
            output_path=str(actual_output_path) if result.returncode == 0 else "",
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.returncode,
        )
    except subprocess.TimeoutExpired:
        return CompileResult(success=False, stderr=f"Compilation timed out ({timeout}s)")
    except Exception as e:
        return CompileResult(success=False, stderr=f"Compilation error: {e}")


def compile_with_benchmark_harness(
    kernel_source: str | Path,
    harness_source: str | Path,
    output_path: str | Path,
    compute_capability: str = "sm_80",
    timeout: int = 120,
    nvcc_threads: int = 0,
    gpu_id: int | None = None,
) -> CompileResult:
    """
    编译 kernel + benchmark harness。

    [优化] 支持 nvcc_threads 和 gpu_id
    """
    kernel_path = Path(kernel_source).resolve()
    harness_path = Path(harness_source).resolve()
    output_path = Path(output_path).resolve()

    nvcc = shutil.which("nvcc")
    if not nvcc:
        return CompileResult(success=False, stderr="nvcc not found")

    arch = compute_capability.replace("sm_", "")
    cmd = [
        nvcc,
        "-o", str(output_path),
        str(harness_path),
        f"-gencode=arch=compute_{arch},code=sm_{arch}",
        "-allow-unsupported-compiler",
        "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH",
        "-O3",
        "-lineinfo",
        f"-I{kernel_path.parent}",
    ]

    # [优化] 添加 -t N
    effective_threads = nvcc_threads if nvcc_threads > 0 else _auto_nvcc_threads()
    if effective_threads > 1:
        cmd.extend(["-t", str(effective_threads)])

    env = None
    if gpu_id is not None:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, errors="replace",
            timeout=timeout, env=env,
        )
        actual_output_path = _actual_output_path(output_path)
        return CompileResult(
            success=result.returncode == 0,
            output_path=str(actual_output_path) if result.returncode == 0 else "",
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.returncode,
        )
    except subprocess.TimeoutExpired:
        return CompileResult(success=False, stderr=f"Compilation timed out ({timeout}s)")
    except Exception as e:
        return CompileResult(success=False, stderr=f"Compilation error: {e}")
