"""
nvcc 编译工具 —— 编译 .cu 文件并返回结果。
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


def compile_cuda(
    source_path: str | Path,
    output_path: str | Path | None = None,
    compute_capability: str = "sm_80",
    extra_flags: list[str] | None = None,
    timeout: int = 120,
) -> CompileResult:
    """
    使用 nvcc 编译 CUDA 源文件。

    Args:
        source_path: .cu 源文件路径
        output_path: 输出可执行文件路径,默认同目录同名
        compute_capability: GPU 架构, 如 "sm_80"
        extra_flags: 额外的 nvcc 编译参数
        timeout: 编译超时秒数

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

    if extra_flags:
        cmd.extend(extra_flags)

    logger.info("Compile command: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            cwd=str(source_path.parent),
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
) -> CompileResult:
    """
    编译 kernel + benchmark harness。
    harness 包含 main() 函数,负责分配内存、调用 kernel、计时。
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

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, errors="replace", timeout=timeout,
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
