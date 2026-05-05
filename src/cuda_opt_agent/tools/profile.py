"""
Profiling 工具 —— 调用 Nsight Compute (ncu) 抓取性能指标。
解析为结构化 dict,同时保留原始文本给 LLM。
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path

from ..models.data import NcuMetrics

logger = logging.getLogger(__name__)

# ncu 需要关注的核心 metric
CORE_METRICS = [
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "launch__shared_mem_per_block",
    "launch__registers_per_thread",
    "sm__warps_active.avg.pct_of_peak_sustained_elapsed",
    "smsp__warp_issue_stalled_barrier.avg.pct_of_peak_sustained_elapsed",
    "smsp__warp_issue_stalled_long_scoreboard.avg.pct_of_peak_sustained_elapsed",
    "smsp__warp_issue_stalled_math_pipe_throttle.avg.pct_of_peak_sustained_elapsed",
    "smsp__warp_issue_stalled_membar.avg.pct_of_peak_sustained_elapsed",
    "smsp__warp_issue_stalled_mio_throttle.avg.pct_of_peak_sustained_elapsed",
    "smsp__warp_issue_stalled_not_selected.avg.pct_of_peak_sustained_elapsed",
    "smsp__warp_issue_stalled_short_scoreboard.avg.pct_of_peak_sustained_elapsed",
    "smsp__warp_issue_stalled_wait.avg.pct_of_peak_sustained_elapsed",
]


def run_ncu_profile(
    executable_path: str | Path,
    output_report_path: str | Path | None = None,
    kernel_name: str | None = None,
    extra_args: list[str] | None = None,
    executable_args: list[str] | None = None,
    timeout: int = 600,
) -> NcuMetrics:
    """
    使用 ncu 对可执行文件进行 profiling。

    Args:
        executable_path: 可执行文件路径
        output_report_path: ncu 报告输出路径 (文本格式)
        kernel_name: 指定 kernel 名 (用于过滤)
        extra_args: 额外 ncu 参数
        executable_args: 传给 benchmark 可执行文件的参数
        timeout: 超时秒数

    Returns:
        NcuMetrics (包含结构化指标和原始文本)
    """
    exe = Path(executable_path)
    if not exe.exists():
        logger.error("Executable not found: %s", exe)
        return NcuMetrics(raw_text=f"ERROR: executable not found: {exe}")

    ncu = shutil.which("ncu")
    if not ncu:
        logger.error("ncu (Nsight Compute) not found")
        return NcuMetrics(raw_text="ERROR: ncu not found")

    if output_report_path is None:
        output_report_path = exe.parent / "ncu_report.txt"
    output_report_path = Path(output_report_path)

    # 构建 ncu 命令
    metrics_str = ",".join(CORE_METRICS)
    cmd = [
        ncu,
        "--metrics", metrics_str,
        "--launch-count", "1",
        "--log-file", str(output_report_path),
    ]

    if kernel_name:
        cmd.extend(["--kernel-name", kernel_name])

    if extra_args:
        cmd.extend(extra_args)

    cmd.append(str(exe))
    cmd.extend(executable_args or ["--warmup", "0", "--rounds", "1"])

    logger.info("ncu command: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        raw_text = ""
        if output_report_path.exists():
            raw_text = output_report_path.read_text(encoding="utf-8", errors="replace")[:20000]
        elif result.stdout:
            raw_text = result.stdout[:20000]

        if result.returncode != 0:
            logger.warning("ncu returned non-zero exit code %d\nstderr: %s", result.returncode, result.stderr[:2000])

        return _parse_ncu_output(raw_text)

    except subprocess.TimeoutExpired:
        logger.error("ncu timed out (%ds)", timeout)
        return NcuMetrics(raw_text=f"ERROR: ncu timed out after {timeout}s")
    except Exception as e:
        logger.error("ncu error: %s", e)
        return NcuMetrics(raw_text=f"ERROR: {e}")


def _parse_ncu_output(raw_text: str) -> NcuMetrics:
    """解析 ncu 文本输出为结构化 NcuMetrics。"""
    metrics = NcuMetrics(raw_text=raw_text)

    def _extract_pct(pattern: str) -> float | None:
        m = re.search(pattern + r".*?([\d]+(?:\.[\d]+)?)\s*$", raw_text, re.MULTILINE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
        return None

    def _extract_int(pattern: str) -> int | None:
        m = re.search(pattern + r".*?(\d+)\s*$", raw_text, re.MULTILINE)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass
        return None

    # 核心吞吐量指标
    metrics.sm_throughput_pct = _extract_pct(r"sm__throughput.*pct_of_peak")
    metrics.compute_memory_throughput_pct = _extract_pct(r"gpu__compute_memory_throughput.*pct_of_peak")
    metrics.l1_throughput_pct = _extract_pct(r"l1tex__throughput.*pct_of_peak")
    metrics.l2_throughput_pct = _extract_pct(r"lts__throughput.*pct_of_peak")
    metrics.dram_throughput_pct = _extract_pct(r"dram__throughput.*pct_of_peak")

    # Launch 配置
    metrics.shared_mem_per_block_bytes = _extract_int(r"launch__shared_mem_per_block")
    metrics.registers_per_thread = _extract_int(r"launch__registers_per_thread")

    # Occupancy
    metrics.occupancy_pct = _extract_pct(r"sm__warps_active.*pct_of_peak")

    # Stall 原因
    stall_patterns = {
        "barrier": r"stalled_barrier.*pct",
        "long_scoreboard": r"stalled_long_scoreboard.*pct",
        "math_pipe_throttle": r"stalled_math_pipe_throttle.*pct",
        "membar": r"stalled_membar.*pct",
        "mio_throttle": r"stalled_mio_throttle.*pct",
        "not_selected": r"stalled_not_selected.*pct",
        "short_scoreboard": r"stalled_short_scoreboard.*pct",
        "wait": r"stalled_wait.*pct",
    }
    for name, pattern in stall_patterns.items():
        val = _extract_pct(pattern)
        if val is not None:
            metrics.stall_reasons[name] = val

    return metrics


def format_ncu_for_prompt(metrics: NcuMetrics) -> str:
    """将 NcuMetrics 格式化为 Prompt 可用的文本。"""
    lines = ["=== ncu Profiling Structured Summary ==="]

    if metrics.sm_throughput_pct is not None:
        lines.append(f"SM Throughput: {metrics.sm_throughput_pct:.1f}% of peak")
    if metrics.compute_memory_throughput_pct is not None:
        lines.append(f"Compute/Memory Throughput: {metrics.compute_memory_throughput_pct:.1f}%")
    if metrics.dram_throughput_pct is not None:
        lines.append(f"DRAM Throughput: {metrics.dram_throughput_pct:.1f}%")
    if metrics.l1_throughput_pct is not None:
        lines.append(f"L1 Throughput: {metrics.l1_throughput_pct:.1f}%")
    if metrics.l2_throughput_pct is not None:
        lines.append(f"L2 Throughput: {metrics.l2_throughput_pct:.1f}%")
    if metrics.occupancy_pct is not None:
        lines.append(f"Occupancy: {metrics.occupancy_pct:.1f}%")
    if metrics.registers_per_thread is not None:
        lines.append(f"Registers/Thread: {metrics.registers_per_thread}")
    if metrics.shared_mem_per_block_bytes is not None:
        lines.append(f"Shared Mem/Block: {metrics.shared_mem_per_block_bytes} bytes")

    if metrics.stall_reasons:
        lines.append("\nStall reason distribution:")
        for name, val in sorted(metrics.stall_reasons.items(), key=lambda x: -x[1]):
            lines.append(f"  {name}: {val:.1f}%")

    lines.append("\n=== ncu Raw Output (excerpt) ===")
    lines.append(metrics.raw_text[:8000])

    return "\n".join(lines)
