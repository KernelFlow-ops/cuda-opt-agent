"""
Profiling 工具 - 调用 Nsight Compute (ncu) 抓取性能指标。

自适应 profiling 将指标采集拆为三阶段:
1. Phase-1 固定采集分类指标,判断 memory/compute/latency bound。
2. Phase-2 根据分类结果只深挖相关瓶颈路径。
3. Phase-3 仅在资源接近饱和时补抓互补方向指标,帮助 LLM 切换优化方向。
"""

from __future__ import annotations

import csv
import io
import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Literal

from ..models.data import NcuMetrics

logger = logging.getLogger(__name__)

BottleneckClass = Literal["memory_bound", "compute_bound", "latency_bound", "mixed", "unknown"]

# Phase-1: classification metrics, always collected.
METRIC_SM_THROUGHPUT = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
METRIC_COMPUTE_MEMORY_THROUGHPUT = "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"
METRIC_DRAM_THROUGHPUT = "dram__throughput.avg.pct_of_peak_sustained_elapsed"
METRIC_OCCUPANCY = "sm__warps_active.avg.pct_of_peak_sustained_elapsed"
METRIC_GPU_TIME = "gpu__time_duration.sum"

# Phase-2: memory path.
METRIC_L1_THROUGHPUT = "l1tex__throughput.avg.pct_of_peak_sustained_elapsed"
METRIC_L2_THROUGHPUT = "lts__throughput.avg.pct_of_peak_sustained_elapsed"
METRIC_DRAM_BYTES_READ = "dram__bytes_read.sum"
METRIC_DRAM_BYTES_WRITE = "dram__bytes_write.sum"
METRIC_L1_HIT_RATE = "l1tex__t_sector_hit_rate.pct"
METRIC_L2_HIT_RATE = "lts__t_sector_hit_rate.pct"
METRIC_LONG_SCOREBOARD = "smsp__warp_issue_stalled_long_scoreboard.avg.pct"
METRIC_MIO_THROTTLE = "smsp__warp_issue_stalled_mio_throttle.avg.pct"

# Phase-2: compute path.
METRIC_MATH_PIPE_THROTTLE = "smsp__warp_issue_stalled_math_pipe_throttle.avg.pct"
METRIC_FMA_PIPE = "sm__pipe_fma_cycles_active.avg.pct"
METRIC_TENSOR_PIPE = "sm__pipe_tensor_op_hmma_cycles_active.avg.pct"
METRIC_FP16_INST = "sm__inst_executed_pipe_fp16.avg.pct"
METRIC_NOT_SELECTED = "smsp__warp_issue_stalled_not_selected.avg.pct"
METRIC_FFMA_INST = "sm__sass_thread_inst_executed_op_ffma_pred_on.avg"

# Phase-2: latency/occupancy path.
METRIC_SHARED_MEM_PER_BLOCK = "launch__shared_mem_per_block"
METRIC_REGISTERS_PER_THREAD = "launch__registers_per_thread"
METRIC_BARRIER = "smsp__warp_issue_stalled_barrier.avg.pct"
METRIC_MEMBAR = "smsp__warp_issue_stalled_membar.avg.pct"
METRIC_SHORT_SCOREBOARD = "smsp__warp_issue_stalled_short_scoreboard.avg.pct"
METRIC_WAIT = "smsp__warp_issue_stalled_wait.avg.pct"

PHASE1_METRICS = [
    METRIC_SM_THROUGHPUT,
    METRIC_COMPUTE_MEMORY_THROUGHPUT,
    METRIC_DRAM_THROUGHPUT,
    METRIC_OCCUPANCY,
    METRIC_GPU_TIME,
]

MEMORY_METRICS = [
    METRIC_L1_THROUGHPUT,
    METRIC_L2_THROUGHPUT,
    METRIC_DRAM_BYTES_READ,
    METRIC_DRAM_BYTES_WRITE,
    METRIC_L1_HIT_RATE,
    METRIC_L2_HIT_RATE,
    METRIC_LONG_SCOREBOARD,
    METRIC_MIO_THROTTLE,
]

COMPUTE_METRICS = [
    METRIC_MATH_PIPE_THROTTLE,
    METRIC_FMA_PIPE,
    METRIC_TENSOR_PIPE,
    METRIC_FP16_INST,
    METRIC_NOT_SELECTED,
    METRIC_FFMA_INST,
]

LATENCY_METRICS = [
    METRIC_SHARED_MEM_PER_BLOCK,
    METRIC_REGISTERS_PER_THREAD,
    METRIC_BARRIER,
    METRIC_MEMBAR,
    METRIC_SHORT_SCOREBOARD,
    METRIC_WAIT,
]

PHASE3_COMPUTE_TOP3 = [METRIC_MATH_PIPE_THROTTLE, METRIC_FMA_PIPE, METRIC_TENSOR_PIPE]
PHASE3_MEMORY_TOP3 = [METRIC_L2_THROUGHPUT, METRIC_LONG_SCOREBOARD, METRIC_MIO_THROTTLE]
PHASE3_OCCUPANCY_TOP3 = [METRIC_SHORT_SCOREBOARD, METRIC_FP16_INST, METRIC_FFMA_INST]

# Backward-compatible metric collection name. The adaptive path no longer collects all
# of these in one call; this list is useful for callers that still need the full set.
CORE_METRICS = list(dict.fromkeys(PHASE1_METRICS + MEMORY_METRICS + COMPUTE_METRICS + LATENCY_METRICS))

METRIC_ALIASES = {
    METRIC_LONG_SCOREBOARD: ["smsp__warp_issue_stalled_long_scoreboard.avg.pct_of_peak_sustained_elapsed"],
    METRIC_MIO_THROTTLE: ["smsp__warp_issue_stalled_mio_throttle.avg.pct_of_peak_sustained_elapsed"],
    METRIC_MATH_PIPE_THROTTLE: ["smsp__warp_issue_stalled_math_pipe_throttle.avg.pct_of_peak_sustained_elapsed"],
    METRIC_FMA_PIPE: ["sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed"],
    METRIC_TENSOR_PIPE: ["sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed"],
    METRIC_NOT_SELECTED: ["smsp__warp_issue_stalled_not_selected.avg.pct_of_peak_sustained_elapsed"],
    METRIC_BARRIER: ["smsp__warp_issue_stalled_barrier.avg.pct_of_peak_sustained_elapsed"],
    METRIC_MEMBAR: ["smsp__warp_issue_stalled_membar.avg.pct_of_peak_sustained_elapsed"],
    METRIC_SHORT_SCOREBOARD: ["smsp__warp_issue_stalled_short_scoreboard.avg.pct_of_peak_sustained_elapsed"],
    METRIC_WAIT: ["smsp__warp_issue_stalled_wait.avg.pct_of_peak_sustained_elapsed"],
}


def run_adaptive_ncu_profile(
    executable_path: str | Path,
    output_report_path: str | Path | None = None,
    kernel_name: str | None = None,
    extra_args: list[str] | None = None,
    executable_args: list[str] | None = None,
    launch_count: int = 3,
    timeout: int = 600,
) -> NcuMetrics:
    """Run adaptive multi-phase ncu profiling and return merged diagnostics."""
    exe = Path(executable_path)
    final_report_path = Path(output_report_path) if output_report_path is not None else exe.parent / "ncu_report.txt"

    phase_results = [
        _run_ncu_metrics(
            exe,
            PHASE1_METRICS,
            phase_name="phase1",
            output_report_path=final_report_path,
            kernel_name=kernel_name,
            extra_args=extra_args,
            executable_args=executable_args,
            launch_count=launch_count,
            timeout=timeout,
        )
    ]

    phase1 = phase_results[0]
    if not _metric_values(phase1):
        diagnosis = _build_diagnosis(phase1, "unknown", _empty_saturation(), ["phase1"])
        phase1.extra["diagnosis"] = diagnosis
        _write_combined_report(final_report_path, phase1)
        return phase1

    initial_classification = classify_ncu_bottleneck(phase1)
    phase2_metrics, phase2_name = _phase2_metrics_for(initial_classification, phase1)
    phase_results.append(
        _run_ncu_metrics(
            exe,
            phase2_metrics,
            phase_name=phase2_name,
            output_report_path=final_report_path,
            kernel_name=kernel_name,
            extra_args=extra_args,
            executable_args=executable_args,
            launch_count=launch_count,
            timeout=timeout,
        )
    )

    merged = _merge_phase_results(phase_results)
    classification = classify_ncu_bottleneck(merged)
    if classification in {"mixed", "unknown"} and initial_classification not in {"mixed", "unknown"}:
        classification = initial_classification

    saturation = check_ncu_saturation(merged, classification)
    phase3_metrics = saturation.get("phase3_metrics", [])
    if phase3_metrics:
        phase_results.append(
            _run_ncu_metrics(
                exe,
                phase3_metrics,
                phase_name="phase3_complementary",
                output_report_path=final_report_path,
                kernel_name=kernel_name,
                extra_args=extra_args,
                executable_args=executable_args,
                launch_count=launch_count,
                timeout=timeout,
            )
        )
        merged = _merge_phase_results(phase_results)
        saturation = check_ncu_saturation(merged, classification)

    phase_names = [str(result.extra.get("phase", f"phase{i + 1}")) for i, result in enumerate(phase_results)]
    merged.extra["diagnosis"] = _build_diagnosis(merged, classification, saturation, phase_names)
    _write_combined_report(final_report_path, merged)
    return merged


def run_ncu_profile(
    executable_path: str | Path,
    output_report_path: str | Path | None = None,
    kernel_name: str | None = None,
    extra_args: list[str] | None = None,
    executable_args: list[str] | None = None,
    launch_count: int = 3,
    timeout: int = 600,
) -> NcuMetrics:
    """Backward-compatible wrapper for adaptive ncu profiling."""
    return run_adaptive_ncu_profile(
        executable_path,
        output_report_path=output_report_path,
        kernel_name=kernel_name,
        extra_args=extra_args,
        executable_args=executable_args,
        launch_count=launch_count,
        timeout=timeout,
    )


def classify_ncu_bottleneck(metrics: NcuMetrics) -> BottleneckClass:
    """Classify the kernel using Phase-1 metrics."""
    sm = _metric_float(metrics, METRIC_SM_THROUGHPUT)
    compute_memory = _metric_float(metrics, METRIC_COMPUTE_MEMORY_THROUGHPUT)
    dram = _metric_float(metrics, METRIC_DRAM_THROUGHPUT)
    occupancy = _metric_float(metrics, METRIC_OCCUPANCY)

    if all(value is None for value in (sm, compute_memory, dram, occupancy)):
        return "unknown"

    sm_value = sm or 0.0
    dram_value = dram or 0.0
    compute_memory_value = compute_memory or 0.0

    memory_signal = dram_value > 60.0 or compute_memory_value > sm_value * 1.5
    compute_signal = sm_value > 60.0 or sm_value > dram_value * 1.5
    latency_signal = (occupancy is not None and occupancy < 40.0) or (sm_value < 30.0 and dram_value < 30.0)

    if latency_signal and sm_value < 30.0 and dram_value < 30.0:
        return "latency_bound"
    if memory_signal and not compute_signal:
        return "memory_bound"
    if compute_signal and not memory_signal:
        return "compute_bound"
    if memory_signal and compute_signal:
        return "memory_bound" if max(dram_value, compute_memory_value) >= sm_value else "compute_bound"
    if latency_signal:
        return "latency_bound"
    return "mixed"


def check_ncu_saturation(metrics: NcuMetrics, classification: BottleneckClass | None = None) -> dict[str, Any]:
    """Return saturation status and optional Phase-3 complementary metrics."""
    classification = classification or classify_ncu_bottleneck(metrics)
    sm = _metric_float(metrics, METRIC_SM_THROUGHPUT) or 0.0
    dram = _metric_float(metrics, METRIC_DRAM_THROUGHPUT) or 0.0
    occupancy = _metric_float(metrics, METRIC_OCCUPANCY) or 0.0
    l2_hit_rate = _metric_float(metrics, METRIC_L2_HIT_RATE)
    fma_pipe = _metric_float(metrics, METRIC_FMA_PIPE) or 0.0

    saturation = _empty_saturation()

    if classification == "memory_bound" and dram > 85.0 and l2_hit_rate is not None and l2_hit_rate < 30.0:
        saturation.update({
            "resource": "dram",
            "utilization": dram,
            "headroom_pct": _headroom(dram),
            "is_saturated": True,
            "reason": "DRAM throughput is near peak and cache hit rate is low",
            "phase3_metrics": PHASE3_COMPUTE_TOP3,
        })
    elif classification == "compute_bound" and sm > 85.0 and fma_pipe > 80.0:
        saturation.update({
            "resource": "compute",
            "utilization": sm,
            "headroom_pct": _headroom(sm),
            "is_saturated": True,
            "reason": "SM and FMA pipelines are near peak",
            "phase3_metrics": PHASE3_MEMORY_TOP3,
        })
    elif occupancy > 80.0 and sm < 30.0 and dram < 30.0:
        saturation.update({
            "resource": "occupancy",
            "utilization": occupancy,
            "headroom_pct": _headroom(occupancy),
            "is_saturated": True,
            "reason": "Many warps are active but both compute and DRAM throughput are low",
            "phase3_metrics": PHASE3_OCCUPANCY_TOP3,
        })

    return saturation


def format_ncu_for_prompt(metrics: NcuMetrics) -> str:
    """Format NcuMetrics as a structured profiling report for LLM prompts."""
    lines = ["=== ncu Adaptive Diagnosis ==="]
    diagnosis = metrics.extra.get("diagnosis") if metrics.extra else None
    if diagnosis:
        lines.append(json.dumps(diagnosis, ensure_ascii=False, indent=2))
    else:
        fallback = _build_diagnosis(metrics, classify_ncu_bottleneck(metrics), _empty_saturation(), ["unknown"])
        lines.append(json.dumps(fallback, ensure_ascii=False, indent=2))

    lines.append("\n=== ncu Key Metrics ===")
    _append_metric_line(lines, "SM Throughput", metrics.sm_throughput_pct, "% of peak")
    _append_metric_line(lines, "Compute/Memory Throughput", metrics.compute_memory_throughput_pct, "%")
    _append_metric_line(lines, "DRAM Throughput", metrics.dram_throughput_pct, "%")
    _append_metric_line(lines, "L1 Throughput", metrics.l1_throughput_pct, "%")
    _append_metric_line(lines, "L2 Throughput", metrics.l2_throughput_pct, "%")
    _append_metric_line(lines, "Occupancy", metrics.occupancy_pct, "%")
    if metrics.registers_per_thread is not None:
        lines.append(f"Registers/Thread: {metrics.registers_per_thread}")
    if metrics.shared_mem_per_block_bytes is not None:
        lines.append(f"Shared Mem/Block: {metrics.shared_mem_per_block_bytes} bytes")

    actionable = _actionable_metric_values(metrics)
    if actionable:
        lines.append("\nActionable Metrics:")
        lines.append(json.dumps(actionable, ensure_ascii=False, indent=2, sort_keys=True))

    if metrics.stall_reasons:
        lines.append("\nStall reason distribution:")
        for name, val in sorted(metrics.stall_reasons.items(), key=lambda item: -item[1]):
            lines.append(f"  {name}: {val:.1f}%")

    lines.append("\n=== ncu Raw Output (excerpt) ===")
    lines.append(metrics.raw_text[:8000])
    return "\n".join(lines)


def _run_ncu_metrics(
    executable_path: str | Path,
    metrics: list[str],
    *,
    phase_name: str,
    output_report_path: Path,
    kernel_name: str | None,
    extra_args: list[str] | None,
    executable_args: list[str] | None,
    launch_count: int,
    timeout: int,
) -> NcuMetrics:
    exe = Path(executable_path)
    requested_metrics = _dedupe_metrics(metrics)
    if not exe.exists():
        logger.error("Executable not found: %s", exe)
        return _error_metrics(phase_name, requested_metrics, f"ERROR: executable not found: {exe}")

    ncu = shutil.which("ncu")
    if not ncu:
        logger.error("ncu (Nsight Compute) not found")
        return _error_metrics(phase_name, requested_metrics, "ERROR: ncu not found")

    phase_report_path = _phase_output_path(output_report_path, phase_name)
    cmd = [
        ncu,
        "--metrics", ",".join(requested_metrics),
        "--launch-count", str(launch_count),
        "--csv",
        "--page", "raw",
        "--log-file", str(phase_report_path),
    ]

    if kernel_name:
        cmd.extend(["--kernel-name", kernel_name])
    if extra_args:
        cmd.extend(extra_args)

    cmd.append(str(exe))
    cmd.extend(executable_args or ["--warmup", "1", "--rounds", "1"])

    logger.info("ncu %s command: %s", phase_name, " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        logger.error("ncu %s timed out (%ds)", phase_name, timeout)
        return _error_metrics(phase_name, requested_metrics, f"ERROR: ncu timed out after {timeout}s")
    except Exception as e:
        logger.error("ncu %s error: %s", phase_name, e)
        return _error_metrics(phase_name, requested_metrics, f"ERROR: {e}")

    raw_text = ""
    if phase_report_path.exists():
        raw_text = phase_report_path.read_text(encoding="utf-8", errors="replace")
    elif result.stdout:
        raw_text = result.stdout

    if result.returncode != 0:
        logger.warning("ncu %s returned non-zero exit code %d\nstderr: %s", phase_name, result.returncode, result.stderr[:2000])

    parsed = _parse_ncu_output(raw_text)
    parsed.raw_text = raw_text[:20000]
    parsed.extra.update({
        "phase": phase_name,
        "requested_metrics": requested_metrics,
        "ncu_returncode": result.returncode,
        "ncu_stderr": result.stderr[:2000],
        "report_path": str(phase_report_path),
    })
    return parsed


def _parse_ncu_output(raw_text: str) -> NcuMetrics:
    """Parse ncu --csv --page raw output into NcuMetrics."""
    metrics = NcuMetrics(raw_text=raw_text)
    csv_text = _extract_csv_payload(raw_text)
    if not csv_text:
        metrics.extra["parse_error"] = "ncu csv header not found"
        return metrics

    try:
        rows = list(csv.reader(io.StringIO(csv_text)))
    except csv.Error as e:
        metrics.extra["parse_error"] = f"ncu csv parse failed: {e}"
        return metrics

    values, units, parser_format = _parse_ncu_csv_rows(rows)

    if not values:
        metrics.extra["parse_error"] = "ncu csv metrics not found"
        return metrics

    metrics.extra.update({
        "parser": "ncu_csv_raw",
        "parser_format": parser_format,
        "metrics": values,
        "metric_units": units,
    })
    _populate_ncu_metric_fields(metrics)
    return metrics


def _extract_csv_payload(raw_text: str) -> str:
    lines = raw_text.splitlines()
    for index, line in enumerate(lines):
        if "Metric Name" in line and "Metric Value" in line:
            return "\n".join(lines[index:])
        if line.lstrip().startswith('"ID"') and "," in line:
            return "\n".join(lines[index:])
    return ""


def _parse_ncu_csv_rows(rows: list[list[str]]) -> tuple[dict[str, float | int | str], dict[str, str], str]:
    if not rows:
        return {}, {}, "empty"

    header = [cell.strip() for cell in rows[0]]
    if "Metric Name" in header and "Metric Value" in header:
        return _parse_ncu_long_csv_rows(header, rows[1:])
    return _parse_ncu_wide_csv_rows(header, rows[1:])


def _parse_ncu_long_csv_rows(
    header: list[str],
    rows: list[list[str]],
) -> tuple[dict[str, float | int | str], dict[str, str], str]:
    values: dict[str, float | int | str] = {}
    units: dict[str, str] = {}
    name_idx = header.index("Metric Name")
    value_idx = header.index("Metric Value")
    unit_idx = header.index("Metric Unit") if "Metric Unit" in header else -1

    for row in rows:
        metric_name = _cell(row, name_idx).strip()
        if not metric_name:
            continue
        values[metric_name] = _parse_metric_value(_cell(row, value_idx))
        if unit_idx >= 0:
            units[metric_name] = _cell(row, unit_idx).strip()
    return values, units, "long"


def _parse_ncu_wide_csv_rows(
    header: list[str],
    rows: list[list[str]],
) -> tuple[dict[str, float | int | str], dict[str, str], str]:
    values: dict[str, float | int | str] = {}
    units: dict[str, str] = {}
    if not rows:
        return values, units, "wide"

    unit_row = rows[0]
    data_rows = rows[1:]
    for idx, metric_name in enumerate(header):
        if metric_name:
            units[metric_name] = _cell(unit_row, idx).strip()

    for row in data_rows:
        if not any(cell.strip() for cell in row):
            continue
        for idx, metric_name in enumerate(header):
            if not metric_name:
                continue
            raw_value = _cell(row, idx)
            if raw_value.strip() == "":
                continue
            values[metric_name] = _parse_metric_value(raw_value)
    return values, units, "wide"


def _cell(row: list[str], index: int) -> str:
    return row[index] if 0 <= index < len(row) else ""


def _parse_metric_value(value: str) -> float | int | str:
    normalized = value.strip().strip('"').replace(",", "")
    if normalized == "":
        return ""
    try:
        number = float(normalized)
    except ValueError:
        return value.strip().strip('"')
    return int(number) if number.is_integer() else number


def _populate_ncu_metric_fields(metrics: NcuMetrics) -> None:
    metrics.sm_throughput_pct = _metric_float(metrics, METRIC_SM_THROUGHPUT)
    metrics.compute_memory_throughput_pct = _metric_float(metrics, METRIC_COMPUTE_MEMORY_THROUGHPUT)
    metrics.l1_throughput_pct = _metric_float(metrics, METRIC_L1_THROUGHPUT)
    metrics.l2_throughput_pct = _metric_float(metrics, METRIC_L2_THROUGHPUT)
    metrics.dram_throughput_pct = _metric_float(metrics, METRIC_DRAM_THROUGHPUT)
    metrics.occupancy_pct = _metric_float(metrics, METRIC_OCCUPANCY)
    metrics.shared_mem_per_block_bytes = _metric_int(metrics, METRIC_SHARED_MEM_PER_BLOCK)
    metrics.registers_per_thread = _metric_int(metrics, METRIC_REGISTERS_PER_THREAD)

    stall_metrics = {
        "barrier": METRIC_BARRIER,
        "long_scoreboard": METRIC_LONG_SCOREBOARD,
        "math_pipe_throttle": METRIC_MATH_PIPE_THROTTLE,
        "membar": METRIC_MEMBAR,
        "mio_throttle": METRIC_MIO_THROTTLE,
        "not_selected": METRIC_NOT_SELECTED,
        "short_scoreboard": METRIC_SHORT_SCOREBOARD,
        "wait": METRIC_WAIT,
    }
    for name, metric_name in stall_metrics.items():
        value = _metric_float(metrics, metric_name)
        if value is not None:
            metrics.stall_reasons[name] = value


def _phase2_metrics_for(classification: BottleneckClass, metrics: NcuMetrics) -> tuple[list[str], str]:
    if classification == "memory_bound":
        return MEMORY_METRICS, "phase2_memory"
    if classification == "compute_bound":
        return COMPUTE_METRICS, "phase2_compute"
    if classification == "latency_bound":
        return LATENCY_METRICS, "phase2_latency"

    sm = _metric_float(metrics, METRIC_SM_THROUGHPUT) or 0.0
    dram = _metric_float(metrics, METRIC_DRAM_THROUGHPUT) or 0.0
    compute_memory = _metric_float(metrics, METRIC_COMPUTE_MEMORY_THROUGHPUT) or 0.0
    if max(dram, compute_memory) > sm:
        return MEMORY_METRICS, "phase2_memory"
    if sm > max(dram, compute_memory):
        return COMPUTE_METRICS, "phase2_compute"
    return LATENCY_METRICS, "phase2_latency"


def _merge_phase_results(phase_results: list[NcuMetrics]) -> NcuMetrics:
    values: dict[str, float | int | str] = {}
    units: dict[str, str] = {}
    raw_parts: list[str] = []
    phases: list[dict[str, Any]] = []
    parse_errors: list[str] = []

    for result in phase_results:
        phase = str(result.extra.get("phase", "unknown"))
        raw_parts.append(f"## {phase}\n{result.raw_text}")
        phases.append({
            "name": phase,
            "requested_metrics": result.extra.get("requested_metrics", []),
            "returncode": result.extra.get("ncu_returncode"),
        })
        values.update(result.extra.get("metrics", {}))
        units.update(result.extra.get("metric_units", {}))
        if result.extra.get("parse_error"):
            parse_errors.append(f"{phase}: {result.extra['parse_error']}")

    merged = NcuMetrics(raw_text="\n\n".join(raw_parts)[:60000])
    if values:
        merged.extra.update({
            "parser": "ncu_csv_raw",
            "metrics": values,
            "metric_units": units,
        })
    if parse_errors:
        merged.extra["parse_warnings"] = parse_errors
    merged.extra["phases"] = phases
    _populate_ncu_metric_fields(merged)
    return merged


def _build_diagnosis(
    metrics: NcuMetrics,
    classification: BottleneckClass,
    saturation: dict[str, Any],
    phase_names: list[str],
) -> dict[str, Any]:
    primary = _primary_bottleneck(metrics, classification)
    secondary = _secondary_opportunity(classification, saturation)
    return {
        "classification": classification,
        "phases": phase_names,
        "saturation": {key: value for key, value in saturation.items() if key != "phase3_metrics"},
        "primary_bottleneck": primary,
        "secondary_opportunity": secondary,
        "actionable_metrics": _actionable_metric_values(metrics),
        "recommendation_hint": _recommendation_hint(classification, saturation, primary, secondary),
    }


def _primary_bottleneck(metrics: NcuMetrics, classification: BottleneckClass) -> dict[str, Any]:
    sm = _metric_float(metrics, METRIC_SM_THROUGHPUT) or 0.0
    compute_memory = _metric_float(metrics, METRIC_COMPUTE_MEMORY_THROUGHPUT) or 0.0
    dram = _metric_float(metrics, METRIC_DRAM_THROUGHPUT) or 0.0
    occupancy = _metric_float(metrics, METRIC_OCCUPANCY) or 0.0

    if classification == "memory_bound":
        utilization = max(dram, compute_memory)
        return {
            "type": "dram_bandwidth" if dram >= compute_memory else "memory_subsystem",
            "utilization": utilization,
            "headroom_pct": _headroom(utilization),
            "dram_throughput_pct": dram,
            "compute_memory_throughput_pct": compute_memory,
            "l2_hit_rate_pct": _metric_float(metrics, METRIC_L2_HIT_RATE),
        }
    if classification == "compute_bound":
        return {
            "type": "compute_pipeline",
            "utilization": sm,
            "headroom_pct": _headroom(sm),
            "fma_pipe_pct": _metric_float(metrics, METRIC_FMA_PIPE),
            "tensor_pipe_pct": _metric_float(metrics, METRIC_TENSOR_PIPE),
        }
    if classification == "latency_bound":
        return {
            "type": "latency_or_occupancy",
            "utilization": occupancy,
            "headroom_pct": _headroom(max(sm, dram, occupancy)),
            "occupancy_pct": occupancy,
            "top_stalls": _top_stalls(metrics),
        }
    return {
        "type": "mixed_or_unknown",
        "utilization": max(sm, dram, compute_memory, occupancy),
        "headroom_pct": _headroom(max(sm, dram, compute_memory, occupancy)),
    }


def _secondary_opportunity(classification: BottleneckClass, saturation: dict[str, Any]) -> dict[str, Any]:
    saturated = bool(saturation.get("is_saturated"))
    resource = saturation.get("resource")
    if saturated and resource == "dram":
        return {"type": "compute_density", "reason": "memory bandwidth is near peak; increase reuse or reduce bytes moved"}
    if saturated and resource == "compute":
        return {"type": "memory_feeding_or_algorithmic_reduction", "reason": "compute pipeline is near peak; reduce work or remove memory stalls"}
    if saturated and resource == "occupancy":
        return {"type": "ilp_or_bank_conflict", "reason": "high occupancy with low throughput suggests issue efficiency problems"}
    if classification == "memory_bound":
        return {"type": "memory_access_pattern", "reason": "improve coalescing, cache locality, or tile reuse"}
    if classification == "compute_bound":
        return {"type": "pipeline_balance", "reason": "check instruction mix and scheduler pressure"}
    if classification == "latency_bound":
        return {"type": "latency_hiding", "reason": "reduce barriers, register pressure, or shared-memory conflicts"}
    return {"type": "collect_more_signal", "reason": "no dominant bottleneck was identified"}


def _recommendation_hint(
    classification: BottleneckClass,
    saturation: dict[str, Any],
    primary: dict[str, Any],
    secondary: dict[str, Any],
) -> str:
    if saturation.get("is_saturated") and saturation.get("resource") == "dram":
        utilization = saturation.get("utilization", 0.0)
        headroom = saturation.get("headroom_pct", 0.0)
        return f"DRAM 已接近峰值({utilization:.1f}%),进一步提升带宽空间约 {headroom:.1f}%。建议提升算术强度、增大 tile reuse、融合后续算子或减少数据搬运。"
    if saturation.get("is_saturated") and saturation.get("resource") == "compute":
        return "计算管线已接近峰值,继续堆计算优化收益有限。建议检查内存供给、减少无效计算或考虑算法级近似/稀疏化。"
    if saturation.get("is_saturated") and saturation.get("resource") == "occupancy":
        return "Occupancy 已高但吞吐双低,优先检查 ILP、bank conflict、barrier 和 scoreboard stall,不要继续单纯增加并发。"
    if classification == "memory_bound":
        return "当前主要受内存路径限制。优先检查访存合并、L1/L2 命中、global memory 等待和 tile reuse。"
    if classification == "compute_bound":
        return "当前主要受计算管线限制。优先检查 FMA/Tensor Core 利用率、指令 mix 和 scheduler 饱和。"
    if classification == "latency_bound":
        return "当前更像 latency/occupancy 问题。优先降低寄存器/共享内存压力、减少同步等待并提升 issue efficiency。"
    return f"瓶颈信号不够单一,先围绕 {primary.get('type')} 和 {secondary.get('type')} 做小步试验。"


def _actionable_metric_values(metrics: NcuMetrics) -> dict[str, Any]:
    values = _metric_values(metrics)
    selected = [
        METRIC_GPU_TIME,
        METRIC_DRAM_BYTES_READ,
        METRIC_DRAM_BYTES_WRITE,
        METRIC_L1_HIT_RATE,
        METRIC_L2_HIT_RATE,
        METRIC_FMA_PIPE,
        METRIC_TENSOR_PIPE,
        METRIC_FP16_INST,
        METRIC_FFMA_INST,
    ]
    return {name: values[name] for name in selected if name in values}


def _top_stalls(metrics: NcuMetrics, limit: int = 3) -> dict[str, float]:
    return dict(sorted(metrics.stall_reasons.items(), key=lambda item: -item[1])[:limit])


def _metric_values(metrics: NcuMetrics) -> dict[str, Any]:
    values = metrics.extra.get("metrics", {}) if metrics.extra else {}
    return values if isinstance(values, dict) else {}


def _metric_float(metrics: NcuMetrics, metric_name: str) -> float | None:
    values = _metric_values(metrics)
    value = values.get(metric_name)
    if value is None:
        for alias in METRIC_ALIASES.get(metric_name, []):
            if alias in values:
                value = values[alias]
                break
    if value is None:
        value = _field_value_for_metric(metrics, metric_name)
    try:
        return float(value) if value is not None and value != "" else None
    except (TypeError, ValueError):
        return None


def _metric_int(metrics: NcuMetrics, metric_name: str) -> int | None:
    value = _metric_float(metrics, metric_name)
    return int(value) if value is not None else None


def _field_value_for_metric(metrics: NcuMetrics, metric_name: str) -> float | int | None:
    field_map = {
        METRIC_SM_THROUGHPUT: metrics.sm_throughput_pct,
        METRIC_COMPUTE_MEMORY_THROUGHPUT: metrics.compute_memory_throughput_pct,
        METRIC_L1_THROUGHPUT: metrics.l1_throughput_pct,
        METRIC_L2_THROUGHPUT: metrics.l2_throughput_pct,
        METRIC_DRAM_THROUGHPUT: metrics.dram_throughput_pct,
        METRIC_OCCUPANCY: metrics.occupancy_pct,
        METRIC_SHARED_MEM_PER_BLOCK: metrics.shared_mem_per_block_bytes,
        METRIC_REGISTERS_PER_THREAD: metrics.registers_per_thread,
    }
    return field_map.get(metric_name)


def _append_metric_line(lines: list[str], label: str, value: float | None, suffix: str) -> None:
    if value is not None:
        lines.append(f"{label}: {value:.1f}{suffix}")


def _empty_saturation() -> dict[str, Any]:
    return {
        "resource": None,
        "utilization": 0.0,
        "headroom_pct": 100.0,
        "is_saturated": False,
        "reason": "",
        "phase3_metrics": [],
    }


def _headroom(utilization: float) -> float:
    return max(0.0, 100.0 - utilization)


def _dedupe_metrics(metrics: list[str]) -> list[str]:
    return list(dict.fromkeys(metrics))


def _phase_output_path(output_report_path: Path, phase_name: str) -> Path:
    suffix = output_report_path.suffix or ".txt"
    return output_report_path.with_name(f"{output_report_path.stem}.{phase_name}{suffix}")


def _error_metrics(phase_name: str, requested_metrics: list[str], raw_text: str) -> NcuMetrics:
    metrics = NcuMetrics(raw_text=raw_text)
    metrics.extra.update({
        "phase": phase_name,
        "requested_metrics": requested_metrics,
        "parse_error": raw_text,
    })
    return metrics


def _write_combined_report(output_report_path: Path, metrics: NcuMetrics) -> None:
    try:
        output_report_path.parent.mkdir(parents=True, exist_ok=True)
        output_report_path.write_text(format_ncu_for_prompt(metrics), encoding="utf-8")
    except Exception as e:
        logger.warning("Could not write combined ncu report %s: %s", output_report_path, e)
