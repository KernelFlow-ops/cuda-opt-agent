"""
TUI 自定义 Widget —— Rich 面板组件。
"""

from __future__ import annotations

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text

from ..models.data import IterationRecord, RunState


def build_dashboard_panel(state: RunState) -> Panel:
    """构建主控面板。"""
    op = state.operator_spec
    hw = state.hardware_spec

    # 基础信息
    info_lines = [
        f"Operator: {op.name}  {list(op.dtypes.values())}  {list(op.shapes.values())}",
        f"Hardware: {hw.gpu_name}  {hw.compute_capability}  CUDA {hw.cuda_version}",
        f"Run:      {state.run_id}   [Iter {len(state.iterations)}/{state.config.max_iterations}]",
    ]

    # 性能概要
    v0 = state.iter_by_id("v0")
    best = state.iter_by_id(state.current_best_id)

    if v0 and v0.benchmark:
        info_lines.append(f"")
        info_lines.append(f"Baseline (v0)         {v0.benchmark.latency_ms_median:.3f} ms")

    if best and best.benchmark and best.version_id != "v0":
        speedup = v0.benchmark.latency_ms_median / best.benchmark.latency_ms_median if v0 and v0.benchmark else 0
        info_lines.append(
            f"Current Best ({best.version_id})     {best.benchmark.latency_ms_median:.3f} ms"
            f"     speedup {speedup:.2f}x"
        )

    # 最后一次尝试
    if len(state.iterations) > 1:
        last = state.iterations[-1]
        if last.benchmark:
            status = "accepted" if last.accepted else "rejected"
            info_lines.append(
                f"Last Trial  ({last.version_id})     {last.benchmark.latency_ms_median:.3f} ms"
                f"     {status}"
            )

    content = "\n".join(info_lines)
    return Panel(content, title="CUDA Opt Agent", border_style="blue")


def build_history_table(state: RunState) -> Table:
    """构建迭代历史表格。"""
    table = Table(title="Iteration History", show_lines=True)
    table.add_column("Version", style="cyan", width=12, overflow="fold")
    table.add_column("Method", style="green", width=30, overflow="fold")
    table.add_column("Hyperparams", width=20, overflow="fold")
    table.add_column("Latency (ms)", justify="right", width=14, overflow="fold")
    table.add_column("Speedup", justify="right", width=10, overflow="fold")
    table.add_column("Status", width=10, overflow="fold")

    v0_lat = None
    for it in state.iterations:
        if it.version_id == "v0" and it.benchmark:
            v0_lat = it.benchmark.latency_ms_median

    for it in state.iterations:
        lat_str = f"{it.benchmark.latency_ms_median:.4f}" if it.benchmark else "N/A"
        speedup_str = ""
        if it.benchmark and v0_lat and v0_lat > 0:
            speedup = v0_lat / it.benchmark.latency_ms_median
            speedup_str = f"{speedup:.2f}x"

        hp_str = ""
        if it.hyperparams:
            hp_str = str(it.hyperparams)[:18]

        status = "[green]yes[/green]" if it.accepted else "[red]no[/red]"

        table.add_row(
            it.version_id,
            it.method_name or "baseline",
            hp_str,
            lat_str,
            speedup_str,
            status,
        )

    return table


def build_progress_bar(state: RunState) -> Progress:
    """构建进度条。"""
    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    )
    total = state.config.max_iterations
    current = len(state.iterations)
    task = progress.add_task("Optimization Progress", total=total, completed=current)
    return progress
