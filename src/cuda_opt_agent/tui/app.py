"""
TUI 主应用 —— Rich + Textual 交互界面。
支持三种视图:主控面板、实时推理流、历史浏览。
"""

from __future__ import annotations

import logging

from rich.console import Console
from rich.panel import Panel

from ..models.data import RunState
from .live import LiveReasoningStream
from .widgets import build_dashboard_panel, build_history_table, build_progress_bar

logger = logging.getLogger(__name__)


class CudaOptApp:
    """
    TUI 主应用。
    提供 Dashboard、Live Reasoning、History 三种视图。
    """

    def __init__(self):
        self.console = Console(safe_box=True)
        self.live_stream = LiveReasoningStream(self.console)
        self._state: RunState | None = None

    def set_state(self, state: RunState) -> None:
        """更新当前状态。"""
        self._state = state

    def show_dashboard(self, state: RunState | None = None) -> None:
        """显示主控面板。"""
        state = state or self._state
        if state is None:
            self.console.print("[red]No run state available[/red]")
            return
        panel = build_dashboard_panel(state)
        self.console.print(panel)

    def show_history(self, state: RunState | None = None) -> None:
        """显示迭代历史表格。"""
        state = state or self._state
        if state is None:
            self.console.print("[red]No run state available[/red]")
            return
        table = build_history_table(state)
        self.console.print(table)

    def show_progress(self, state: RunState | None = None) -> None:
        """显示进度条。"""
        state = state or self._state
        if state is None:
            return
        progress = build_progress_bar(state)
        self.console.print(progress)

    def show_iteration_summary(self, iteration_record) -> None:
        """显示单次迭代的摘要。"""
        it = iteration_record
        status = "[green]ACCEPTED[/green]" if it.accepted else "[red]REJECTED[/red]"
        lat = f"{it.benchmark.latency_ms_median:.4f} ms" if it.benchmark else "N/A"

        self.console.print(Panel(
            f"Version: {it.version_id}\n"
            f"Method: {it.method_name or 'baseline'}\n"
            f"Latency: {lat}\n"
            f"Status: {status}",
            title=f"Iteration {it.version_id}",
            border_style="cyan",
        ))

    def print_welcome(self) -> None:
        """打印欢迎信息。"""
        self.console.print(Panel(
            "[bold cyan]CUDA Operator Optimization Agent[/bold cyan]\n"
            "LLM-driven automated CUDA kernel optimization\n"
            "-------------------------------------",
            border_style="blue",
        ))

    def print_final_report(self, state: RunState) -> None:
        """打印最终报告摘要。"""
        self.console.print("\n")
        self.show_dashboard(state)
        self.console.print("\n")
        self.show_history(state)

        v0 = state.iter_by_id("v0")
        best = state.iter_by_id(state.current_best_id)
        if v0 and v0.benchmark and best and best.benchmark:
            speedup = v0.benchmark.latency_ms_median / best.benchmark.latency_ms_median
            self.console.print(f"\n[bold green]Final speedup: {speedup:.2f}x[/bold green]")
            self.console.print(
                f"v0: {v0.benchmark.latency_ms_median:.4f} ms -> "
                f"{best.version_id}: {best.benchmark.latency_ms_median:.4f} ms"
            )
