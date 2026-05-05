"""
实时推理流 —— 流式渲染 LLM 节点输出。
颜色编码: analyze=蓝, decide=黄, apply=绿, reflect=紫
"""

from __future__ import annotations

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

# 节点颜色映射
NODE_COLORS = {
    "analyze": "blue",
    "decide": "yellow",
    "apply_direct": "green",
    "hp_search": "green",
    "reflect": "magenta",
    "bootstrap": "cyan",
    "compile_validate": "white",
    "profile_best": "white",
    "evaluate": "red",
}


class LiveReasoningStream:
    """管理实时推理流展示。"""

    def __init__(self, console: Console | None = None):
        self.console = console or Console()
        self._live: Live | None = None
        self._current_node: str = ""
        self._buffer: str = ""

    def start(self) -> None:
        """开始 Live 渲染。"""
        self._live = Live(console=self.console, refresh_per_second=4)
        self._live.start()

    def stop(self) -> None:
        """停止 Live 渲染。"""
        if self._live:
            self._live.stop()
            self._live = None

    def set_node(self, node_name: str) -> None:
        """切换到新节点。"""
        self._current_node = node_name
        self._buffer = ""
        color = NODE_COLORS.get(node_name, "white")
        if self._live:
            self._live.update(
                Panel(
                    Text(f"{node_name} running...", style=f"bold {color}"),
                    title=f"[{color}]{node_name}[/{color}]",
                )
            )

    def update(self, text: str) -> None:
        """更新当前节点的输出。"""
        self._buffer = text
        color = NODE_COLORS.get(self._current_node, "white")
        if self._live:
            content = Markdown(text[:2000])
            self._live.update(
                Panel(
                    content,
                    title=f"[{color}]{self._current_node}[/{color}]",
                    border_style=color,
                )
            )

    def append(self, chunk: str) -> None:
        """流式追加文本。"""
        self._buffer += chunk
        self.update(self._buffer)

    def finish_node(self, summary: str = "") -> None:
        """完成当前节点。"""
        color = NODE_COLORS.get(self._current_node, "white")
        self.console.print(
            f"  [{color}]DONE {self._current_node}[/{color}]"
            + (f": {summary}" if summary else "")
        )
