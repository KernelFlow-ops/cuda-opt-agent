"""
Phase 6 测试 —— TUI 组件。
"""

import pytest
from io import StringIO
from rich.console import Console


class TestWidgets:
    def test_dashboard_panel(self, sample_run_state):
        from cuda_opt_agent.tui.widgets import build_dashboard_panel
        panel = build_dashboard_panel(sample_run_state)
        assert panel is not None
        # 渲染不报错即可
        console = Console(file=StringIO())
        console.print(panel)

    def test_history_table(self, sample_run_state):
        from cuda_opt_agent.tui.widgets import build_history_table
        table = build_history_table(sample_run_state)
        assert table is not None
        console = Console(file=StringIO())
        console.print(table)

    def test_progress_bar(self, sample_run_state):
        from cuda_opt_agent.tui.widgets import build_progress_bar
        progress = build_progress_bar(sample_run_state)
        assert progress is not None

    def test_empty_state_dashboard(self):
        from cuda_opt_agent.models.data import RunState, OperatorSpec, AgentConfig
        state = RunState(
            run_id="test",
            operator_spec=OperatorSpec(name="test", signature=""),
            config=AgentConfig(),
        )
        from cuda_opt_agent.tui.widgets import build_dashboard_panel
        panel = build_dashboard_panel(state)
        console = Console(file=StringIO())
        console.print(panel)


class TestLiveStream:
    def test_create(self):
        from cuda_opt_agent.tui.live import LiveReasoningStream
        stream = LiveReasoningStream(Console(file=StringIO()))
        assert stream is not None

    def test_node_colors(self):
        from cuda_opt_agent.tui.live import NODE_COLORS
        assert "analyze" in NODE_COLORS
        assert "decide" in NODE_COLORS
        assert "reflect" in NODE_COLORS


class TestCudaOptApp:
    def test_create(self):
        from cuda_opt_agent.tui.app import CudaOptApp
        app = CudaOptApp()
        assert app is not None

    def test_show_dashboard(self, sample_run_state):
        from cuda_opt_agent.tui.app import CudaOptApp
        app = CudaOptApp()
        app.console = Console(file=StringIO())
        app.show_dashboard(sample_run_state)

    def test_show_history(self, sample_run_state):
        from cuda_opt_agent.tui.app import CudaOptApp
        app = CudaOptApp()
        app.console = Console(file=StringIO())
        app.show_history(sample_run_state)

    def test_print_welcome(self):
        from cuda_opt_agent.tui.app import CudaOptApp
        app = CudaOptApp()
        app.console = Console(file=StringIO())
        app.print_welcome()

    def test_final_report(self, sample_run_state):
        from cuda_opt_agent.tui.app import CudaOptApp
        app = CudaOptApp()
        app.console = Console(file=StringIO())
        app.print_final_report(sample_run_state)
