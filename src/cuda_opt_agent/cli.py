"""
CLI 入口 —— 使用 Typer 构建命令行工具。

用法:
    cuda-opt run gemm --shape 4096,4096,4096 --dtype fp16 --max-iters 30
    cuda-opt resume gemm
    cuda-opt resume --run-dir runs/gemm_run_20260501T120000
    cuda-opt list-runs
    cuda-opt show-run runs/gemm_run_20260501T120000
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from .config import load_config
from .models.data import AgentConfig, OperatorSpec


def _configure_windows_console_encoding() -> None:
    """Use GBK for Windows console streams while keeping file writes explicit."""
    if os.name != "nt":
        return
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="gbk", errors="replace")


_configure_windows_console_encoding()

app = typer.Typer(
    name="cuda-opt",
    help="CUDA operator optimization agent - LLM-driven automated kernel optimization",
    add_completion=False,
)
console = Console(safe_box=True)


def _setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )


@app.command()
def run(
    operator: str = typer.Argument(..., help="Operator name, e.g. gemm, softmax, conv2d"),
    signature: str = typer.Option("", "--sig", help="Operator signature, e.g. 'C = A @ B'"),
    shape: str = typer.Option("4096,4096,4096", "--shape", help="Comma-separated shape"),
    dtype: str = typer.Option("fp16", "--dtype", help="Data type"),
    max_iters: int = typer.Option(30, "--max-iters", help="Maximum iteration count"),
    env_file: Optional[str] = typer.Option(None, "--env", help="Path to .env file"),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
    log_file: Optional[str] = typer.Option(None, "--log-file", help="UTF-8 log file path"),
    auto: bool = typer.Option(False, "--auto", help="Run without interactive intervention"),
) -> None:
    """Start a new optimization run."""
    _setup_logging(log_level, log_file)

    config = load_config(env_file)
    config.max_iterations = max_iters

    # 解析形状
    shape_list = [int(s.strip()) for s in shape.split(",")]

    # 构建算子规格
    if operator.lower() == "gemm" and len(shape_list) == 3:
        m, n, k = shape_list
        op_spec = OperatorSpec(
            name=operator,
            signature=signature or f"C[M,N] = A[M,K] @ B[K,N], M={m}, N={n}, K={k}",
            dtypes={"A": dtype, "B": dtype, "C": dtype},
            shapes={"A": [m, k], "B": [k, n], "C": [m, n]},
            constraints=[
                "Benchmark must use the requested full GEMM shape.",
                "Correctness check must run the kernel on the requested full GEMM shape and verify sampled output elements against a CPU reference for those elements; do not use a reduced validation shape as the only correctness check.",
            ],
        )
    else:
        op_spec = OperatorSpec(
            name=operator,
            signature=signature or f"{operator} operation",
            dtypes={"input": dtype, "output": dtype},
            shapes={"input": shape_list},
        )

    console.print(f"[bold cyan]Starting optimization: {operator}[/bold cyan]")
    console.print(f"  Shape: {shape_list}")
    console.print(f"  Dtype: {dtype}")
    console.print(f"  Max iterations: {max_iters}")

    from .tui.app import CudaOptApp
    tui = CudaOptApp()
    tui.print_welcome()

    from .agent.graph import run_optimization
    try:
        final_state = run_optimization(op_spec, config=config)
        if final_state:
            tui.print_final_report(final_state)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user. State saved; use resume to continue.[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Run failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def resume(
    operator: Optional[str] = typer.Argument(None, help="Operator name; finds latest unfinished run"),
    run_dir: Optional[str] = typer.Option(None, "--run-dir", help="Run directory to resume"),
    extra_iters: int = typer.Option(0, "--extra-iters", help="Additional iterations"),
    env_file: Optional[str] = typer.Option(None, "--env", help="Path to .env file"),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
    log_file: Optional[str] = typer.Option(None, "--log-file", help="UTF-8 log file path"),
) -> None:
    """Resume an existing optimization run."""
    _setup_logging(log_level, log_file)

    if not operator and not run_dir:
        console.print("[red]Specify an operator name or a run directory[/red]")
        raise typer.Exit(1)

    config = load_config(env_file)
    if extra_iters > 0:
        config.max_iterations += extra_iters

    from .agent.graph import run_optimization
    from .tui.app import CudaOptApp

    tui = CudaOptApp()
    tui.print_welcome()

    try:
        final_state = run_optimization(
            operator_spec=OperatorSpec(name=operator or "", signature=""),
            config=config,
            resume_dir=run_dir,
        )
        if final_state:
            tui.print_final_report(final_state)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user. State saved.[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Resume failed: {e}[/red]")
        raise typer.Exit(1)


@app.command("list-runs")
def list_runs(
    runs_dir: str = typer.Option("runs", "--dir", help="Runs directory"),
) -> None:
    """List all runs."""
    from rich.table import Table

    runs_path = Path(runs_dir)
    if not runs_path.exists():
        console.print("[yellow]No run records found[/yellow]")
        return

    table = Table(title="Runs")
    table.add_column("Run ID", style="cyan", overflow="fold")
    table.add_column("Status", width=10, overflow="fold")
    table.add_column("Iters", justify="right", width=8, overflow="fold")
    table.add_column("Best", width=8, overflow="fold")
    table.add_column("Updated At", width=22, overflow="fold")

    from .memory.persistence import PersistenceManager
    pm = PersistenceManager(runs_dir)

    for d in sorted(runs_path.iterdir()):
        if not d.is_dir():
            continue
        try:
            state = pm.load_state(d)
            status_color = {
                "running": "yellow",
                "paused": "blue",
                "done": "green",
                "failed": "red",
            }.get(state.status.value, "white")

            table.add_row(
                state.run_id,
                f"[{status_color}]{state.status.value}[/{status_color}]",
                str(len(state.iterations)),
                state.current_best_id or "-",
                state.updated_at[:19],
            )
        except Exception:
            table.add_row(d.name, "[red]?[/red]", "?", "?", "?")

    console.print(table)


@app.command("show-run")
def show_run(
    run_dir: str = typer.Argument(..., help="Run directory path"),
) -> None:
    """Show run details."""
    from .memory.persistence import PersistenceManager
    from .tui.app import CudaOptApp

    pm = PersistenceManager()
    run_path = Path(run_dir)

    try:
        state = pm.load_state(run_path)
        tui = CudaOptApp()
        tui.print_final_report(state)
    except Exception as e:
        console.print(f"[red]Failed to load run: {e}[/red]")
        raise typer.Exit(1)


def main() -> None:
    """CLI 入口点。"""
    app()


if __name__ == "__main__":
    main()
