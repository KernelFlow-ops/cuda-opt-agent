"""
CLI 入口 —— 使用 Typer 构建命令行工具。

用法:
    cuda-opt new softmax --spec tasks/softmax_fp16.yaml
    cuda-opt new softmax --task "写一个 fp16 softmax,沿最后一维归一化"
    cuda-opt tune kernels/fused_attention.cu
    cuda-opt run gemm --shape 4096,4096,4096 --dtype fp16 --max-iters 30  # 兼容入口
    cuda-opt resume gemm
    cuda-opt list
    cuda-opt show runs/gemm_run_20260501T120000
"""

from __future__ import annotations

import difflib
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from .config import load_config
from .models.data import AgentConfig, OperatorSpec
from .shape_profiles import default_shape_profiles, dims_to_profile, parse_shape_profiles
from .task_spec import load_operator_spec, resolve_existing_cuda_path


def _normalize_console_encoding(value: str) -> str:
    normalized = value.strip().lower().replace("_", "-")
    aliases = {
        "utf8": "utf-8",
        "utf-8": "utf-8",
        "gbk": "gbk",
        "cp936": "gbk",
    }
    if normalized not in aliases:
        raise ValueError("CONSOLE_ENCODING must be auto, utf-8, gbk, or default")
    return aliases[normalized]


def _auto_console_encoding() -> str | None:
    if os.name != "nt":
        return None
    capture_env = any(os.getenv(name) for name in ("CI", "PYCHARM_HOSTED", "VSCODE_PID"))
    if capture_env or not sys.stdout.isatty() or not sys.stderr.isatty():
        return "utf-8"
    return "gbk"


def _configure_console_encoding(mode: str | None = None) -> None:
    """Configure console stream encoding while keeping all file writes UTF-8."""
    requested = (mode or os.getenv("CONSOLE_ENCODING", "auto")).strip().lower()
    if requested in {"", "auto"}:
        encoding = _auto_console_encoding()
    elif requested in {"default", "none"}:
        return
    else:
        encoding = _normalize_console_encoding(requested)

    if encoding is None:
        return

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding=encoding, errors="replace")


_configure_console_encoding()

app = typer.Typer(
    name="cuda-opt",
    help="CUDA operator optimization agent - LLM-driven automated kernel optimization",
    add_completion=False,
)
console = Console(safe_box=True)


GEMM_FULL_SHAPE_CONSTRAINTS = [
    "Benchmark must use the requested full GEMM shape.",
    "Correctness check must run the kernel on the requested full GEMM shape and verify sampled output elements against a CPU reference for those elements; do not use a reduced validation shape as the only correctness check.",
]


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


def _apply_config_overrides(
    config: AgentConfig,
    *,
    dtype: str | None = None,
    max_iters: int | None = None,
    consecutive_reject_limit: int | None = None,
    accept_epsilon: float | None = None,
    decide_reselect_max_retries: int | None = None,
    hp_candidate_count: int | None = None,
    hp_compile_workers: int | None = None,
    multi_shape_aggregator: str | None = None,
) -> None:
    if dtype is not None:
        config.default_dtype = dtype
    if max_iters is not None:
        config.max_iterations = max_iters
    if consecutive_reject_limit is not None:
        config.consecutive_reject_limit = consecutive_reject_limit
    if accept_epsilon is not None:
        config.accept_epsilon = accept_epsilon
    if decide_reselect_max_retries is not None:
        config.decide_reselect_max_retries = decide_reselect_max_retries
    if hp_candidate_count is not None:
        config.hp_candidate_count = hp_candidate_count
    if hp_compile_workers is not None:
        config.hp_compile_workers = hp_compile_workers
    if multi_shape_aggregator is not None:
        config.multi_shape_aggregator = multi_shape_aggregator


def _parse_shape(shape: str | None) -> list[int]:
    if not shape:
        return []
    try:
        return [int(s.strip()) for s in shape.split(",") if s.strip()]
    except ValueError as e:
        raise typer.BadParameter("shape must be comma-separated integers") from e


def _generic_dtypes(dtype: str) -> dict[str, str]:
    return {"input": dtype, "output": dtype}


def _apply_dtype_to_spec(spec: OperatorSpec, dtype: str) -> None:
    if spec.dtypes:
        spec.dtypes = {name: dtype for name in spec.dtypes}
    else:
        spec.dtypes = _generic_dtypes(dtype)


def _profiles_from_cli(
    operator: str,
    *,
    shape_list: list[int],
    shapes: str | None,
    shape_profile: str | None,
) -> list[dict]:
    if shapes and shape_profile:
        raise ValueError("Use only one of --shapes or --shape-profile")
    if shapes:
        return parse_shape_profiles(operator, shapes)
    if shape_profile:
        return default_shape_profiles(operator, shape_profile)
    if shape_list:
        return [dims_to_profile(operator, shape_list)]
    return []


def _gemm_shapes_from_profile(profile: dict) -> dict[str, list[int]] | None:
    try:
        m = int(profile["M"])
        n = int(profile["N"])
        k = int(profile["K"])
    except (KeyError, TypeError, ValueError):
        return None
    return {"A": [m, k], "B": [k, n], "C": [m, n]}


def _operator_spec_from_fields(
    *,
    operator: str,
    signature: str | None,
    shape: str | None,
    dtype: str,
    shapes: str | None = None,
    shape_profile: str | None = None,
    task_description: str = "",
    seed_code_path: str | None = None,
) -> OperatorSpec:
    shape_list = _parse_shape(shape)
    profiles = _profiles_from_cli(operator, shape_list=shape_list, shapes=shapes, shape_profile=shape_profile)
    gemm_shapes = _gemm_shapes_from_profile(profiles[0]) if profiles else None
    if operator.lower() == "gemm" and gemm_shapes:
        m = gemm_shapes["A"][0]
        n = gemm_shapes["C"][1]
        k = gemm_shapes["A"][1]
        return OperatorSpec(
            name=operator,
            signature=signature or f"C[M,N] = A[M,K] @ B[K,N], M={m}, N={n}, K={k}",
            dtypes={"A": dtype, "B": dtype, "C": dtype},
            shapes=gemm_shapes,
            constraints=list(GEMM_FULL_SHAPE_CONSTRAINTS),
            task_description=task_description,
            seed_code_path=seed_code_path,
            shape_profiles=profiles,
        )

    shapes = {"input": shape_list} if shape_list else {}
    return OperatorSpec(
        name=operator,
        signature=signature or f"{operator} operation",
        dtypes=_generic_dtypes(dtype),
        shapes=shapes,
        task_description=task_description,
        seed_code_path=seed_code_path,
        shape_profiles=profiles,
    )


def _build_task_spec(
    operator: str | None,
    *,
    task: str,
    signature: str | None,
    shape: str | None,
    shapes: str | None,
    shape_profile: str | None,
    config: AgentConfig,
) -> OperatorSpec:
    if not operator:
        raise ValueError("--task mode requires an operator name, e.g. cuda-opt new softmax --task ...")
    return _operator_spec_from_fields(
        operator=operator,
        signature=signature,
        shape=shape,
        shapes=shapes,
        shape_profile=shape_profile,
        dtype=config.default_dtype,
        task_description=task,
    )


def _build_seed_spec(
    file_cu: str,
    *,
    operator: str | None,
    task: str | None,
    signature: str | None,
    shape: str | None,
    shapes: str | None,
    shape_profile: str | None,
    config: AgentConfig,
) -> OperatorSpec:
    seed_code_path = resolve_existing_cuda_path(file_cu)
    op_name = operator or Path(seed_code_path).stem
    return _operator_spec_from_fields(
        operator=op_name,
        signature=signature,
        shape=shape,
        shapes=shapes,
        shape_profile=shape_profile,
        dtype=config.default_dtype,
        task_description=task or f"Optimize existing CUDA implementation from {Path(seed_code_path).name}.",
        seed_code_path=seed_code_path,
    )


def _prompt_operator_spec(operator: str | None, config: AgentConfig) -> OperatorSpec:
    op_name = operator or typer.prompt("Operator name")
    signature = typer.prompt("Signature", default=f"{op_name} operation")
    dtype = typer.prompt("Dtype", default=config.default_dtype)
    shape = typer.prompt("Comma-separated shape (optional)", default="")
    task_description = typer.prompt("Task description (optional)", default="")
    constraints_text = typer.prompt("Constraints (optional; separate with ';')", default="")

    spec = _operator_spec_from_fields(
        operator=op_name,
        signature=signature,
        shape=shape,
        dtype=dtype,
        task_description=task_description,
    )
    spec.constraints.extend([c.strip() for c in constraints_text.split(";") if c.strip()])
    return spec


def _load_spec_mode(spec_file: str, operator: str | None, dtype_override: str | None, config: AgentConfig) -> OperatorSpec:
    spec = load_operator_spec(spec_file)
    if operator and spec.name != operator:
        raise ValueError(f"Operator argument '{operator}' does not match spec name '{spec.name}'")
    if dtype_override is not None:
        _apply_dtype_to_spec(spec, dtype_override)
    elif not spec.dtypes:
        spec.dtypes = _generic_dtypes(config.default_dtype)
    return spec


def _run_task(op_spec: OperatorSpec, *, config: AgentConfig) -> None:
    console.print(f"[bold cyan]Starting optimization: {op_spec.name}[/bold cyan]")
    if op_spec.signature:
        console.print(f"  Signature: {op_spec.signature}")
    if op_spec.shapes:
        console.print(f"  Shapes: {op_spec.shapes}")
    if op_spec.shape_profiles:
        console.print(f"  Shape profiles: {len(op_spec.shape_profiles)}")
    if op_spec.dtypes:
        console.print(f"  Dtypes: {op_spec.dtypes}")
    if op_spec.task_description:
        short_task = op_spec.task_description.replace("\n", " ")[:120]
        console.print(f"  Task: {short_task}")
    if op_spec.seed_code_path:
        console.print(f"  Seed code: {op_spec.seed_code_path}")
    console.print(f"  Max iterations: {config.max_iterations}")

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


def _load_config_with_overrides(
    env_file: str | None,
    *,
    dtype: str | None,
    max_iters: int | None,
    consecutive_reject_limit: int | None,
    accept_epsilon: float | None,
    decide_reselect_max_retries: int | None,
    hp_candidate_count: int | None,
    hp_compile_workers: int | None,
    multi_shape_aggregator: str | None,
) -> AgentConfig:
    config = load_config(env_file)
    _configure_console_encoding()
    _apply_config_overrides(
        config,
        dtype=dtype,
        max_iters=max_iters,
        consecutive_reject_limit=consecutive_reject_limit,
        accept_epsilon=accept_epsilon,
        decide_reselect_max_retries=decide_reselect_max_retries,
        hp_candidate_count=hp_candidate_count,
        hp_compile_workers=hp_compile_workers,
        multi_shape_aggregator=multi_shape_aggregator,
    )
    return config


@app.command()
def new(
    operator: Optional[str] = typer.Argument(None, help="Operator name, e.g. gemm, softmax, conv2d"),
    spec_file: Optional[str] = typer.Option(None, "--spec", help="YAML/TOML/JSON task spec file"),
    task: Optional[str] = typer.Option(None, "--task", help="Natural language task description"),
    from_cu: Optional[str] = typer.Option(None, "--from-cu", help="Existing .cu file to use as v0 seed"),
    signature: Optional[str] = typer.Option(None, "--sig", help="Operator signature, e.g. 'C = A @ B'"),
    shape: Optional[str] = typer.Option(None, "--shape", help="Comma-separated shape"),
    shapes: Optional[str] = typer.Option(None, "--shapes", help="Semicolon-separated shape profiles, e.g. '1024^3;2048^3'"),
    shape_profile: Optional[str] = typer.Option(None, "--shape-profile", help="Default profile: small, medium, large, sweep"),
    dtype: Optional[str] = typer.Option(
        None, "--dtype", help="Data type (defaults to DEFAULT_DTYPE in .env, fallback fp16)"
    ),
    max_iters: Optional[int] = typer.Option(
        None, "--max-iters", help="Maximum iteration count (defaults to MAX_ITERATIONS in .env, fallback 30)"
    ),
    consecutive_reject_limit: Optional[int] = typer.Option(
        None,
        "--consecutive-reject-limit",
        help="Stop after this many consecutive rejects (defaults to CONSECUTIVE_REJECT_LIMIT in .env)",
    ),
    accept_epsilon: Optional[float] = typer.Option(
        None,
        "--accept-epsilon",
        help="Required relative improvement to accept a trial (defaults to ACCEPT_EPSILON in .env)",
    ),
    decide_reselect_max_retries: Optional[int] = typer.Option(
        None,
        "--decide-reselect-max-retries",
        help="Reselect when decide picks a blacklisted method (defaults to DECIDE_RESELECT_MAX_RETRIES in .env)",
    ),
    hp_candidate_count: Optional[int] = typer.Option(
        None,
        "--hp-candidate-count",
        help="Hyperparameter candidates per search (defaults to HP_CANDIDATE_COUNT in .env)",
    ),
    hp_compile_workers: Optional[int] = typer.Option(
        None,
        "--hp-compile-workers",
        help="Parallel HP candidate compile workers (defaults to HP_COMPILE_WORKERS in .env; 0=auto, 1=serial)",
    ),
    multi_shape_aggregator: Optional[str] = typer.Option(
        None,
        "--multi-shape-aggregator",
        help="Aggregate multi-shape latency with mean, worst, or weighted",
    ),
    env_file: Optional[str] = typer.Option(None, "--env", help="Path to .env file"),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
    log_file: Optional[str] = typer.Option(None, "--log-file", help="UTF-8 log file path"),
    auto: bool = typer.Option(False, "--auto", help="Do not enter the interactive task wizard"),
) -> None:
    """Start a new optimization task from a spec file, text task, seed code, or wizard."""
    _setup_logging(log_level, log_file)
    config = _load_config_with_overrides(
        env_file,
        dtype=dtype,
        max_iters=max_iters,
        consecutive_reject_limit=consecutive_reject_limit,
        accept_epsilon=accept_epsilon,
        decide_reselect_max_retries=decide_reselect_max_retries,
        hp_candidate_count=hp_candidate_count,
        hp_compile_workers=hp_compile_workers,
        multi_shape_aggregator=multi_shape_aggregator,
    )

    modes = [spec_file is not None, task is not None, from_cu is not None]
    if sum(modes) > 1:
        console.print("[red]Use only one input mode: --spec, --task, or --from-cu[/red]")
        raise typer.Exit(1)

    try:
        if spec_file:
            op_spec = _load_spec_mode(spec_file, operator, dtype, config)
        elif task:
            op_spec = _build_task_spec(
                operator,
                task=task,
                signature=signature,
                shape=shape,
                shapes=shapes,
                shape_profile=shape_profile,
                config=config,
            )
        elif from_cu:
            op_spec = _build_seed_spec(
                from_cu,
                operator=operator,
                task=None,
                signature=signature,
                shape=shape,
                shapes=shapes,
                shape_profile=shape_profile,
                config=config,
            )
        elif auto:
            console.print("[red]Specify --spec, --task, or --from-cu when --auto is enabled[/red]")
            raise typer.Exit(1)
        else:
            op_spec = _prompt_operator_spec(operator, config)
    except (FileNotFoundError, RuntimeError, ValueError, typer.BadParameter) as e:
        console.print(f"[red]Invalid task input: {e}[/red]")
        raise typer.Exit(1)

    _run_task(op_spec, config=config)


@app.command()
def tune(
    file_cu: str = typer.Argument(..., help="Existing .cu file to optimize"),
    operator: Optional[str] = typer.Option(None, "--operator", help="Operator name; defaults to file stem"),
    task: Optional[str] = typer.Option(None, "--task", help="Additional natural language task context"),
    signature: Optional[str] = typer.Option(None, "--sig", help="Operator signature"),
    shape: Optional[str] = typer.Option(None, "--shape", help="Comma-separated shape"),
    shapes: Optional[str] = typer.Option(None, "--shapes", help="Semicolon-separated shape profiles, e.g. '1024^3;2048^3'"),
    shape_profile: Optional[str] = typer.Option(None, "--shape-profile", help="Default profile: small, medium, large, sweep"),
    dtype: Optional[str] = typer.Option(
        None, "--dtype", help="Data type (defaults to DEFAULT_DTYPE in .env, fallback fp16)"
    ),
    max_iters: Optional[int] = typer.Option(
        None, "--max-iters", help="Maximum iteration count (defaults to MAX_ITERATIONS in .env, fallback 30)"
    ),
    consecutive_reject_limit: Optional[int] = typer.Option(
        None,
        "--consecutive-reject-limit",
        help="Stop after this many consecutive rejects (defaults to CONSECUTIVE_REJECT_LIMIT in .env)",
    ),
    accept_epsilon: Optional[float] = typer.Option(
        None,
        "--accept-epsilon",
        help="Required relative improvement to accept a trial (defaults to ACCEPT_EPSILON in .env)",
    ),
    decide_reselect_max_retries: Optional[int] = typer.Option(
        None,
        "--decide-reselect-max-retries",
        help="Reselect when decide picks a blacklisted method (defaults to DECIDE_RESELECT_MAX_RETRIES in .env)",
    ),
    hp_candidate_count: Optional[int] = typer.Option(
        None,
        "--hp-candidate-count",
        help="Hyperparameter candidates per search (defaults to HP_CANDIDATE_COUNT in .env)",
    ),
    hp_compile_workers: Optional[int] = typer.Option(
        None,
        "--hp-compile-workers",
        help="Parallel HP candidate compile workers (defaults to HP_COMPILE_WORKERS in .env; 0=auto, 1=serial)",
    ),
    multi_shape_aggregator: Optional[str] = typer.Option(
        None,
        "--multi-shape-aggregator",
        help="Aggregate multi-shape latency with mean, worst, or weighted",
    ),
    env_file: Optional[str] = typer.Option(None, "--env", help="Path to .env file"),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
    log_file: Optional[str] = typer.Option(None, "--log-file", help="UTF-8 log file path"),
) -> None:
    """Optimize an existing .cu implementation as the v0 baseline."""
    _setup_logging(log_level, log_file)
    config = _load_config_with_overrides(
        env_file,
        dtype=dtype,
        max_iters=max_iters,
        consecutive_reject_limit=consecutive_reject_limit,
        accept_epsilon=accept_epsilon,
        decide_reselect_max_retries=decide_reselect_max_retries,
        hp_candidate_count=hp_candidate_count,
        hp_compile_workers=hp_compile_workers,
        multi_shape_aggregator=multi_shape_aggregator,
    )
    try:
        op_spec = _build_seed_spec(
            file_cu,
            operator=operator,
            task=task,
            signature=signature,
            shape=shape,
            shapes=shapes,
            shape_profile=shape_profile,
            config=config,
        )
    except (FileNotFoundError, RuntimeError, ValueError, typer.BadParameter) as e:
        console.print(f"[red]Invalid task input: {e}[/red]")
        raise typer.Exit(1)

    _run_task(op_spec, config=config)


@app.command()
def run(
    operator: str = typer.Argument(..., help="Operator name, e.g. gemm, softmax, conv2d"),
    signature: str = typer.Option("", "--sig", help="Operator signature, e.g. 'C = A @ B'"),
    shape: str = typer.Option("4096,4096,4096", "--shape", help="Comma-separated shape"),
    shapes: Optional[str] = typer.Option(None, "--shapes", help="Semicolon-separated shape profiles, e.g. '1024^3;2048^3'"),
    shape_profile: Optional[str] = typer.Option(None, "--shape-profile", help="Default profile: small, medium, large, sweep"),
    dtype: Optional[str] = typer.Option(
        None, "--dtype", help="Data type (defaults to DEFAULT_DTYPE in .env, fallback fp16)"
    ),
    max_iters: Optional[int] = typer.Option(
        None, "--max-iters", help="Maximum iteration count (defaults to MAX_ITERATIONS in .env, fallback 30)"
    ),
    consecutive_reject_limit: Optional[int] = typer.Option(
        None,
        "--consecutive-reject-limit",
        help="Stop after this many consecutive rejects (defaults to CONSECUTIVE_REJECT_LIMIT in .env)",
    ),
    accept_epsilon: Optional[float] = typer.Option(
        None,
        "--accept-epsilon",
        help="Required relative improvement to accept a trial (defaults to ACCEPT_EPSILON in .env)",
    ),
    decide_reselect_max_retries: Optional[int] = typer.Option(
        None,
        "--decide-reselect-max-retries",
        help="Reselect when decide picks a blacklisted method (defaults to DECIDE_RESELECT_MAX_RETRIES in .env)",
    ),
    hp_candidate_count: Optional[int] = typer.Option(
        None,
        "--hp-candidate-count",
        help="Hyperparameter candidates per search (defaults to HP_CANDIDATE_COUNT in .env)",
    ),
    hp_compile_workers: Optional[int] = typer.Option(
        None,
        "--hp-compile-workers",
        help="Parallel HP candidate compile workers (defaults to HP_COMPILE_WORKERS in .env; 0=auto, 1=serial)",
    ),
    multi_shape_aggregator: Optional[str] = typer.Option(
        None,
        "--multi-shape-aggregator",
        help="Aggregate multi-shape latency with mean, worst, or weighted",
    ),
    env_file: Optional[str] = typer.Option(None, "--env", help="Path to .env file"),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
    log_file: Optional[str] = typer.Option(None, "--log-file", help="UTF-8 log file path"),
    auto: bool = typer.Option(False, "--auto", help="Run without interactive intervention"),
) -> None:
    """Compatibility entrypoint; prefer `cuda-opt new` for new tasks."""
    _setup_logging(log_level, log_file)
    config = _load_config_with_overrides(
        env_file,
        dtype=dtype,
        max_iters=max_iters,
        consecutive_reject_limit=consecutive_reject_limit,
        accept_epsilon=accept_epsilon,
        decide_reselect_max_retries=decide_reselect_max_retries,
        hp_candidate_count=hp_candidate_count,
        hp_compile_workers=hp_compile_workers,
        multi_shape_aggregator=multi_shape_aggregator,
    )
    op_spec = _operator_spec_from_fields(
        operator=operator,
        signature=signature,
        shape=shape,
        shapes=shapes,
        shape_profile=shape_profile,
        dtype=config.default_dtype,
    )
    _run_task(op_spec, config=config)


@app.command()
def resume(
    target: Optional[str] = typer.Argument(None, help="Operator name, run directory, or run id"),
    run_dir: Optional[str] = typer.Option(None, "--run-dir", help="Run directory to resume"),
    extra_iters: int = typer.Option(0, "--extra-iters", help="Additional iterations"),
    env_file: Optional[str] = typer.Option(None, "--env", help="Path to .env file"),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
    log_file: Optional[str] = typer.Option(None, "--log-file", help="UTF-8 log file path"),
) -> None:
    """Resume an existing optimization run."""
    _setup_logging(log_level, log_file)
    config = load_config(env_file)
    _configure_console_encoding()
    if extra_iters > 0:
        config.max_iterations += extra_iters

    resolved_run_dir = run_dir
    operator = None
    if target and not resolved_run_dir:
        target_path = Path(target)
        runs_target_path = Path(config.runs_dir) / target
        if target_path.exists():
            resolved_run_dir = str(target_path)
        elif runs_target_path.exists():
            resolved_run_dir = str(runs_target_path)
        else:
            operator = target

    if operator and not resolved_run_dir:
        from .memory.persistence import PersistenceManager
        pm = PersistenceManager(config.runs_dir)
        latest = pm.find_latest_unfinished_run(operator)
        if latest:
            resolved_run_dir = str(latest)

    if not resolved_run_dir:
        console.print("[red]Specify an operator name, run id, or run directory[/red]")
        raise typer.Exit(1)

    from .agent.graph import run_optimization
    from .tui.app import CudaOptApp

    tui = CudaOptApp()
    tui.print_welcome()

    try:
        final_state = run_optimization(
            operator_spec=OperatorSpec(name=operator or "", signature=""),
            config=config,
            resume_dir=resolved_run_dir,
        )
        if final_state:
            tui.print_final_report(final_state)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user. State saved.[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Resume failed: {e}[/red]")
        raise typer.Exit(1)


def _list_runs_impl(runs_dir: str) -> None:
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


@app.command("list-runs", hidden=True)
def list_runs(
    runs_dir: str = typer.Option("runs", "--dir", help="Runs directory"),
) -> None:
    """List all runs."""
    _list_runs_impl(runs_dir)


@app.command("list")
def list_command(
    runs_dir: str = typer.Option("runs", "--dir", help="Runs directory"),
) -> None:
    """List all runs."""
    _list_runs_impl(runs_dir)


def _show_run_impl(run_dir: str) -> None:
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


@app.command("show-run", hidden=True)
def show_run(
    run_dir: str = typer.Argument(..., help="Run directory path"),
) -> None:
    """Show run details."""
    _show_run_impl(run_dir)


@app.command("show")
def show_command(
    run_dir: str = typer.Argument(..., help="Run directory path"),
) -> None:
    """Show run details."""
    _show_run_impl(run_dir)


def _resolve_code_ref(run_dir: str, ref: str) -> Path:
    direct = Path(ref)
    if direct.exists():
        return direct

    run_path = Path(run_dir)
    candidate = run_path / f"iter{ref}" / "code.cu"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not resolve code ref '{ref}' in {run_path}")


@app.command("diff")
def diff_command(
    run_dir: str = typer.Argument(..., help="Run directory path"),
    left: str = typer.Argument(..., help="Left version id or code path"),
    right: str = typer.Argument(..., help="Right version id or code path"),
) -> None:
    """Show a unified diff between two code versions."""
    try:
        left_path = _resolve_code_ref(run_dir, left)
        right_path = _resolve_code_ref(run_dir, right)
        diff = difflib.unified_diff(
            left_path.read_text(encoding="utf-8").splitlines(),
            right_path.read_text(encoding="utf-8").splitlines(),
            fromfile=str(left_path),
            tofile=str(right_path),
            lineterm="",
        )
        console.print("\n".join(diff) or "[green]No differences[/green]")
    except Exception as e:
        console.print(f"[red]Diff failed: {e}[/red]")
        raise typer.Exit(1)


def main() -> None:
    """CLI 入口点。"""
    app()


if __name__ == "__main__":
    main()
