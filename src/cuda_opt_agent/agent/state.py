"""Graph state definition."""
from __future__ import annotations
from typing import Any, TypedDict


class GraphState(TypedDict, total=False):
    """LangGraph state — all fields optional."""
    # Core
    operator_spec: Any
    hardware_spec: Any
    run_state: Any
    config: Any

    # Iteration control
    iteration_count: int
    max_iterations: int
    should_stop: bool
    stop_reason: str

    # NCU / Profile
    current_ncu: Any
    current_benchmark: Any
    ncu_profile: Any
    ncu_raw: str
    analysis_result: dict

    # Decision
    method_decision: Any
    chosen_method: str
    has_hyperparams: bool
    decide_rationale: str
    bottleneck_analysis: str

    # Code
    current_code: str
    new_code: str
    new_version_id: str
    bootstrap_code: str
    bootstrap_code_path: str
    current_best_code: str
    current_best_id: str
    trial_code: str
    trial_code_path: str

    # Compilation / Correctness
    trial_version_id: str
    trial_compile_ok: bool
    trial_correctness_ok: bool
    compile_error: str
    correctness_error: str
    hp_correctness_failures: list[dict[str, Any]]
    hp_all_compiled_ok: bool

    # Benchmark
    trial_benchmark: Any
    trial_ncu: Any
    trial_latency_ms: float
    trial_speedup: float
    trial_accepted: bool
    trial_notes: str

    # History
    reflection: dict
    iterations: list
    blacklist: list[str]
    error: str | None

    # Enhanced decide context
    effective_methods_list: list[str]
    external_knowledge: str | None
    consecutive_rejects: int

    # ref.py / benchmark
    ref_py_path: str
    benchmark_runner_path: str

    # Library comparison
    library_baseline_ms: float | None
    library_name: str | None

    # HP Search
    hp_candidates: list
    hp_results: list
