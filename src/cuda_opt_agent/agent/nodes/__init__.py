from __future__ import annotations

from ...memory.knowledge import KnowledgeBase
from ...memory.run_state import RunStateManager
from ..llm_client import LLMClient
from ._helpers import (
    GpuPool,
    _active_shape_profiles,
    _benchmark_multi,
    _build_code_diff_context,
    _compile_hp_candidate_job,
    _compile_hp_candidates,
    _compile_hp_candidates_async,
    _iter_compile_hp_candidates_async,
    _generate_code_diff,
    _generate_final_report,
    _hardware_summary,
    _hp_compile_worker_count,
    _hyperparams_text,
    _iteration_outcome_text,
    _kernel_executable,
    _method_history_text,
    _operator_context,
    _per_shape_summary,
    _profile_args_from_benchmark,
    _read_seed_code,
    _selected_hyperparams,
)
from .analyze import analyze_node
from .apply_direct import apply_direct_node
from .bootstrap import bootstrap_node
from .compare_library import compare_library_node  # [改进] 新增
from .compile_validate import _repair_code, compile_and_validate_node
from .decide import decide_node
from .evaluate import evaluate_node
from .hp_search import hp_search_node, _smart_truncate_code, _build_correctness_failure_history
from .init import init_node
from .profile import profile_best_node
from .reflect import reflect_node
from .terminate import terminate_node


class AgentNodes:
    """
    Thin facade used by graph.py. Actual node implementations live in sibling modules.
    """

    def __init__(
        self,
        state_manager: RunStateManager,
        kb: KnowledgeBase,
        llm: LLMClient,
        stream_sink=None,
    ):
        self.sm = state_manager
        self.kb = kb
        self.llm = llm
        self.stream_sink = stream_sink

    _operator_context = staticmethod(_operator_context)
    _read_seed_code = staticmethod(_read_seed_code)
    _active_shape_profiles = staticmethod(_active_shape_profiles)
    _benchmark_multi = _benchmark_multi
    _profile_args_from_benchmark = _profile_args_from_benchmark
    _per_shape_summary = staticmethod(_per_shape_summary)
    _iteration_outcome_text = staticmethod(_iteration_outcome_text)
    _hyperparams_text = staticmethod(_hyperparams_text)
    _method_history_text = _method_history_text
    _selected_hyperparams = staticmethod(_selected_hyperparams)
    _hp_compile_worker_count = _hp_compile_worker_count
    _compile_hp_candidates = _compile_hp_candidates
    _hardware_summary = _hardware_summary
    _kernel_executable = staticmethod(_kernel_executable)
    _generate_final_report = _generate_final_report

    # [优化] 新增
    _build_code_diff_context = staticmethod(_build_code_diff_context)
    _generate_code_diff = staticmethod(_generate_code_diff)
    _iter_compile_hp_candidates_async = staticmethod(_iter_compile_hp_candidates_async)

    init_node = init_node
    bootstrap_node = bootstrap_node
    compare_library_node = compare_library_node  # [改进] 新增
    compile_and_validate_node = compile_and_validate_node
    _repair_code = _repair_code
    profile_best_node = profile_best_node
    analyze_node = analyze_node
    decide_node = decide_node
    hp_search_node = hp_search_node
    apply_direct_node = apply_direct_node
    evaluate_node = evaluate_node
    reflect_node = reflect_node
    terminate_node = terminate_node


__all__ = ["AgentNodes", "_compile_hp_candidate_job", "_iter_compile_hp_candidates_async", "GpuPool"]
