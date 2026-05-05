"""
LangGraph 状态机 —— 编排整个迭代优化流程。
对应技术总纲 §2.3 的状态图。
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from ..config import get_api_key, load_config
from ..memory.knowledge import KnowledgeBase
from ..memory.run_state import RunStateManager
from ..models.data import AgentConfig, HardwareSpec, OperatorSpec, RunState, RunStatus
from .llm_client import LLMClient
from .nodes import AgentNodes
from .state import GraphState

logger = logging.getLogger(__name__)


def build_graph(
    config: AgentConfig | None = None,
    state_manager: RunStateManager | None = None,
    kb: KnowledgeBase | None = None,
    llm: LLMClient | None = None,
    entry_point: str = "init",
) -> StateGraph:
    """
    构建 LangGraph 状态机。

    状态转移:
        [*] → init → bootstrap → compile_validate → profile_best → analyze
        analyze → decide
        decide → hp_search (含超参) / apply_direct (无超参)
        hp_search → evaluate
        apply_direct → evaluate
        evaluate → reflect
        reflect → terminate (满足停止条件) / profile_best (继续)
        terminate → [*]
    """
    if config is None:
        config = load_config()

    if state_manager is None:
        state_manager = RunStateManager(config)

    if kb is None:
        kb = KnowledgeBase(config.knowledge_base_dir)

    if llm is None:
        llm = LLMClient(provider=config.llm_provider, model=config.llm_model)

    nodes = AgentNodes(state_manager=state_manager, kb=kb, llm=llm)

    # ── 构建图 ──
    graph = StateGraph(GraphState)

    # 添加节点
    graph.add_node("init", nodes.init_node)
    graph.add_node("bootstrap", nodes.bootstrap_node)
    graph.add_node("compile_validate", nodes.compile_and_validate_node)
    graph.add_node("profile_best", nodes.profile_best_node)
    graph.add_node("analyze", nodes.analyze_node)
    graph.add_node("decide", nodes.decide_node)
    graph.add_node("hp_search", nodes.hp_search_node)
    graph.add_node("apply_direct", nodes.apply_direct_node)
    graph.add_node("evaluate", nodes.evaluate_node)
    graph.add_node("reflect", nodes.reflect_node)
    graph.add_node("terminate", nodes.terminate_node)

    # ── 边 ──
    graph.set_entry_point(entry_point)
    graph.add_edge("init", "bootstrap")
    graph.add_edge("bootstrap", "compile_validate")

    # compile_validate → profile_best (如果成功) 或 terminate (失败)
    graph.add_conditional_edges(
        "compile_validate",
        _route_after_compile,
        {"profile_best": "profile_best", "terminate": "terminate"},
    )

    graph.add_edge("profile_best", "analyze")
    graph.add_edge("analyze", "decide")

    # decide → hp_search / apply_direct / terminate (give_up)
    graph.add_conditional_edges(
        "decide",
        _route_after_decide,
        {"hp_search": "hp_search", "apply_direct": "apply_direct", "terminate": "terminate"},
    )

    graph.add_edge("hp_search", "evaluate")
    graph.add_edge("apply_direct", "evaluate")
    graph.add_edge("evaluate", "reflect")

    # reflect → profile_best (继续) / terminate (停止)
    graph.add_conditional_edges(
        "reflect",
        _route_after_reflect,
        {"profile_best": "profile_best", "terminate": "terminate"},
    )

    graph.add_edge("terminate", END)

    return graph


def _route_after_compile(state: GraphState) -> str:
    """编译成功 → profile; 失败 → terminate。"""
    if state.get("trial_compile_ok") and state.get("trial_correctness_ok"):
        return "profile_best"
    return "terminate"


def _route_after_decide(state: GraphState) -> str:
    """根据决策路由: 含超参 / 不含超参 / 放弃。"""
    if state.get("should_stop"):
        return "terminate"
    if state.get("has_hyperparams"):
        return "hp_search"
    return "apply_direct"


def _route_after_reflect(state: GraphState) -> str:
    """继续迭代或终止。"""
    if state.get("should_stop"):
        return "terminate"
    return "profile_best"


# ════════════════════════════════════════
# 高层入口
# ════════════════════════════════════════
def run_optimization(
    operator_spec: OperatorSpec,
    config: AgentConfig | None = None,
    resume_dir: str | None = None,
) -> RunState:
    """
    运行完整的优化流程。

    Args:
        operator_spec: 算子规格
        config: 配置(可选,默认从 .env 加载)
        resume_dir: 续跑目录(可选)

    Returns:
        最终 RunState
    """
    if config is None:
        config = load_config()

    state_manager = RunStateManager(config)
    kb = KnowledgeBase(config.knowledge_base_dir)
    llm = LLMClient(provider=config.llm_provider, model=config.llm_model)

    # 新建或续跑
    if resume_dir:
        run_state = state_manager.resume_run(run_dir=resume_dir)
        if run_state is None:
            raise RuntimeError(f"Could not resume from {resume_dir}")
    else:
        from ..tools.hardware import collect_hardware_info
        hw = collect_hardware_info()
        run_state = state_manager.init_new_run(operator_spec, hw)

    entry_point = "profile_best" if resume_dir and run_state.iterations else "init"
    graph = build_graph(
        config=config,
        state_manager=state_manager,
        kb=kb,
        llm=llm,
        entry_point=entry_point,
    )
    compiled = graph.compile()

    active_operator_spec = run_state.operator_spec if resume_dir else operator_spec

    # 初始状态
    initial_state: GraphState = {
        "operator_spec": active_operator_spec,
        "hardware_spec": run_state.hardware_spec,
        "run_state": run_state,
        "should_stop": False,
        "iteration_count": len(run_state.iterations),
    }

    # 如果是续跑且已有迭代,直接从 profile_best 继续。
    if run_state.iterations:
        logger.info("Resume mode: skipping bootstrap and starting from profile_best")

    # 运行
    try:
        final_state = compiled.invoke(initial_state)
        return state_manager.state
    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C); saving state...")
        if state_manager.state:
            state_manager.state.status = RunStatus.PAUSED
            state_manager._save()
        return state_manager.state
    except Exception as e:
        logger.error("Run failed with exception: %s", e, exc_info=True)
        state_manager.mark_failed()
        raise
