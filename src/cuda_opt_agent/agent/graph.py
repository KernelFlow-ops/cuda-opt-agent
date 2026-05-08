"""
LangGraph 状态机 —— 编排整个迭代优化流程。
对应技术总纲 §2.3 的状态图。

[改进]:
  - 新增 compare_library 节点: bootstrap 后可选地对比 cuDNN/cuBLAS 基线
  - 状态转移:
      [*] → init → bootstrap → compile_validate → compare_library → profile_best → analyze
      analyze → decide
      decide → hp_search (含超参) / apply_direct (无超参) / terminate (max_iterations)
      hp_search → evaluate
      apply_direct → evaluate
      evaluate → reflect
      reflect → terminate (满足停止条件) / profile_best (继续)
      terminate → [*]
"""

from __future__ import annotations

import asyncio
import logging
import warnings
from typing import Any

from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

# LangGraph currently instantiates langchain-core's Reviver without
# allowed_objects during import. Keep CLI output clean while upstream catches up.
warnings.filterwarnings(
    "ignore",
    message=r"The default value of `allowed_objects` will change in a future version\.",
    category=LangChainPendingDeprecationWarning,
)

from langgraph.graph import END, StateGraph

from ..config import get_api_key, load_config
from ..memory.knowledge import KnowledgeBase
from ..memory.run_state import RunStateManager
from ..models.data import AgentConfig, HardwareSpec, OperatorSpec, RunState, RunStatus
from .llm_client import LLMClient, LLMStreamSink
from .nodes import AgentNodes
from .state import GraphState

logger = logging.getLogger(__name__)


def build_graph(
    config: AgentConfig | None = None,
    state_manager: RunStateManager | None = None,
    kb: KnowledgeBase | None = None,
    llm: LLMClient | None = None,
    entry_point: str = "init",
    stream_sink: LLMStreamSink | None = None,
) -> StateGraph:
    """
    构建 LangGraph 状态机。

    [改进] 新增 compare_library 节点。
    """
    if config is None:
        config = load_config()

    if state_manager is None:
        state_manager = RunStateManager(config)

    if kb is None:
        kb = KnowledgeBase(config.knowledge_base_dir)

    if llm is None:
        llm = LLMClient(provider=config.llm_provider, model=config.llm_model, stream_sink=stream_sink)
    elif stream_sink is not None:
        llm.stream_sink = stream_sink

    nodes = AgentNodes(state_manager=state_manager, kb=kb, llm=llm, stream_sink=stream_sink)

    # ── 构建图 ──
    graph = StateGraph(GraphState)

    # 添加节点
    graph.add_node("init", nodes.init_node)
    graph.add_node("bootstrap", nodes.bootstrap_node)
    graph.add_node("compile_validate", nodes.compile_and_validate_node)
    graph.add_node("compare_library", nodes.compare_library_node)  # [改进] 新增
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

    # compile_validate → compare_library (如果成功) 或 terminate (失败)
    # [改进] 编译成功后先走 compare_library 再 profile
    graph.add_conditional_edges(
        "compile_validate",
        _route_after_compile,
        {"compare_library": "compare_library", "terminate": "terminate"},
    )

    # compare_library → profile_best (总是继续, library comparison 是非阻塞的)
    graph.add_edge("compare_library", "profile_best")

    graph.add_edge("profile_best", "analyze")
    graph.add_edge("analyze", "decide")

    # decide → hp_search / apply_direct / terminate (max_iterations)
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
    """编译成功 → compare_library; 失败 → terminate。

    [改进] 改为先走 compare_library 再 profile。
    """
    if state.get("trial_compile_ok") and state.get("trial_correctness_ok"):
        return "compare_library"
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
    stream_sink: LLMStreamSink | None = None,
) -> RunState:
    """Synchronous compatibility wrapper around the async optimization path."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(run_optimization_async(
            operator_spec,
            config=config,
            resume_dir=resume_dir,
            stream_sink=stream_sink,
        ))
    raise RuntimeError("run_optimization() cannot be used from a running event loop; use run_optimization_async().")


async def run_optimization_async(
    operator_spec: OperatorSpec,
    config: AgentConfig | None = None,
    resume_dir: str | None = None,
    stream_sink: LLMStreamSink | None = None,
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
    llm = LLMClient(provider=config.llm_provider, model=config.llm_model, stream_sink=stream_sink)

    # 新建或续跑
    if resume_dir:
        run_state = state_manager.resume_run(run_dir=resume_dir)
        if run_state is None:
            raise FileNotFoundError(f"Cannot resume from {resume_dir}")
        entry_point = _infer_resume_entry_point(run_state)
    else:
        run_state = state_manager.new_run(operator_spec)
        entry_point = "init"

    graph = build_graph(
        config=config,
        state_manager=state_manager,
        kb=kb,
        llm=llm,
        entry_point=entry_point,
        stream_sink=stream_sink,
    )

    compiled = graph.compile()

    active_operator_spec = run_state.operator_spec if resume_dir else operator_spec
    initial_state: dict[str, Any] = {
        "operator_spec": active_operator_spec,
        "hardware_spec": run_state.hardware_spec,
        "run_state": run_state,
        "should_stop": False,
        "iteration_count": len(run_state.iterations),
    }

    final_state = await compiled.ainvoke(initial_state)

    state_manager.mark_done()
    return state_manager.state


def _infer_resume_entry_point(run_state: RunState) -> str:
    """从 RunState 推断续跑入口点。"""
    if not run_state.iterations:
        return "init"
    last = run_state.iterations[-1]
    if last.accepted and last.version_id == run_state.current_best_id:
        return "profile_best"
    return "profile_best"
