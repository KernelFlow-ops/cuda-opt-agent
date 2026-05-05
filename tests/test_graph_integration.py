"""
集成测试 —— LangGraph 状态机路由逻辑 + 节点交互。
使用 Mock LLM,不需要真实 API。
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from cuda_opt_agent.agent.state import GraphState
from cuda_opt_agent.models.data import (
    AgentConfig,
    BenchmarkResult,
    HardwareSpec,
    IterationRecord,
    MethodDecision,
    NcuMetrics,
    OperatorSpec,
    RunState,
    RunStatus,
)


class TestGraphRouting:
    """测试条件边路由逻辑。"""

    def test_route_after_compile_success(self):
        from cuda_opt_agent.agent.graph import _route_after_compile
        state = {"trial_compile_ok": True, "trial_correctness_ok": True}
        assert _route_after_compile(state) == "profile_best"

    def test_route_after_compile_failure(self):
        from cuda_opt_agent.agent.graph import _route_after_compile
        state = {"trial_compile_ok": False, "trial_correctness_ok": False}
        assert _route_after_compile(state) == "terminate"

    def test_route_after_compile_correctness_fail(self):
        from cuda_opt_agent.agent.graph import _route_after_compile
        state = {"trial_compile_ok": True, "trial_correctness_ok": False}
        assert _route_after_compile(state) == "terminate"

    def test_route_after_decide_hp(self):
        from cuda_opt_agent.agent.graph import _route_after_decide
        state = {"should_stop": False, "has_hyperparams": True}
        assert _route_after_decide(state) == "hp_search"

    def test_route_after_decide_no_hp(self):
        from cuda_opt_agent.agent.graph import _route_after_decide
        state = {"should_stop": False, "has_hyperparams": False}
        assert _route_after_decide(state) == "apply_direct"

    def test_route_after_decide_give_up(self):
        from cuda_opt_agent.agent.graph import _route_after_decide
        state = {"should_stop": True}
        assert _route_after_decide(state) == "terminate"

    def test_route_after_reflect_continue(self):
        from cuda_opt_agent.agent.graph import _route_after_reflect
        state = {"should_stop": False}
        assert _route_after_reflect(state) == "profile_best"

    def test_route_after_reflect_stop(self):
        from cuda_opt_agent.agent.graph import _route_after_reflect
        state = {"should_stop": True}
        assert _route_after_reflect(state) == "terminate"


class TestGraphBuild:
    """测试状态机构建。"""

    def test_build_graph(self, sample_agent_config):
        from cuda_opt_agent.agent.graph import build_graph
        from cuda_opt_agent.agent.llm_client import LLMClient
        from cuda_opt_agent.memory.knowledge import KnowledgeBase
        from cuda_opt_agent.memory.run_state import RunStateManager

        sm = RunStateManager(sample_agent_config)
        kb = KnowledgeBase(sample_agent_config.knowledge_base_dir)
        llm = LLMClient()

        graph = build_graph(config=sample_agent_config, state_manager=sm, kb=kb, llm=llm)
        assert graph is not None

    def test_graph_has_all_nodes(self, sample_agent_config):
        from cuda_opt_agent.agent.graph import build_graph
        from cuda_opt_agent.agent.llm_client import LLMClient
        from cuda_opt_agent.memory.knowledge import KnowledgeBase
        from cuda_opt_agent.memory.run_state import RunStateManager

        sm = RunStateManager(sample_agent_config)
        kb = KnowledgeBase(sample_agent_config.knowledge_base_dir)
        llm = LLMClient()

        graph = build_graph(config=sample_agent_config, state_manager=sm, kb=kb, llm=llm)

        expected_nodes = [
            "init", "bootstrap", "compile_validate", "profile_best",
            "analyze", "decide", "hp_search", "apply_direct",
            "evaluate", "reflect", "terminate",
        ]
        for node in expected_nodes:
            assert node in graph.nodes, f"缺少节点: {node}"


class TestAgentNodesMocked:
    """使用 Mock LLM 测试各节点逻辑。"""

    def _make_nodes(self, config):
        from cuda_opt_agent.agent.llm_client import LLMClient
        from cuda_opt_agent.agent.nodes import AgentNodes
        from cuda_opt_agent.memory.knowledge import KnowledgeBase
        from cuda_opt_agent.memory.run_state import RunStateManager

        sm = RunStateManager(config)
        kb = KnowledgeBase(config.knowledge_base_dir)
        llm = MagicMock(spec=LLMClient)
        return AgentNodes(state_manager=sm, kb=kb, llm=llm), sm, llm

    def test_init_node(self, sample_agent_config, sample_operator_spec):
        nodes, sm, llm = self._make_nodes(sample_agent_config)

        with patch("cuda_opt_agent.agent.nodes.collect_hardware_info") as mock_hw:
            mock_hw.return_value = HardwareSpec(
                gpu_name="Mock GPU",
                compute_capability="sm_80",
            )
            state: GraphState = {"operator_spec": sample_operator_spec}
            result = nodes.init_node(state)
            assert "hardware_spec" in result
            assert result["hardware_spec"].gpu_name == "Mock GPU"

    def test_bootstrap_node(self, sample_agent_config, sample_operator_spec, sample_hardware_spec):
        nodes, sm, llm = self._make_nodes(sample_agent_config)

        # Mock LLM 返回
        llm.format_prompt.return_value = "test prompt"
        llm.invoke.return_value = '''```cuda
#include <cuda_runtime.h>
__global__ void kernel() {}
int main() { return 0; }
```'''

        state: GraphState = {
            "operator_spec": sample_operator_spec,
            "hardware_spec": sample_hardware_spec,
        }
        result = nodes.bootstrap_node(state)
        assert "current_code" in result
        assert "__global__" in result["current_code"]
        assert result["new_version_id"] == "v0"

    def test_bootstrap_node_includes_seed_code(self, sample_agent_config,
                                               sample_operator_spec,
                                               sample_hardware_spec,
                                               tmp_dir):
        nodes, sm, llm = self._make_nodes(sample_agent_config)
        seed = tmp_dir / "seed.cu"
        seed.write_text("__global__ void seeded_kernel() {}\n", encoding="utf-8")
        op_spec = sample_operator_spec.model_copy(update={
            "seed_code_path": str(seed),
            "task_description": "Keep the algorithm unchanged.",
        })

        llm.format_prompt.return_value = "test prompt"
        llm.invoke.return_value = '''```cuda
#include <cuda_runtime.h>
__global__ void kernel() {}
int main() { return 0; }
```'''

        state: GraphState = {
            "operator_spec": op_spec,
            "hardware_spec": sample_hardware_spec,
        }
        result = nodes.bootstrap_node(state)

        _, kwargs = llm.format_prompt.call_args
        assert "seeded_kernel" in kwargs["seed_code_section"]
        assert "v0 baseline" in kwargs["bootstrap_mode_instruction"]
        assert kwargs["task_description"] == "Keep the algorithm unchanged."
        assert result["new_version_id"] == "v0"

    def test_analyze_node(self, sample_agent_config, sample_operator_spec,
                          sample_hardware_spec, sample_run_state,
                          sample_ncu_metrics, sample_benchmark_result):
        nodes, sm, llm = self._make_nodes(sample_agent_config)
        sm.state = sample_run_state
        sm.run_dir = None  # 不需要实际目录

        llm.format_prompt.return_value = "test prompt"
        llm.invoke_json.return_value = {
            "bottlenecks": [
                {"type": "memory_bound", "severity": 5, "evidence": "DRAM 92%", "description": "memory bound"}
            ],
            "observations": "high DRAM utilization",
        }

        state: GraphState = {
            "operator_spec": sample_operator_spec,
            "hardware_spec": sample_hardware_spec,
            "run_state": sample_run_state,
            "current_ncu": sample_ncu_metrics,
            "current_benchmark": sample_benchmark_result,
            "current_code": "__global__ void k() {}",
        }
        result = nodes.analyze_node(state)
        assert "analysis_result" in result
        assert "bottlenecks" in result["analysis_result"]

    def test_decide_node_normal(self, sample_agent_config, sample_operator_spec,
                                 sample_hardware_spec, sample_run_state,
                                 sample_benchmark_result):
        nodes, sm, llm = self._make_nodes(sample_agent_config)
        sm.state = sample_run_state

        llm.format_prompt.return_value = "test prompt"
        llm.invoke_json.return_value = {
            "method_name": "shared_memory_tiling",
            "has_hyperparams": True,
            "hyperparams_schema": {"tile_m": {"type": "int"}},
            "rationale": "DRAM bound",
            "expected_impact": "high",
            "confidence": 0.8,
            "give_up": False,
        }

        state: GraphState = {
            "operator_spec": sample_operator_spec,
            "hardware_spec": sample_hardware_spec,
            "run_state": sample_run_state,
            "current_benchmark": sample_benchmark_result,
            "analysis_result": {"bottlenecks": []},
        }
        result = nodes.decide_node(state)
        assert "method_decision" in result
        assert result["method_decision"].method_name == "shared_memory_tiling"
        assert result["has_hyperparams"] is True

    def test_decide_node_give_up(self, sample_agent_config, sample_operator_spec,
                                 sample_hardware_spec, sample_run_state,
                                 sample_benchmark_result):
        nodes, sm, llm = self._make_nodes(sample_agent_config)
        sm.state = sample_run_state

        llm.format_prompt.return_value = "test prompt"
        llm.invoke_json.return_value = {
            "method_name": "none",
            "has_hyperparams": False,
            "rationale": "所有方向已尝试",
            "expected_impact": "none",
            "confidence": 0.0,
            "give_up": True,
        }

        state: GraphState = {
            "operator_spec": sample_operator_spec,
            "hardware_spec": sample_hardware_spec,
            "run_state": sample_run_state,
            "current_benchmark": sample_benchmark_result,
            "analysis_result": {},
        }
        result = nodes.decide_node(state)
        assert result["should_stop"] is True

    def test_hp_search_compiles_all_candidates_before_gpu_work(self, sample_agent_config,
                                                               sample_operator_spec,
                                                               sample_hardware_spec,
                                                               sample_run_state):
        config = sample_agent_config.model_copy(update={"hp_compile_workers": 1})
        nodes, sm, llm = self._make_nodes(config)
        sm.state = sample_run_state
        sm.run_dir = sm.persistence.create_run_dir("test")

        llm.format_prompt.return_value = "prompt"
        llm.invoke_json.return_value = [
            {"index": 0, "hyperparams": {"tile": 64}, "rationale": "safe"},
            {"index": 1, "hyperparams": {"tile": 128}, "rationale": "fast"},
        ]
        llm.invoke.side_effect = [
            "```cuda\n__global__ void k0() {}\n```",
            "```cuda\n__global__ void k1() {}\n```",
        ]
        events = []

        from cuda_opt_agent.tools.compile import CompileResult
        from cuda_opt_agent.tools.correctness import CorrectnessResult

        def fake_compile(code_path, output_path, compute_capability):
            events.append(("compile", str(code_path)))
            return CompileResult(success=True, output_path=str(output_path), stdout="ok", stderr="", return_code=0)

        def fake_correctness(exe_path, shape_profiles, dtype):
            events.append(("check", str(exe_path)))
            return [{"correct": True, "message": "ok"}]

        def fake_benchmark(exe_path, op):
            events.append(("bench", str(exe_path)))
            latency = 2.0 if len([e for e in events if e[0] == "bench"]) == 1 else 1.0
            return BenchmarkResult(latency_ms_median=latency, latency_ms_p95=latency)

        state: GraphState = {
            "operator_spec": sample_operator_spec,
            "hardware_spec": sample_hardware_spec,
            "run_state": sample_run_state,
            "current_ncu": NcuMetrics(),
            "current_code": "__global__ void base() {}",
            "method_decision": MethodDecision(
                method_name="tiling",
                has_hyperparams=True,
                hyperparams_schema={"tile": {"type": "int"}},
            ),
        }

        with patch("cuda_opt_agent.agent.nodes.compile_cuda", side_effect=fake_compile), \
             patch("cuda_opt_agent.agent.nodes.check_correctness_multi", side_effect=fake_correctness), \
             patch.object(nodes, "_benchmark_multi", side_effect=fake_benchmark):
            result = nodes.hp_search_node(state)

        event_types = [event[0] for event in events]
        assert event_types == ["compile", "compile", "check", "bench", "check", "bench"]
        assert result["trial_benchmark"].latency_ms_median == 1.0
        assert result["new_version_id"].endswith("cand1")

    def test_hp_compile_worker_count_auto_uses_cpu_limit(self, sample_agent_config):
        config = sample_agent_config.model_copy(update={"hp_compile_workers": 0})
        nodes, _, _ = self._make_nodes(config)

        with patch("cuda_opt_agent.agent.nodes.os.cpu_count", return_value=8):
            assert nodes._hp_compile_worker_count(5) == 5

        config = sample_agent_config.model_copy(update={"hp_compile_workers": 2})
        nodes, _, _ = self._make_nodes(config)
        assert nodes._hp_compile_worker_count(5) == 2

    def test_evaluate_ignores_stale_trial_benchmark(self, sample_agent_config,
                                                     sample_operator_spec,
                                                     sample_hardware_spec,
                                                     sample_run_state):
        nodes, sm, llm = self._make_nodes(sample_agent_config)
        sm.state = sample_run_state
        sm.run_dir = sm.persistence.create_run_dir("test")

        stale_benchmark = BenchmarkResult(latency_ms_median=0.1, latency_ms_p95=0.1)
        fresh_benchmark = BenchmarkResult(latency_ms_median=0.9, latency_ms_p95=0.9)

        state: GraphState = {
            "operator_spec": sample_operator_spec,
            "hardware_spec": sample_hardware_spec,
            "run_state": sample_run_state,
            "current_benchmark": BenchmarkResult(latency_ms_median=1.0, latency_ms_p95=1.0),
            "new_code": "__global__ void k() {}",
            "new_version_id": "v1",
            "trial_version_id": "v0_hp_cand0",
            "trial_benchmark": stale_benchmark,
        }

        with patch.object(nodes, "compile_and_validate_node") as mock_compile, \
             patch("cuda_opt_agent.agent.nodes.run_benchmark_multi") as mock_benchmark:
            mock_compile.return_value = {
                "trial_version_id": "v1",
                "trial_compile_ok": True,
                "trial_correctness_ok": True,
            }
            mock_benchmark.return_value = fresh_benchmark

            result = nodes.evaluate_node(state)

        mock_compile.assert_called_once()
        assert result["trial_version_id"] == "v1"
        assert result["trial_benchmark"] == fresh_benchmark
        assert result["trial_accepted"] is True

    def test_evaluate_reuses_matching_trial_benchmark(self, sample_agent_config,
                                                      sample_operator_spec,
                                                      sample_hardware_spec,
                                                      sample_run_state):
        nodes, sm, llm = self._make_nodes(sample_agent_config)
        sm.state = sample_run_state
        sm.run_dir = sm.persistence.create_run_dir("test")

        trial_benchmark = BenchmarkResult(latency_ms_median=0.9, latency_ms_p95=0.9)
        state: GraphState = {
            "operator_spec": sample_operator_spec,
            "hardware_spec": sample_hardware_spec,
            "run_state": sample_run_state,
            "current_benchmark": BenchmarkResult(latency_ms_median=1.0, latency_ms_p95=1.0),
            "new_version_id": "v2_hp_cand0",
            "trial_version_id": "v2_hp_cand0",
            "trial_benchmark": trial_benchmark,
        }

        with patch.object(nodes, "compile_and_validate_node") as mock_compile:
            result = nodes.evaluate_node(state)

        mock_compile.assert_not_called()
        assert result["trial_benchmark"] == trial_benchmark
        assert result["trial_accepted"] is True

    def test_terminate_node(self, sample_agent_config, sample_operator_spec,
                            sample_hardware_spec, sample_run_state):
        nodes, sm, llm = self._make_nodes(sample_agent_config)
        sm.state = sample_run_state
        sm.run_dir = sm.persistence.create_run_dir("test")

        state: GraphState = {
            "run_state": sample_run_state,
            "stop_reason": "达到最大迭代数",
        }
        result = nodes.terminate_node(state)
        assert result["should_stop"] is True
        assert sm.state.status == RunStatus.DONE


class TestEndToEndMocked:
    """使用 Mock 的端到端流程测试。"""

    def test_full_state_lifecycle(self, sample_agent_config, sample_operator_spec, sample_hardware_spec):
        """验证完整的状态生命周期: 创建 → 迭代 → 终止。"""
        from cuda_opt_agent.memory.run_state import RunStateManager

        sm = RunStateManager(sample_agent_config)
        state = sm.init_new_run(sample_operator_spec, sample_hardware_spec)

        assert state.status == RunStatus.RUNNING
        assert len(state.iterations) == 0

        # 模拟 v0
        v0 = IterationRecord(
            version_id="v0",
            compile_ok=True,
            correctness_ok=True,
            benchmark=BenchmarkResult(latency_ms_median=2.0, latency_ms_p95=2.5),
            accepted=True,
        )
        sm.add_iteration(v0)
        sm.update_best("v0")
        assert state.current_best_id == "v0"

        # 模拟 v1 (accepted)
        v1 = IterationRecord(
            version_id="v1",
            parent_id="v0",
            method_name="tiling",
            compile_ok=True,
            correctness_ok=True,
            benchmark=BenchmarkResult(latency_ms_median=1.0, latency_ms_p95=1.2),
            accepted=True,
        )
        sm.add_iteration(v1)
        sm.update_best("v1")
        assert state.current_best_id == "v1"

        # 模拟 v2 (rejected)
        v2 = IterationRecord(
            version_id="v2",
            parent_id="v1",
            method_name="vectorization",
            compile_ok=True,
            correctness_ok=True,
            benchmark=BenchmarkResult(latency_ms_median=1.05, latency_ms_p95=1.3),
            accepted=False,
        )
        sm.add_iteration(v2)
        sm.add_to_blacklist("vectorization", "no speedup")

        assert state.current_best_id == "v1"  # 仍然是 v1
        assert state.is_method_blacklisted("vectorization")

        # 终止
        sm.mark_done()
        assert state.status == RunStatus.DONE
