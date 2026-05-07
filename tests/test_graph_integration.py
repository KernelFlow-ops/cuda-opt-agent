"""
集成测试 —— LangGraph 状态机路由逻辑 + 节点交互。
使用 Mock LLM,不需要真实 API。
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cuda_opt_agent.agent.state import GraphState
from cuda_opt_agent.agent.temperatures import TEMP_APPLY_METHOD, TEMP_PROPOSE_HP
from cuda_opt_agent.models.data import (
    AgentConfig,
    BlacklistEntry,
    BenchmarkResult,
    HardwareSpec,
    IterationRecord,
    MethodDecision,
    NcuMetrics,
    OperatorSpec,
    RunState,
    RunStatus,
)


def run_async(coro):
    return asyncio.run(coro)


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
        llm.ainvoke = AsyncMock()
        llm.ainvoke_json = AsyncMock()
        llm.ainvoke_structured = AsyncMock()
        return AgentNodes(state_manager=sm, kb=kb, llm=llm), sm, llm

    def test_init_node(self, sample_agent_config, sample_operator_spec):
        nodes, sm, llm = self._make_nodes(sample_agent_config)

        with patch("cuda_opt_agent.agent.nodes.init.collect_hardware_info") as mock_hw:
            mock_hw.return_value = HardwareSpec(
                gpu_name="Mock GPU",
                compute_capability="sm_80",
            )
            state: GraphState = {"operator_spec": sample_operator_spec}
            result = run_async(nodes.init_node(state))
            assert "hardware_spec" in result
            assert result["hardware_spec"].gpu_name == "Mock GPU"

    def test_bootstrap_node(self, sample_agent_config, sample_operator_spec, sample_hardware_spec):
        nodes, sm, llm = self._make_nodes(sample_agent_config)

        # Mock LLM 返回
        llm.format_prompt.return_value = "test prompt"
        llm.ainvoke.return_value = '''```cuda
#include <cuda_runtime.h>
__global__ void kernel() {}
int main() { return 0; }
```'''

        state: GraphState = {
            "operator_spec": sample_operator_spec,
            "hardware_spec": sample_hardware_spec,
        }
        result = run_async(nodes.bootstrap_node(state))
        assert "current_code" in result
        assert "__global__" in result["current_code"]
        assert result["new_version_id"] == "v0"
        assert llm.ainvoke.call_args.kwargs["temperature"] == 0.2

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
        llm.ainvoke.return_value = '''```cuda
#include <cuda_runtime.h>
__global__ void kernel() {}
int main() { return 0; }
```'''

        state: GraphState = {
            "operator_spec": op_spec,
            "hardware_spec": sample_hardware_spec,
        }
        result = run_async(nodes.bootstrap_node(state))

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
        llm.ainvoke_json.return_value = {
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
        result = run_async(nodes.analyze_node(state))
        assert "analysis_result" in result
        assert "bottlenecks" in result["analysis_result"]
        assert llm.ainvoke_json.call_args.kwargs["temperature"] == 0.1

    def test_profile_args_use_ncu_config(self, sample_agent_config):
        config = sample_agent_config.model_copy(update={
            "ncu_warmup_rounds": 3,
            "ncu_profile_rounds": 2,
        })
        nodes, _, _ = self._make_nodes(config)
        bm = BenchmarkResult(extra={"worst_shape": {"M": 128, "N": 256}})

        assert nodes._profile_args_from_benchmark(bm) == [
            "--shape", "M=128", "N=256",
            "--warmup", "3",
            "--rounds", "2",
        ]

    def test_profile_best_passes_ncu_launch_count(self, sample_agent_config,
                                                        sample_operator_spec,
                                                        sample_hardware_spec,
                                                        sample_iteration_record):
        config = sample_agent_config.model_copy(update={"ncu_launch_count": 7})
        nodes, sm, _ = self._make_nodes(config)
        sm.state = RunState(
            run_id="test_run",
            operator_spec=sample_operator_spec,
            hardware_spec=sample_hardware_spec,
            iterations=[sample_iteration_record],
            current_best_id="v0",
            config=config,
        )
        sm.run_dir = sm.persistence.create_run_dir("test")
        iter_dir = sm.run_dir / "iterv0"
        iter_dir.mkdir(parents=True, exist_ok=True)
        exe_path = iter_dir / "kernel.exe"
        exe_path.write_text("", encoding="utf-8")
        code_path = iter_dir / "code.cu"
        code_path.write_text("__global__ void k() {}", encoding="utf-8")

        state: GraphState = {
            "operator_spec": sample_operator_spec,
            "hardware_spec": sample_hardware_spec,
            "run_state": sm.state,
        }
        bm = BenchmarkResult(latency_ms_median=1.0, latency_ms_p95=1.1, extra={"worst_shape": {}})
        ncu = NcuMetrics(sm_throughput_pct=10.0)

        with patch.object(nodes, "_benchmark_multi", return_value=bm), \
             patch("cuda_opt_agent.agent.nodes.profile.run_ncu_profile", return_value=ncu) as mock_profile:
            result = run_async(nodes.profile_best_node(state))

        assert result["current_ncu"] == ncu
        assert mock_profile.call_args.kwargs["launch_count"] == 7

    def test_decide_node_normal(self, sample_agent_config, sample_operator_spec,
                                      sample_hardware_spec, sample_run_state,
                                      sample_benchmark_result):
        nodes, sm, llm = self._make_nodes(sample_agent_config)
        sm.state = sample_run_state

        llm.format_prompt.return_value = "test prompt"
        llm.ainvoke_json.return_value = {
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
        result = run_async(nodes.decide_node(state))
        assert "method_decision" in result
        assert result["method_decision"].method_name == "shared_memory_tiling"
        assert result["has_hyperparams"] is True
        assert llm.ainvoke_json.call_args.kwargs["temperature"] == 0.1

    def test_decide_node_includes_method_hp_history(self, sample_agent_config,
                                                          sample_operator_spec,
                                                          sample_hardware_spec,
                                                          sample_run_state,
                                                          sample_benchmark_result):
        nodes, sm, llm = self._make_nodes(sample_agent_config)
        sm.state = sample_run_state
        sample_run_state.iterations.append(IterationRecord(
            version_id="v1_hp_cand0",
            parent_id="v0",
            method_name="tiling",
            has_hyperparams=True,
            hyperparams={"tile": 128, "k": 32},
            compile_ok=True,
            correctness_ok=True,
            benchmark=BenchmarkResult(latency_ms_median=1.5, latency_ms_p95=1.6),
            accepted=False,
        ))

        llm.format_prompt.return_value = "test prompt"
        llm.ainvoke_json.return_value = {
            "method_name": "shared_memory_tiling",
            "has_hyperparams": True,
            "hyperparams_schema": {"tile": {"type": "int"}},
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
            "analysis_result": {},
        }
        run_async(nodes.decide_node(state))

        method_history = llm.format_prompt.call_args.kwargs["method_history"]
        assert "v1_hp_cand0" in method_history
        assert '"tile": 128' in method_history
        assert "1.5000 ms" in method_history

    def test_decide_node_give_up(self, sample_agent_config, sample_operator_spec,
                                       sample_hardware_spec, sample_run_state,
                                       sample_benchmark_result):
        nodes, sm, llm = self._make_nodes(sample_agent_config)
        sm.state = sample_run_state

        llm.format_prompt.return_value = "test prompt"
        llm.ainvoke_json.return_value = {
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
        result = run_async(nodes.decide_node(state))
        assert result["should_stop"] is True
        assert llm.ainvoke_json.call_count == 1

    def test_decide_node_reselects_blacklisted_method(self, sample_agent_config,
                                                            sample_operator_spec,
                                                            sample_hardware_spec,
                                                            sample_run_state,
                                                            sample_benchmark_result):
        nodes, sm, llm = self._make_nodes(sample_agent_config)
        sm.state = sample_run_state
        sample_run_state.blacklist.append(BlacklistEntry(
            method_name_normalized="tiling",
            reason="no speedup",
        ))

        llm.format_prompt.return_value = "test prompt"
        llm.ainvoke_json.side_effect = [
            {
                "method_name": "tiling",
                "has_hyperparams": False,
                "rationale": "try again",
                "expected_impact": "medium",
                "confidence": 0.5,
                "give_up": False,
            },
            {
                "method_name": "warp_shuffle",
                "has_hyperparams": False,
                "rationale": "reduce shared memory traffic",
                "expected_impact": "medium",
                "confidence": 0.6,
                "give_up": False,
            },
        ]

        state: GraphState = {
            "operator_spec": sample_operator_spec,
            "hardware_spec": sample_hardware_spec,
            "run_state": sample_run_state,
            "current_benchmark": sample_benchmark_result,
            "analysis_result": {},
        }
        result = run_async(nodes.decide_node(state))

        assert "should_stop" not in result
        assert result["method_decision"].method_name == "warp_shuffle"
        assert llm.ainvoke_json.call_count == 2
        assert "tiling" in llm.format_prompt.call_args_list[1].kwargs["rejected_methods"]

    def test_decide_node_gives_up_after_reselect_retries_exhausted(self,
                                                                        sample_agent_config,
                                                                        sample_operator_spec,
                                                                        sample_hardware_spec,
                                                                        sample_run_state,
                                                                        sample_benchmark_result):
        config = sample_agent_config.model_copy(update={"decide_reselect_max_retries": 2})
        nodes, sm, llm = self._make_nodes(config)
        sm.state = sample_run_state
        sample_run_state.blacklist.append(BlacklistEntry(
            method_name_normalized="tiling",
            reason="no speedup",
        ))

        llm.format_prompt.return_value = "test prompt"
        llm.ainvoke_json.return_value = {
            "method_name": "tiling",
            "has_hyperparams": False,
            "rationale": "still seems best",
            "expected_impact": "medium",
            "confidence": 0.5,
            "give_up": False,
        }

        state: GraphState = {
            "operator_spec": sample_operator_spec,
            "hardware_spec": sample_hardware_spec,
            "run_state": sample_run_state,
            "current_benchmark": sample_benchmark_result,
            "analysis_result": {},
        }
        result = run_async(nodes.decide_node(state))

        assert result["should_stop"] is True
        assert result["method_decision"].give_up is True
        assert "exhausted decide reselection retries" in result["stop_reason"]
        assert llm.ainvoke_json.call_count == 3

    def test_hp_search_compiles_all_candidates_before_gpu_work(self, sample_agent_config,
                                                                     sample_operator_spec,
                                                                     sample_hardware_spec,
                                                                     sample_run_state):
        config = sample_agent_config.model_copy(update={"hp_compile_workers": 1})
        nodes, sm, llm = self._make_nodes(config)
        sm.state = sample_run_state
        sm.run_dir = sm.persistence.create_run_dir("test")

        llm.format_prompt.return_value = "prompt"
        llm.ainvoke_json.return_value = [
            {"index": 0, "hyperparams": {"tile": 64}, "rationale": "safe"},
            {"index": 1, "hyperparams": {"tile": 128}, "rationale": "fast"},
        ]
        llm.ainvoke.side_effect = [
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

        with patch("cuda_opt_agent.agent.nodes._helpers.compile_cuda", side_effect=fake_compile), \
             patch("cuda_opt_agent.agent.nodes.hp_search.check_correctness_multi", side_effect=fake_correctness), \
             patch.object(nodes, "_benchmark_multi", side_effect=fake_benchmark):
            result = run_async(nodes.hp_search_node(state))

        event_types = [event[0] for event in events]
        assert event_types == ["compile", "compile", "check", "bench", "check", "bench"]
        assert result["trial_benchmark"].latency_ms_median == 1.0
        assert result["new_version_id"].endswith("cand1")
        assert llm.ainvoke_json.call_args.kwargs["temperature"] == TEMP_PROPOSE_HP
        assert [call.kwargs["temperature"] for call in llm.ainvoke.call_args_list] == [
            TEMP_APPLY_METHOD,
            TEMP_APPLY_METHOD,
        ]
        assert [call.kwargs["node_name"] for call in llm.ainvoke.call_args_list] == [
            "hp_search:cand0",
            "hp_search:cand1",
        ]

    def test_hp_search_repairs_compile_failed_candidate(self, sample_agent_config,
                                                              sample_operator_spec,
                                                              sample_hardware_spec,
                                                              sample_run_state):
        config = sample_agent_config.model_copy(update={
            "hp_compile_workers": 1,
            "compile_repair_max_retries": 1,
        })
        nodes, sm, llm = self._make_nodes(config)
        sm.state = sample_run_state
        sm.run_dir = sm.persistence.create_run_dir("test")

        llm.format_prompt.return_value = "prompt"
        llm.ainvoke_json.return_value = [
            {"index": 0, "hyperparams": {"tile": 64}, "rationale": "safe"},
        ]
        llm.ainvoke.side_effect = [
            "```cuda\n__global__ void broken() {}\n```",
            "```cuda\n__global__ void repaired() {}\n```",
        ]
        events = []

        from cuda_opt_agent.tools.compile import CompileResult

        def fake_initial_compile(code_path, output_path, compute_capability):
            events.append(("initial_compile", Path(code_path).name))
            return CompileResult(
                success=False,
                output_path=str(output_path),
                stdout="",
                stderr="bad generated code",
                return_code=1,
            )

        def fake_repair_compile(code_path, output_path, compute_capability, **kwargs):
            events.append(("repair_compile", Path(code_path).name))
            output_path = Path(output_path)
            output_path.write_text("kernel", encoding="utf-8")
            return CompileResult(
                success=True,
                output_path=str(output_path),
                stdout="ok",
                stderr="",
                return_code=0,
            )

        def fake_correctness(exe_path, shape_profiles, dtype, gpu_id=None):
            events.append(("check", Path(exe_path).name))
            return [{"correct": True, "message": "ok"}]

        def fake_benchmark(exe_path, op, gpu_id=None):
            events.append(("bench", Path(exe_path).name))
            return BenchmarkResult(latency_ms_median=1.0, latency_ms_p95=1.1)

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

        with patch("cuda_opt_agent.agent.nodes._helpers.compile_cuda", side_effect=fake_initial_compile), \
             patch("cuda_opt_agent.agent.nodes.hp_search.compile_cuda", side_effect=fake_repair_compile), \
             patch("cuda_opt_agent.agent.nodes.hp_search.check_correctness_multi", side_effect=fake_correctness), \
             patch.object(nodes, "_benchmark_multi", side_effect=fake_benchmark):
            result = run_async(nodes.hp_search_node(state))

        assert [event[0] for event in events] == [
            "initial_compile",
            "repair_compile",
            "check",
            "bench",
        ]
        assert result["trial_compile_ok"] is True
        assert result["trial_correctness_ok"] is True
        assert "repaired" in result["new_code"]

        iter_dir = sm.run_dir / f"iter{result['new_version_id']}"
        assert "repaired" in (iter_dir / "code.cu").read_text(encoding="utf-8")
        assert (iter_dir / "kernel").exists()
        assert "bad generated code" in (iter_dir / "compile.log").read_text(encoding="utf-8")

    def test_hp_search_handles_absolute_output_path_for_canonical_kernel(
        self,
        sample_agent_config,
        sample_operator_spec,
        sample_hardware_spec,
        sample_run_state,
        tmp_path,
        monkeypatch,
    ):
        monkeypatch.chdir(tmp_path)
        config = sample_agent_config.model_copy(update={
            "runs_dir": "runs",
            "knowledge_base_dir": "kb",
            "hp_compile_workers": 1,
        })
        nodes, sm, llm = self._make_nodes(config)
        sm.state = sample_run_state
        sm.run_dir = sm.persistence.create_run_dir("test")

        llm.format_prompt.return_value = "prompt"
        llm.ainvoke_json.return_value = [
            {"index": 0, "hyperparams": {"tile": 64}, "rationale": "safe"},
        ]
        llm.ainvoke.return_value = "```cuda\n__global__ void k() {}\n```"
        events = []

        from cuda_opt_agent.tools.compile import CompileResult

        def fake_compile(code_path, output_path, compute_capability):
            events.append(("compile", str(output_path)))
            abs_output_path = Path(output_path).resolve()
            abs_output_path.parent.mkdir(parents=True, exist_ok=True)
            abs_output_path.write_text("kernel", encoding="utf-8")
            return CompileResult(
                success=True,
                output_path=str(abs_output_path),
                stdout="ok",
                stderr="",
                return_code=0,
            )

        def fake_correctness(exe_path, shape_profiles, dtype, gpu_id=None):
            events.append(("check", str(exe_path)))
            return [{"correct": True, "message": "ok"}]

        def fake_benchmark(exe_path, op, gpu_id=None):
            events.append(("bench", str(exe_path)))
            return BenchmarkResult(latency_ms_median=1.0, latency_ms_p95=1.1)

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

        with patch("cuda_opt_agent.agent.nodes._helpers.compile_cuda", side_effect=fake_compile), \
             patch("cuda_opt_agent.agent.nodes.hp_search.check_correctness_multi", side_effect=fake_correctness), \
             patch.object(nodes, "_benchmark_multi", side_effect=fake_benchmark):
            result = run_async(nodes.hp_search_node(state))

        assert [event[0] for event in events] == ["compile", "check", "bench"]
        assert result["trial_compile_ok"] is True
        assert result["trial_correctness_ok"] is True
        assert result["trial_benchmark"].latency_ms_median == 1.0

    def test_hp_search_includes_known_hp_trials(self, sample_agent_config,
                                                      sample_operator_spec,
                                                      sample_hardware_spec,
                                                      sample_run_state):
        nodes, sm, llm = self._make_nodes(sample_agent_config)
        sm.state = sample_run_state
        sample_run_state.iterations.append(IterationRecord(
            version_id="v1_hp_cand0",
            parent_id="v0",
            method_name="tiling",
            has_hyperparams=True,
            hyperparams={"tile": 128},
            compile_ok=True,
            correctness_ok=False,
            accepted=False,
        ))

        llm.format_prompt.return_value = "prompt"
        llm.ainvoke_json.return_value = []

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
        run_async(nodes.hp_search_node(state))

        known_hp_trials = llm.format_prompt.call_args.kwargs["known_hp_trials"]
        assert "v1_hp_cand0" in known_hp_trials
        assert '"tile": 128' in known_hp_trials
        assert "failed correctness" in known_hp_trials

    def test_reflect_records_selected_hp_candidate(self, sample_agent_config,
                                                         sample_operator_spec,
                                                         sample_hardware_spec,
                                                         sample_run_state):
        nodes, sm, llm = self._make_nodes(sample_agent_config)
        sm.state = sample_run_state
        sm.run_dir = sm.persistence.create_run_dir("test")
        llm.format_prompt.return_value = "prompt"
        llm.ainvoke_json.return_value = {"why_ineffective": "no speedup"}

        state: GraphState = {
            "operator_spec": sample_operator_spec,
            "hardware_spec": sample_hardware_spec,
            "run_state": sample_run_state,
            "method_decision": MethodDecision(method_name="tiling", has_hyperparams=True),
            "current_benchmark": BenchmarkResult(latency_ms_median=1.0, latency_ms_p95=1.0),
            "trial_benchmark": BenchmarkResult(latency_ms_median=1.2, latency_ms_p95=1.3),
            "new_version_id": "v1_hp_cand0",
            "trial_compile_ok": True,
            "trial_correctness_ok": True,
            "trial_accepted": False,
            "hp_candidates": [
                {
                    "version_id": "v1_hp_cand0",
                    "hyperparams": {"tile": 128, "k": 32},
                }
            ],
        }
        run_async(nodes.reflect_node(state))

        record = sample_run_state.iter_by_id("v1_hp_cand0")
        assert record is not None
        assert record.hyperparams == {"tile": 128, "k": 32}
        assert sample_run_state.blacklist[-1].hyperparam_constraint == {"tile": 128, "k": 32}
        assert '{"k": 32, "tile": 128}' in llm.format_prompt.call_args.kwargs["hyperparams"]
        assert llm.ainvoke_json.call_args.kwargs["temperature"] == 0.5

    def test_apply_direct_uses_apply_temperature(self, sample_agent_config,
                                                       sample_operator_spec,
                                                       sample_hardware_spec,
                                                       sample_run_state):
        nodes, sm, llm = self._make_nodes(sample_agent_config)
        llm.format_prompt.return_value = "prompt"
        llm.ainvoke.return_value = "```cuda\n__global__ void k() {}\n```"

        state: GraphState = {
            "operator_spec": sample_operator_spec,
            "hardware_spec": sample_hardware_spec,
            "run_state": sample_run_state,
            "current_ncu": NcuMetrics(),
            "current_code": "__global__ void base() {}",
            "method_decision": MethodDecision(method_name="warp_shuffle"),
        }

        result = run_async(nodes.apply_direct_node(state))

        assert result["new_code"]
        assert llm.ainvoke.call_args.kwargs["temperature"] == TEMP_APPLY_METHOD

    def test_hp_compile_worker_count_auto_uses_cpu_limit(self, sample_agent_config):
        config = sample_agent_config.model_copy(update={"hp_compile_workers": 0})
        nodes, _, _ = self._make_nodes(config)

        with patch("cuda_opt_agent.agent.nodes._helpers.os.cpu_count", return_value=8):
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

        with patch.object(nodes, "compile_and_validate_node", new_callable=AsyncMock) as mock_compile, \
             patch("cuda_opt_agent.agent.nodes._helpers.run_benchmark_multi") as mock_benchmark:
            mock_compile.return_value = {
                "trial_version_id": "v1",
                "trial_compile_ok": True,
                "trial_correctness_ok": True,
            }
            mock_benchmark.return_value = fresh_benchmark

            result = run_async(nodes.evaluate_node(state))

        mock_compile.assert_called_once()
        assert result["trial_version_id"] == "v1"
        assert result["trial_benchmark"] == fresh_benchmark
        assert result["trial_accepted"] is True
        assert result["trial_compile_ok"] is True
        assert result["trial_correctness_ok"] is True

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

        with patch.object(nodes, "compile_and_validate_node", new_callable=AsyncMock) as mock_compile:
            result = run_async(nodes.evaluate_node(state))

        mock_compile.assert_not_called()
        assert result["trial_benchmark"] == trial_benchmark
        assert result["trial_accepted"] is True
        assert result["trial_compile_ok"] is True
        assert result["trial_correctness_ok"] is True

    def test_terminate_node(self, sample_agent_config, sample_operator_spec,
                                  sample_hardware_spec, sample_run_state):
        nodes, sm, llm = self._make_nodes(sample_agent_config)
        sm.state = sample_run_state
        sm.run_dir = sm.persistence.create_run_dir("test")

        state: GraphState = {
            "run_state": sample_run_state,
            "stop_reason": "达到最大迭代数",
        }
        result = run_async(nodes.terminate_node(state))
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
