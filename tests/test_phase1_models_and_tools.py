"""
Phase 1 测试 —— 数据模型 + 工具层。
不依赖 GPU 和 LLM API,纯单元测试。
"""

import json
import pytest
from pathlib import Path


# ════════════════════════════════════════════
# 数据模型测试
# ════════════════════════════════════════════
class TestOperatorSpec:
    def test_create(self, sample_operator_spec):
        assert sample_operator_spec.name == "gemm"
        assert "fp16" in sample_operator_spec.dtypes.values()

    def test_serialization(self, sample_operator_spec):
        data = sample_operator_spec.model_dump()
        assert isinstance(data, dict)
        assert data["name"] == "gemm"

        # 反序列化
        from cuda_opt_agent.models.data import OperatorSpec
        restored = OperatorSpec.model_validate(data)
        assert restored.name == sample_operator_spec.name

    def test_json_roundtrip(self, sample_operator_spec):
        json_str = sample_operator_spec.model_dump_json()
        from cuda_opt_agent.models.data import OperatorSpec
        restored = OperatorSpec.model_validate_json(json_str)
        assert restored == sample_operator_spec


class TestHardwareSpec:
    def test_create(self, sample_hardware_spec):
        assert sample_hardware_spec.gpu_name == "NVIDIA A100-SXM4-80GB"
        assert sample_hardware_spec.compute_capability == "sm_80"

    def test_signature(self, sample_hardware_spec):
        sig = sample_hardware_spec.signature
        assert "sm_80" in sig
        assert "a100" in sig.lower()

    def test_defaults(self):
        from cuda_opt_agent.models.data import HardwareSpec
        hw = HardwareSpec()
        assert hw.gpu_name == ""
        assert hw.sm_count == 0


class TestBenchmarkResult:
    def test_create(self, sample_benchmark_result):
        assert sample_benchmark_result.latency_ms_median == 1.234
        assert sample_benchmark_result.throughput_gflops == 5678.9

    def test_extra_fields(self):
        from cuda_opt_agent.models.data import BenchmarkResult
        bm = BenchmarkResult(
            latency_ms_median=1.0,
            latency_ms_p95=1.2,
            extra={"min_ms": 0.9, "max_ms": 1.5},
        )
        assert bm.extra["min_ms"] == 0.9


class TestNcuMetrics:
    def test_create(self, sample_ncu_metrics):
        assert sample_ncu_metrics.dram_throughput_pct == 92.1
        assert "long_scoreboard" in sample_ncu_metrics.stall_reasons


class TestRunState:
    def test_create(self, sample_run_state):
        assert sample_run_state.run_id == "gemm_run_20260501T120000"
        assert sample_run_state.current_best_id == "v0"

    def test_iter_by_id(self, sample_run_state):
        v0 = sample_run_state.iter_by_id("v0")
        assert v0 is not None
        assert v0.version_id == "v0"

        missing = sample_run_state.iter_by_id("v999")
        assert missing is None

    def test_accepted_iterations(self, sample_run_state):
        accepted = sample_run_state.accepted_iterations()
        assert len(accepted) == 1

    def test_consecutive_rejects(self, sample_run_state):
        assert sample_run_state.consecutive_rejects() == 0

    def test_next_version_id(self, sample_run_state):
        vid = sample_run_state.next_version_id()
        assert vid == "v1"

        vid_hp = sample_run_state.next_version_id(has_hp=True)
        assert "_hp" in vid_hp

    def test_blacklist_check(self, sample_run_state):
        assert not sample_run_state.is_method_blacklisted("tiling")

        from cuda_opt_agent.models.data import BlacklistEntry
        sample_run_state.blacklist.append(BlacklistEntry(
            method_name_normalized="shared_memory_tiling",
            reason="test",
        ))
        assert sample_run_state.is_method_blacklisted("shared_memory_tiling")
        assert sample_run_state.is_method_blacklisted("Shared Memory Tiling")  # 归一化


class TestMethodDecision:
    def test_create(self):
        from cuda_opt_agent.models.data import MethodDecision
        d = MethodDecision(
            method_name="shared_memory_tiling",
            has_hyperparams=True,
            hyperparams_schema={"tile_m": {"type": "int", "range": [16, 256]}},
            rationale="DRAM throughput 92%, need to cache data in shared memory",
            expected_impact="high",
            confidence=0.8,
        )
        assert d.method_name == "shared_memory_tiling"
        assert d.has_hyperparams is True
        assert d.give_up is False


# ════════════════════════════════════════════
# 枚举与归一化测试
# ════════════════════════════════════════════
class TestNormalization:
    def test_normalize_method_name(self):
        from cuda_opt_agent.models.enums import normalize_method_name
        assert normalize_method_name("Shared Memory Tiling") == "shared_memory_tiling"
        assert normalize_method_name("  loop-unrolling  ") == "loop_unrolling"
        assert normalize_method_name("use__double__buffer") == "use_double_buffer"

    def test_make_blacklist_key(self):
        from cuda_opt_agent.models.enums import make_blacklist_key
        key1 = make_blacklist_key("tiling")
        assert key1 == "tiling"

        key2 = make_blacklist_key("tiling", {"tile_m": 128})
        assert "::" in key2
        assert "tile_m" in key2


# ════════════════════════════════════════════
# 工具层测试 (不需 GPU)
# ════════════════════════════════════════════
class TestCompileTool:
    def test_missing_source(self, tmp_dir):
        from cuda_opt_agent.tools.compile import compile_cuda
        result = compile_cuda(tmp_dir / "nonexistent.cu")
        assert result.success is False
        assert "not found" in result.stderr.lower()

    def test_compile_command_no_nvcc(self, tmp_dir, monkeypatch):
        """当 nvcc 不可用时应该优雅失败。"""
        import shutil
        monkeypatch.setattr(shutil, "which", lambda x: None)
        from cuda_opt_agent.tools.compile import compile_cuda

        src = tmp_dir / "test.cu"
        src.write_text("__global__ void k() {}", encoding="utf-8")
        result = compile_cuda(src)
        assert result.success is False
        assert "nvcc" in result.stderr.lower()


class TestBenchmarkTool:
    def test_parse_json_output(self):
        from cuda_opt_agent.tools.benchmark import _parse_benchmark_output
        output = json.dumps({
            "latencies_ms": [1.0, 1.1, 1.05, 0.95, 1.02],
        })
        result = _parse_benchmark_output(output)
        assert result.latency_ms_median > 0
        assert result.latency_ms_p95 > 0

    def test_parse_invalid_output(self):
        from cuda_opt_agent.tools.benchmark import _parse_benchmark_output
        result = _parse_benchmark_output("not json at all")
        assert result.latency_ms_median == 0

    def test_parse_preformatted_output(self):
        from cuda_opt_agent.tools.benchmark import _parse_benchmark_output
        output = 'Some logs\n{"latency_ms_median": 0.5, "latency_ms_p95": 0.6}'
        result = _parse_benchmark_output(output)
        assert result.latency_ms_median == 0.5


class TestProfileTool:
    def test_parse_ncu_output(self):
        from cuda_opt_agent.tools.profile import _parse_ncu_output
        fake_output = """
        sm__throughput.avg.pct_of_peak_sustained_elapsed  45.2  45.2
        dram__throughput.avg.pct_of_peak_sustained_elapsed  92.1  92.1
        launch__registers_per_thread  64
        smsp__warp_issue_stalled_long_scoreboard.avg.pct  25.3  25.3
        """
        metrics = _parse_ncu_output(fake_output)
        assert metrics.raw_text == fake_output
        # 部分字段可能解析到,取决于正则匹配

    def test_format_for_prompt(self, sample_ncu_metrics):
        from cuda_opt_agent.tools.profile import format_ncu_for_prompt
        text = format_ncu_for_prompt(sample_ncu_metrics)
        assert "DRAM" in text
        assert "92.1" in text


class TestCorrectnessTool:
    def test_tolerance_map(self):
        from cuda_opt_agent.tools.correctness import get_tolerance
        atol, rtol = get_tolerance("fp16")
        assert atol > 1e-4  # fp16 容差更宽
        assert rtol > 1e-3

        atol32, rtol32 = get_tolerance("fp32")
        assert atol32 < atol  # fp32 更严格

    def test_parse_correctness_output(self):
        from cuda_opt_agent.tools.correctness import _parse_correctness_output
        output = json.dumps({"correct": True, "max_abs_error": 1e-5, "max_rel_error": 1e-4})
        result = _parse_correctness_output(output, 0, 1e-4, 1e-3)
        assert result.correct is True
        assert result.max_abs_error == 1e-5


class TestHardwareTool:
    def test_nvcc_version_no_nvcc(self, monkeypatch):
        import shutil
        monkeypatch.setattr(shutil, "which", lambda x: None)
        from cuda_opt_agent.tools.hardware import _get_nvcc_version
        assert _get_nvcc_version() == "unknown"

    def test_sm_count_fallback(self):
        from cuda_opt_agent.tools.hardware import _query_sm_count_fallback
        assert _query_sm_count_fallback("sm_80") == 108
        assert _query_sm_count_fallback("sm_90") == 132
        assert _query_sm_count_fallback("sm_99") == 0
