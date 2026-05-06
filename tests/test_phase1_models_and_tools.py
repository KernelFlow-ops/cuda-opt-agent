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

    def test_shape_profiles_fill_shapes(self):
        from cuda_opt_agent.models.data import OperatorSpec
        spec = OperatorSpec(
            name="softmax",
            signature="y[B,N] = softmax(x[B,N], dim=-1)",
            dtypes={"x": "fp16", "y": "fp16"},
            shape_profiles=[{"x": [1024, 1024], "y": [1024, 1024]}],
            task_description="Use max-trick for numerical stability.",
        )
        assert spec.shapes == {"x": [1024, 1024], "y": [1024, 1024]}
        assert spec.seed_code_path is None


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

    def test_run_benchmark_multi_worst_aggregator(self, monkeypatch, tmp_dir):
        from cuda_opt_agent.models.data import BenchmarkResult
        import cuda_opt_agent.tools.benchmark as benchmark_module

        exe = tmp_dir / "kernel.exe"
        exe.write_text("", encoding="utf-8")
        calls = []

        def fake_run_benchmark(executable_path, **kwargs):
            calls.append(kwargs.get("extra_args"))
            latency = 1.0 if len(calls) == 1 else 3.0
            return BenchmarkResult(latency_ms_median=latency, latency_ms_p95=latency + 0.1)

        monkeypatch.setattr(benchmark_module, "run_benchmark", fake_run_benchmark)
        result = benchmark_module.run_benchmark_multi(
            exe,
            [{"M": 1024, "N": 1024, "K": 1024}, {"M": 2048, "N": 2048, "K": 2048}],
            aggregator="worst",
        )

        assert result.latency_ms_median == 3.0
        assert result.extra["shape_count"] == 2
        assert result.extra["worst_shape"] == {"M": 2048, "N": 2048, "K": 2048}
        assert calls[0] == ["--shape", "M=1024", "N=1024", "K=1024"]
        assert calls[1] == ["--shape", "M=2048", "N=2048", "K=2048"]


class TestShapeProfiles:
    def test_parse_shape_profiles_power_syntax(self):
        from cuda_opt_agent.shape_profiles import parse_shape_profiles
        profiles = parse_shape_profiles("gemm", "1024^3;2048^3")
        assert profiles == [
            {"M": 1024, "N": 1024, "K": 1024},
            {"M": 2048, "N": 2048, "K": 2048},
        ]

    def test_shape_profile_to_args(self):
        from cuda_opt_agent.shape_profiles import shape_profile_to_args
        args = shape_profile_to_args({"x": [1024, 1024], "_weight": 2})
        assert args == ["--shape", "x=1024,1024"]


class TestProfileTool:
    def test_parse_ncu_output(self):
        from cuda_opt_agent.tools.profile import _parse_ncu_output
        fake_output = """==PROF== Connected to process
"ID","Metric Name","Metric Unit","Metric Value"
"1","sm__throughput.avg.pct_of_peak_sustained_elapsed","%","45.2"
"1","gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed","%","51.5"
"1","dram__throughput.avg.pct_of_peak_sustained_elapsed","%","92.1"
"1","launch__registers_per_thread","register/thread","64"
"1","launch__shared_mem_per_block","byte","1,024"
"1","smsp__warp_issue_stalled_long_scoreboard.avg.pct_of_peak_sustained_elapsed","%","25.3"
"""
        metrics = _parse_ncu_output(fake_output)
        assert metrics.raw_text == fake_output
        assert metrics.sm_throughput_pct == 45.2
        assert metrics.compute_memory_throughput_pct == 51.5
        assert metrics.dram_throughput_pct == 92.1
        assert metrics.registers_per_thread == 64
        assert metrics.shared_mem_per_block_bytes == 1024
        assert metrics.stall_reasons["long_scoreboard"] == 25.3
        assert metrics.extra["parser"] == "ncu_csv_raw"

    def test_parse_ncu_output_rejects_non_csv_text(self):
        from cuda_opt_agent.tools.profile import _parse_ncu_output
        fake_output = """
        sm__throughput.avg.pct_of_peak_sustained_elapsed  45.2  45.2
        dram__throughput.avg.pct_of_peak_sustained_elapsed  92.1  92.1
        """
        metrics = _parse_ncu_output(fake_output)
        assert metrics.sm_throughput_pct is None
        assert metrics.dram_throughput_pct is None
        assert metrics.extra["parse_error"] == "ncu csv header not found"

    def test_parse_ncu_output_wide_csv(self):
        from cuda_opt_agent.tools.profile import _parse_ncu_output
        fake_output = '''==PROF== Connected
"ID","Kernel Name","sm__throughput.avg.pct_of_peak_sustained_elapsed","gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed","dram__throughput.avg.pct_of_peak_sustained_elapsed","sm__warps_active.avg.pct_of_peak_sustained_elapsed","gpu__time_duration.sum"
"","","%","%","%","%","nsecond"
"0","rmsnorm_kernel","31.16","60.89","86.74","12","29696"
'''
        metrics = _parse_ncu_output(fake_output)
        assert metrics.sm_throughput_pct == 31.16
        assert metrics.compute_memory_throughput_pct == 60.89
        assert metrics.dram_throughput_pct == 86.74
        assert metrics.occupancy_pct == 12.0
        assert metrics.extra["parser"] == "ncu_csv_raw"
        assert metrics.extra["parser_format"] == "wide"
        assert metrics.extra["metrics"]["gpu__time_duration.sum"] == 29696

    def test_run_ncu_profile_uses_csv_raw_page(self, monkeypatch, tmp_dir):
        import cuda_opt_agent.tools.profile as profile_module

        exe = tmp_dir / "kernel.exe"
        exe.write_text("", encoding="utf-8")
        captured = {}

        class Result:
            returncode = 0
            stdout = '"Metric Name","Metric Unit","Metric Value"\n"sm__throughput.avg.pct_of_peak_sustained_elapsed","%","12.5"\n'
            stderr = ""

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            return Result()

        monkeypatch.setattr(profile_module.shutil, "which", lambda name: "ncu")
        monkeypatch.setattr(profile_module.subprocess, "run", fake_run)

        metrics = profile_module.run_ncu_profile(exe, output_report_path=tmp_dir / "ncu.csv")

        assert "--csv" in captured["cmd"]
        assert captured["cmd"][captured["cmd"].index("--page") + 1] == "raw"
        assert captured["cmd"][captured["cmd"].index("--launch-count") + 1] == "3"
        assert captured["cmd"][captured["cmd"].index("--warmup") + 1] == "1"
        assert captured["cmd"][captured["cmd"].index("--rounds") + 1] == "1"
        assert metrics.sm_throughput_pct == 12.5

    def test_run_ncu_profile_accepts_custom_launch_count(self, monkeypatch, tmp_dir):
        import cuda_opt_agent.tools.profile as profile_module

        exe = tmp_dir / "kernel.exe"
        exe.write_text("", encoding="utf-8")
        captured = {}

        class Result:
            returncode = 0
            stdout = '"Metric Name","Metric Unit","Metric Value"\n"sm__throughput.avg.pct_of_peak_sustained_elapsed","%","12.5"\n'
            stderr = ""

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            return Result()

        monkeypatch.setattr(profile_module.shutil, "which", lambda name: "ncu")
        monkeypatch.setattr(profile_module.subprocess, "run", fake_run)

        profile_module.run_ncu_profile(exe, output_report_path=tmp_dir / "ncu.csv", launch_count=5)

        assert captured["cmd"][captured["cmd"].index("--launch-count") + 1] == "5"

    def test_classify_ncu_bottleneck_paths(self):
        import cuda_opt_agent.tools.profile as profile_module
        from cuda_opt_agent.models.data import NcuMetrics

        def make_metrics(values):
            metrics = NcuMetrics(raw_text="")
            metrics.extra["metrics"] = values
            profile_module._populate_ncu_metric_fields(metrics)
            return metrics

        memory_metrics = make_metrics({
            profile_module.METRIC_SM_THROUGHPUT: 30.0,
            profile_module.METRIC_COMPUTE_MEMORY_THROUGHPUT: 72.0,
            profile_module.METRIC_DRAM_THROUGHPUT: 68.0,
            profile_module.METRIC_OCCUPANCY: 65.0,
        })
        compute_metrics = make_metrics({
            profile_module.METRIC_SM_THROUGHPUT: 76.0,
            profile_module.METRIC_DRAM_THROUGHPUT: 32.0,
            profile_module.METRIC_OCCUPANCY: 70.0,
        })
        latency_metrics = make_metrics({
            profile_module.METRIC_SM_THROUGHPUT: 18.0,
            profile_module.METRIC_DRAM_THROUGHPUT: 12.0,
            profile_module.METRIC_OCCUPANCY: 35.0,
        })

        assert profile_module.classify_ncu_bottleneck(memory_metrics) == "memory_bound"
        assert profile_module.classify_ncu_bottleneck(compute_metrics) == "compute_bound"
        assert profile_module.classify_ncu_bottleneck(latency_metrics) == "latency_bound"

    def test_adaptive_profile_triggers_phase3_on_memory_saturation(self, monkeypatch, tmp_dir):
        import cuda_opt_agent.tools.profile as profile_module
        from cuda_opt_agent.models.data import NcuMetrics

        exe = tmp_dir / "kernel.exe"
        exe.write_text("", encoding="utf-8")
        calls = []

        def make_metrics(values, phase):
            metrics = NcuMetrics(raw_text=phase)
            metrics.extra.update({
                "phase": phase,
                "requested_metrics": list(values),
                "metrics": values,
                "metric_units": {},
            })
            profile_module._populate_ncu_metric_fields(metrics)
            return metrics

        def fake_run_ncu_metrics(executable_path, metrics, *, phase_name, **kwargs):
            calls.append((phase_name, metrics))
            if phase_name == "phase1":
                return make_metrics({
                    profile_module.METRIC_SM_THROUGHPUT: 35.0,
                    profile_module.METRIC_COMPUTE_MEMORY_THROUGHPUT: 78.0,
                    profile_module.METRIC_DRAM_THROUGHPUT: 88.0,
                    profile_module.METRIC_OCCUPANCY: 62.0,
                }, phase_name)
            if phase_name == "phase2_memory":
                return make_metrics({
                    profile_module.METRIC_L2_HIT_RATE: 22.0,
                    profile_module.METRIC_LONG_SCOREBOARD: 31.0,
                }, phase_name)
            return make_metrics({
                profile_module.METRIC_FMA_PIPE: 45.0,
                profile_module.METRIC_TENSOR_PIPE: 0.0,
                profile_module.METRIC_MATH_PIPE_THROTTLE: 8.0,
            }, phase_name)

        monkeypatch.setattr(profile_module, "_run_ncu_metrics", fake_run_ncu_metrics)
        result = profile_module.run_adaptive_ncu_profile(exe, output_report_path=tmp_dir / "ncu.txt")

        assert [phase for phase, _ in calls] == ["phase1", "phase2_memory", "phase3_complementary"]
        assert calls[1][1] == profile_module.MEMORY_METRICS
        assert calls[2][1] == profile_module.PHASE3_COMPUTE_TOP3
        assert result.extra["diagnosis"]["classification"] == "memory_bound"
        assert result.extra["diagnosis"]["saturation"]["is_saturated"] is True
        assert "算术强度" in result.extra["diagnosis"]["recommendation_hint"]

    def test_format_for_prompt(self, sample_ncu_metrics):
        from cuda_opt_agent.tools.profile import format_ncu_for_prompt
        text = format_ncu_for_prompt(sample_ncu_metrics)
        assert "DRAM" in text
        assert "92.1" in text
        assert "classification" in text


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

    def test_check_correctness_multi_passes_shape_args(self, monkeypatch, tmp_dir):
        import cuda_opt_agent.tools.correctness as correctness_module
        from cuda_opt_agent.tools.correctness import CorrectnessResult

        exe = tmp_dir / "kernel.exe"
        exe.write_text("", encoding="utf-8")
        calls = []

        def fake_check_correctness(executable_path, **kwargs):
            calls.append(kwargs.get("extra_args"))
            return CorrectnessResult(correct=True, max_abs_error=0.0, max_rel_error=0.0, message="ok")

        monkeypatch.setattr(correctness_module, "check_correctness", fake_check_correctness)
        results = correctness_module.check_correctness_multi(
            exe,
            [{"B": 1024, "N": 1024}, {"B": 4096, "N": 4096}],
            dtype="fp16",
        )

        assert all(r["correct"] for r in results)
        assert calls[0] == ["--shape", "B=1024", "N=1024"]
        assert calls[1] == ["--shape", "B=4096", "N=4096"]


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
