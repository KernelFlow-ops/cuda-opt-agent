"""
pytest 共享 fixtures。
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

# 确保 .env 不干扰测试
os.environ.setdefault("LLM_PROVIDER", "anthropic")
os.environ.setdefault("RUNS_DIR", "test_runs")
os.environ.setdefault("KNOWLEDGE_BASE_DIR", "test_kb")


@pytest.fixture
def tmp_dir(tmp_path):
    """提供临时目录。"""
    return tmp_path


@pytest.fixture
def sample_operator_spec():
    """示例算子规格。"""
    from cuda_opt_agent.models.data import OperatorSpec
    return OperatorSpec(
        name="gemm",
        signature="C = A @ B, A:[M,K] B:[K,N]",
        dtypes={"A": "fp16", "B": "fp16", "C": "fp16"},
        shapes={"A": [4096, 4096], "B": [4096, 4096]},
        constraints=["M, N, K 均为 128 的倍数"],
    )


@pytest.fixture
def sample_hardware_spec():
    """示例硬件规格。"""
    from cuda_opt_agent.models.data import HardwareSpec
    return HardwareSpec(
        gpu_name="NVIDIA A100-SXM4-80GB",
        compute_capability="sm_80",
        sm_count=108,
        shared_mem_per_block_kb=164,
        l2_cache_mb=40,
        has_tensor_cores=True,
        cuda_version="CUDA 12.4",
        driver_version="550.54.14",
        raw_dump="(mock device query output)",
    )


@pytest.fixture
def sample_agent_config(tmp_dir):
    """示例 Agent 配置。"""
    from cuda_opt_agent.models.data import AgentConfig
    return AgentConfig(
        llm_provider="anthropic",
        llm_model="claude-sonnet-4-20250514",
        max_iterations=5,
        consecutive_reject_limit=3,
        accept_epsilon=0.005,
        compile_repair_max_retries=2,
        hp_candidate_count=3,
        benchmark_warmup_rounds=2,
        benchmark_measure_rounds=5,
        runs_dir=str(tmp_dir / "runs"),
        knowledge_base_dir=str(tmp_dir / "kb"),
    )


@pytest.fixture
def sample_benchmark_result():
    """示例 benchmark 结果。"""
    from cuda_opt_agent.models.data import BenchmarkResult
    return BenchmarkResult(
        latency_ms_median=1.234,
        latency_ms_p95=1.456,
        throughput_gflops=5678.9,
    )


@pytest.fixture
def sample_ncu_metrics():
    """示例 ncu 指标。"""
    from cuda_opt_agent.models.data import NcuMetrics
    return NcuMetrics(
        sm_throughput_pct=45.2,
        compute_memory_throughput_pct=78.5,
        dram_throughput_pct=92.1,
        l1_throughput_pct=35.0,
        l2_throughput_pct=41.3,
        occupancy_pct=62.5,
        registers_per_thread=64,
        shared_mem_per_block_bytes=49152,
        stall_reasons={"long_scoreboard": 25.3, "mio_throttle": 12.1},
        raw_text="(mock ncu output)",
    )


@pytest.fixture
def sample_iteration_record(sample_benchmark_result):
    """示例迭代记录。"""
    from cuda_opt_agent.models.data import IterationRecord
    return IterationRecord(
        version_id="v0",
        parent_id=None,
        method_name=None,
        has_hyperparams=False,
        code_path="iterv0/code.cu",
        compile_ok=True,
        correctness_ok=True,
        benchmark=sample_benchmark_result,
        accepted=True,
    )


@pytest.fixture
def sample_run_state(sample_operator_spec, sample_hardware_spec, sample_agent_config, sample_iteration_record):
    """示例运行状态。"""
    from cuda_opt_agent.models.data import RunState, RunStatus
    return RunState(
        run_id="gemm_run_20260501T120000",
        operator_spec=sample_operator_spec,
        hardware_spec=sample_hardware_spec,
        iterations=[sample_iteration_record],
        current_best_id="v0",
        status=RunStatus.RUNNING,
        config=sample_agent_config,
    )


@pytest.fixture
def sample_cuda_code():
    """示例 CUDA 代码。"""
    return '''
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vector_add(float* A, float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    // simple test
    return 0;
}
'''
