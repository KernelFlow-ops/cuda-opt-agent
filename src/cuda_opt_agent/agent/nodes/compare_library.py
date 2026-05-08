"""
Compare Library 节点 —— 在 bootstrap 后对比 cuDNN/cuBLAS 等价实现。

[改进] 新增节点:
  获取 cuDNN/cuBLAS 的等价实现 latency 作为参照基线。
  若 v0 已接近库实现 (>= 0.9x)，可建议提前终止。

实现方式:
  1. 根据 operator_name 判断是否有对应的库实现
  2. 让 LLM 生成一个使用 cuDNN/cuBLAS 的 benchmark .cu 文件
  3. 编译运行获取 library latency
  4. 结果存入 run_state.library_baseline_ms
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from pathlib import Path

from ...codegen.normalizer import extract_cuda_code
from ...models.data import BenchmarkResult
from ...tools.compile import compile_cuda
from ..temperatures import TEMP_BOOTSTRAP

logger = logging.getLogger(__name__)

# 算子到库函数的映射 (用于判断是否适用)
LIBRARY_OPERATOR_MAP = {
    "batchnorm": "cudnnBatchNormalizationForwardTraining / cudnnBatchNormalizationForwardInference",
    "layernorm": "cudnnNormalizationForwardTraining",
    "softmax": "cudnnSoftmaxForward",
    "gemm": "cublasSgemm / cublasHgemm / cublasGemmEx",
    "conv2d": "cudnnConvolutionForward",
    "relu": "cudnnActivationForward",
    "gelu": "cudnnActivationForward (GELU)",
    "pooling": "cudnnPoolingForward",
    "reduce": "cub::DeviceReduce",
    "scan": "cub::DeviceScan",
}


def _include_dirs_from_env() -> list[Path]:
    dirs: list[Path] = []
    for var in ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH"):
        for entry in os.environ.get(var, "").split(os.pathsep):
            if entry:
                dirs.append(Path(entry))
    return dirs


def _default_include_dirs() -> list[Path]:
    dirs: list[Path] = []
    nvcc = shutil.which("nvcc")
    if nvcc:
        dirs.append(Path(nvcc).resolve().parents[1] / "include")
    dirs.extend([
        Path("/usr/local/cuda/include"),
        Path("/usr/include"),
        Path("/usr/local/include"),
    ])
    dirs.extend(_include_dirs_from_env())
    return dirs


def _header_available(header: str) -> bool:
    seen: set[Path] = set()
    for include_dir in _default_include_dirs():
        resolved = include_dir.expanduser().resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        if (resolved / header).exists():
            return True
    return False


def _missing_required_header(library_function: str) -> str | None:
    required_headers: list[str] = []
    library_lower = library_function.lower()
    if "cudnn" in library_lower:
        required_headers.append("cudnn.h")
    if "cublas" in library_lower:
        required_headers.append("cublas_v2.h")

    for header in required_headers:
        if not _header_available(header):
            return header
    return None


LIBRARY_BENCHMARK_PROMPT = """你是一位 CUDA 性能工程师。请为以下算子编写一个使用 {library_function} 的 benchmark 程序。

## 算子规格
- 名称: {operator_name}
- 签名: {signature}
- 数据类型: {dtypes}
- 张量形状: {shapes}

## 硬件信息
{hardware_summary}

## 要求
1. 编写一个完整的 .cu 文件, 使用 {library_function} 实现等价的算子功能
2. 包含 cudaEvent 计时, 输出 JSON 格式: {{"latency_ms_median": float, "latency_ms_p95": float}}
3. warmup 10 次, 测量 100 次, 取 median
4. 支持命令行参数: --shape key=value [key=value ...] / --warmup N / --rounds N
5. 链接必要的库 (-lcudnn, -lcublas 等)
6. 使用简洁实现, 只需要测速度, 不需要正确性校验
7. 如果库 API 需要 workspace, 请正确分配

请直接输出完整的 .cu 代码, 用 ```cuda 包裹。
"""


async def compare_library_node(self, state: dict) -> dict:
    """
    [改进] 对比 cuDNN/cuBLAS 等库实现的性能。

    若当前算子有对应的库实现, 生成并运行 benchmark 获取 library baseline。
    结果存入 run_state.library_baseline_ms。
    """
    logger.info("=== COMPARE LIBRARY ===")
    op = state["operator_spec"]
    hw = state["hardware_spec"]
    run_state = state["run_state"]

    if not self.sm.config.enable_library_comparison:
        logger.info("Library comparison disabled, skipping")
        return {}

    # 检查是否有对应的库实现
    op_lower = op.name.lower().replace("_", "").replace("-", "")
    library_function = None
    for op_key, lib_func in LIBRARY_OPERATOR_MAP.items():
        if op_key in op_lower:
            library_function = lib_func
            break

    if not library_function:
        logger.info("No library equivalent found for operator '%s', skipping", op.name)
        return {}

    missing_header = _missing_required_header(library_function)
    if missing_header:
        logger.warning(
            "Skipping library comparison for '%s': required header %s not found",
            op.name,
            missing_header,
        )
        return {}

    logger.info("Generating library benchmark for '%s' using %s", op.name, library_function)

    try:
        # Phase 1: LLM 生成库 benchmark 代码
        import json as _json
        prompt = LIBRARY_BENCHMARK_PROMPT.format(
            library_function=library_function,
            operator_name=op.name,
            signature=op.signature,
            dtypes=_json.dumps(op.dtypes),
            shapes=_json.dumps(op.shapes),
            hardware_summary=self._hardware_summary(hw),
        )

        response = await self.llm.ainvoke(prompt, temperature=TEMP_BOOTSTRAP, node_name="compare_library")
        code = extract_cuda_code(response)

        if not code:
            logger.warning("Failed to generate library benchmark code")
            return {}

        # Phase 2: 编译
        lib_dir = self.sm.run_dir / "library_baseline"
        lib_dir.mkdir(parents=True, exist_ok=True)
        code_path = lib_dir / "library_benchmark.cu"
        code_path.write_text(code, encoding="utf-8")

        # 确定链接标志
        link_flags = []
        if "cudnn" in library_function.lower():
            link_flags.extend(["-lcudnn"])
        if "cublas" in library_function.lower():
            link_flags.extend(["-lcublas"])
        if "cub" in library_function.lower():
            pass  # CUB is header-only

        cr = await asyncio.to_thread(
            compile_cuda,
            code_path,
            lib_dir / "library_benchmark",
            hw.compute_capability,
            extra_flags=link_flags,
        )

        if not cr.success:
            logger.warning(
                "Library benchmark compilation failed (this is non-critical): %s",
                cr.stderr[:500],
            )
            return {}

        # Phase 3: 运行 benchmark
        exe_path = Path(cr.output_path)
        bm = await asyncio.to_thread(self._benchmark_multi, exe_path, op)

        if bm.latency_ms_median > 0:
            run_state.library_baseline_ms = bm.latency_ms_median
            self.sm._save()
            logger.info(
                "Library baseline: %.4f ms (v0 best: %.4f ms, ratio: %.2f)",
                bm.latency_ms_median,
                run_state.best_latency_ms() or 0,
                (run_state.best_latency_ms() or 1) / bm.latency_ms_median,
            )
        else:
            logger.warning("Library benchmark returned zero latency")

    except Exception as e:
        # 库对比是非关键路径, 失败不应影响主流程
        logger.warning("Library comparison failed (non-critical): %s", e)

    return {}
