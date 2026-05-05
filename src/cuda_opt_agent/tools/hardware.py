"""
硬件信息采集 —— 启动时调用,结果注入所有 Prompt。
使用 pynvml + nvidia-smi + nvcc 探测。
"""

from __future__ import annotations

import logging
import shutil
import subprocess

from ..models.data import HardwareSpec

logger = logging.getLogger(__name__)


def collect_hardware_info() -> HardwareSpec:
    """
    采集当前机器的 GPU 硬件信息。
    优先用 pynvml,不可用时 fallback 到 nvidia-smi 解析。
    """
    spec = HardwareSpec()

    # ── 1) pynvml ──
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        spec.gpu_name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(spec.gpu_name, bytes):
            spec.gpu_name = spec.gpu_name.decode()

        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        spec.compute_capability = f"sm_{major}{minor}"

        try:
            spec.sm_count = pynvml.nvmlDeviceGetNumGpuCores(handle)
        except Exception:
            spec.sm_count = _query_sm_count_fallback(spec.compute_capability)

        spec.has_tensor_cores = major >= 7

        driver = pynvml.nvmlSystemGetDriverVersion()
        spec.driver_version = driver if isinstance(driver, str) else driver.decode()

        pynvml.nvmlShutdown()
    except Exception as e:
        logger.warning("pynvml collection failed; falling back to nvidia-smi: %s", e)
        _fill_from_nvidia_smi(spec)

    # ── 2) nvcc 版本 ──
    spec.cuda_version = _get_nvcc_version()

    # ── 3) 完整 deviceQuery dump ──
    spec.raw_dump = _get_device_query_dump()

    # ── 4) 共享内存 / L2 (尝试从 raw_dump 解析) ──
    _parse_raw_dump_extras(spec)

    return spec


def _get_nvcc_version() -> str:
    """探测 nvcc 版本。"""
    nvcc = shutil.which("nvcc")
    if not nvcc:
        return "unknown"
    try:
        result = subprocess.run(
            [nvcc, "--version"],
            capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.splitlines():
            if "release" in line.lower():
                return line.strip()
        return result.stdout.strip()[:200]
    except Exception:
        return "unknown"


def _get_device_query_dump() -> str:
    """尝试运行 nvidia-smi -q 获取完整设备信息。"""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return ""
    try:
        result = subprocess.run(
            [nvidia_smi, "-q"],
            capture_output=True, text=True, timeout=30,
        )
        return result.stdout[:8000]
    except Exception:
        return ""


def _fill_from_nvidia_smi(spec: HardwareSpec) -> None:
    """fallback: 从 nvidia-smi 解析基础信息。"""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        logger.error("nvidia-smi is unavailable; hardware information will be incomplete")
        return
    try:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=name,compute_cap,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        parts = result.stdout.strip().split(",")
        if len(parts) >= 3:
            spec.gpu_name = parts[0].strip()
            cc = parts[1].strip().replace(".", "")
            spec.compute_capability = f"sm_{cc}"
            spec.driver_version = parts[2].strip()
            major = int(cc[0]) if cc else 0
            spec.has_tensor_cores = major >= 7
    except Exception as e:
        logger.error("Failed to parse nvidia-smi output: %s", e)


def _query_sm_count_fallback(cc: str) -> int:
    """根据已知架构返回 SM 数量近似值。"""
    known = {
        "sm_90": 132,   # H100
        "sm_89": 128,   # RTX 4090
        "sm_86": 82,    # RTX 3090
        "sm_80": 108,   # A100
        "sm_75": 72,    # T4
        "sm_70": 80,    # V100
    }
    return known.get(cc, 0)


def _parse_raw_dump_extras(spec: HardwareSpec) -> None:
    """从 nvidia-smi -q 输出解析 shared_mem 和 l2 信息。"""
    text = spec.raw_dump.lower()
    import re

    # L2 cache
    m = re.search(r"l2 cache size\s*:\s*(\d+)\s*(mb|kb)", text)
    if m:
        val = int(m.group(1))
        unit = m.group(2)
        spec.l2_cache_mb = val if unit == "mb" else val // 1024

    # Shared memory per block (通常在 deviceQuery 中)
    m = re.search(r"shared mem(?:ory)?\s*(?:per (?:block|multiprocessor))?\s*:\s*(\d+)\s*(kb|bytes)", text)
    if m:
        val = int(m.group(1))
        unit = m.group(2)
        spec.shared_mem_per_block_kb = val if unit == "kb" else val // 1024

    # 默认值
    if spec.shared_mem_per_block_kb == 0:
        defaults = {"sm_90": 228, "sm_89": 100, "sm_86": 100, "sm_80": 164, "sm_75": 64, "sm_70": 96}
        spec.shared_mem_per_block_kb = defaults.get(spec.compute_capability, 48)

    if spec.l2_cache_mb == 0:
        defaults = {"sm_90": 50, "sm_89": 72, "sm_86": 6, "sm_80": 40, "sm_75": 4, "sm_70": 6}
        spec.l2_cache_mb = defaults.get(spec.compute_capability, 4)
