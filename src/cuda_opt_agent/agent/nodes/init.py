from __future__ import annotations

import logging

from ...tools.hardware import collect_hardware_info

logger = logging.getLogger(__name__)


def init_node(self, state: dict) -> dict:
    """初始化节点:采集硬件信息。"""
    logger.info("=== INIT: collecting hardware info ===")
    hw = collect_hardware_info()
    logger.info("GPU: %s (%s), CUDA: %s", hw.gpu_name, hw.compute_capability, hw.cuda_version)
    return {"hardware_spec": hw}
