from __future__ import annotations

import asyncio
import logging

from ...tools.hardware import collect_hardware_info

logger = logging.getLogger(__name__)


async def init_node(self, state: dict) -> dict:
    """初始化节点:采集硬件信息。"""
    logger.info("=== INIT: collecting hardware info ===")
    hw = await asyncio.to_thread(collect_hardware_info)
    logger.info("GPU: %s (%s), CUDA: %s", hw.gpu_name, hw.compute_capability, hw.cuda_version)
    if self.sm.state:
        self.sm.state.hardware_spec = hw
        await asyncio.to_thread(self.sm._save)
    return {"hardware_spec": hw, "run_state": self.sm.state or state.get("run_state")}
