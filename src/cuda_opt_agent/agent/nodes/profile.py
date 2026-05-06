from __future__ import annotations

import asyncio
import logging

from ...models.data import BenchmarkResult, NcuMetrics
from ...tools.profile import run_ncu_profile

logger = logging.getLogger(__name__)


async def profile_best_node(self, state: dict) -> dict:
    """对当前 best 进行 benchmark + ncu profiling。"""
    logger.info("=== PROFILE BEST ===")
    run_state = state["run_state"]
    best = run_state.iter_by_id(run_state.current_best_id)

    if best is None:
        return {"error": "Best version not found"}

    best_dir = self.sm.run_dir / f"iter{best.version_id}"
    exe_path = self._kernel_executable(best_dir)

    if not exe_path.exists():
        raise FileNotFoundError(f"Best executable not found: {exe_path}")

    bm = await asyncio.to_thread(self._benchmark_multi, exe_path, run_state.operator_spec)
    ncu = await asyncio.to_thread(
        run_ncu_profile,
        exe_path,
        output_report_path=best_dir / "ncu_report.txt",
        executable_args=self._profile_args_from_benchmark(bm),
        launch_count=self.sm.config.ncu_launch_count,
    )

    code_path = self.sm.run_dir / best.code_path if best.code_path else best_dir / "code.cu"
    code = await asyncio.to_thread(code_path.read_text, encoding="utf-8") if code_path.exists() else ""

    best.benchmark = bm
    best.ncu_metrics = ncu
    best.ncu_report_path = str((best_dir / "ncu_report.txt").relative_to(self.sm.run_dir))
    await asyncio.to_thread(self.sm._save)

    return {
        "current_benchmark": bm,
        "current_ncu": ncu,
        "current_code": code,
        "run_state": self.sm.state,
    }
