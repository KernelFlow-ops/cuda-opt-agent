from __future__ import annotations

import asyncio
import logging
import sys

from ...models.data import BenchmarkResult, NcuMetrics
from ...tools.profile import run_ncu_profile
from ...tools.ref_eval import run_ref_benchmark_multi

logger = logging.getLogger(__name__)


async def profile_best_node(self, state: dict) -> dict:
    """对当前 best 进行 benchmark + ncu profiling。"""
    logger.info("=== PROFILE BEST ===")
    run_state = state["run_state"]
    best = run_state.iter_by_id(run_state.current_best_id)

    if best is None:
        return {"error": "Best version not found"}

    best_dir = self.sm.run_dir / f"iter{best.version_id}"
    code_path = self.sm.run_dir / best.code_path if best.code_path else best_dir / "code.cu"
    ref_path = self._ref_py_path(state, self.sm.run_dir)

    if ref_path and ref_path.exists():
        op = run_state.operator_spec
        hw = run_state.hardware_spec
        dtype = list(op.dtypes.values())[0] if op.dtypes else "fp32"
        bm = await asyncio.to_thread(
            run_ref_benchmark_multi,
            ref_path,
            code_path,
            self._active_shape_profiles(op),
            func_name=self._kernel_function_name(op),
            compute_capability=hw.compute_capability,
            dtype=dtype,
            warmup_rounds=self.sm.config.benchmark_warmup_rounds,
            measure_rounds=self.sm.config.benchmark_measure_rounds,
            aggregator=self.sm.config.multi_shape_aggregator,
        )
        executable_args = [
            str(ref_path),
            "--cuda", str(code_path),
            "--func", self._kernel_function_name(op),
            "--benchmark",
            "--arch", hw.compute_capability,
            "--dtype", dtype,
        ]
        executable_args.extend(self._profile_args_from_benchmark(bm))
        ncu = await asyncio.to_thread(
            run_ncu_profile,
            sys.executable,
            output_report_path=best_dir / "ncu_report.txt",
            kernel_name=f"regex:.*{op.name}.*",
            extra_args=["--kernel-name-base", "demangled"],
            executable_args=executable_args,
            launch_count=self.sm.config.ncu_launch_count,
        )
    else:
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
