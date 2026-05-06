from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def terminate_node(self, state: dict) -> dict:
    """终止节点:生成最终报告。"""
    logger.info("=== TERMINATE ===")
    run_state = state["run_state"]
    self.sm.mark_done()

    if self.sm.run_dir:
        report = self._generate_final_report(run_state)
        (self.sm.run_dir / "final_report.md").write_text(report, encoding="utf-8")

    return {"should_stop": True, "stop_reason": state.get("stop_reason", "done")}
