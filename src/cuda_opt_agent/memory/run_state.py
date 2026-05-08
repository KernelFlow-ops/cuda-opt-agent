"""
RunStateManager —— 封装 RunState 的创建、修改、续跑逻辑。

[改进]:
  - add_to_blacklist 支持 subspace / pattern_signature / regression_severity / trigger_conditions
  - should_stop 强制只按 max_iterations 停止
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..models.data import (
    AgentConfig,
    BlacklistEntry,
    HardwareSpec,
    IterationRecord,
    OperatorSpec,
    RunState,
    RunStatus,
)
from ..models.enums import normalize_method_name
from .persistence import PersistenceManager

logger = logging.getLogger(__name__)


class RunStateManager:
    """封装 RunState 的创建、修改、续跑逻辑。"""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.persistence = PersistenceManager(config.runs_dir)
        self.state: RunState | None = None
        self.run_dir: Path | None = None

    def init_new_run(
        self, operator_spec: OperatorSpec, hardware_spec: HardwareSpec,
    ) -> RunState:
        """创建新运行。"""
        self.run_dir = self.persistence.create_run_dir(operator_spec.name)
        self.state = RunState(
            run_id=self.run_dir.name,
            operator_spec=operator_spec,
            hardware_spec=hardware_spec,
            config=self.config,
            status=RunStatus.RUNNING,
        )
        self._save()
        self.persistence.save_config(self.config.model_dump(), self.run_dir)
        logger.info("Created run: %s", self.run_dir)
        return self.state

    def new_run(self, operator_spec: OperatorSpec) -> RunState:
        """创建新运行；硬件信息稍后由 init 节点采集并回填。"""
        return self.init_new_run(operator_spec, HardwareSpec())

    def resume_run(
        self, operator_name: str | None = None, run_dir: str | Path | None = None,
    ) -> RunState | None:
        """续跑已有运行。"""
        if run_dir:
            self.run_dir = Path(run_dir)
        elif operator_name:
            self.run_dir = self.persistence.find_latest_unfinished_run(operator_name)
        else:
            return None

        if self.run_dir is None or not self.run_dir.exists():
            logger.error("No resumable run found")
            return None

        self.state = self.persistence.try_recover_state(self.run_dir)
        if self.state is None:
            return None

        self.state.status = RunStatus.RUNNING
        self._save()
        logger.info("Resuming run: %s (%d existing iterations)", self.run_dir, len(self.state.iterations))
        return self.state

    def add_iteration(self, record: IterationRecord) -> None:
        """添加一次迭代记录。"""
        assert self.state is not None
        self.state.iterations.append(record)
        if record.accepted:
            self.state.current_best_id = record.version_id
        self.persistence.append_history(record, self.run_dir)
        self._save()

    def add_to_blacklist(
        self,
        method_name: str,
        reason: str,
        hp_constraint: dict | None = None,
        failed_at_version: str = "",
        subspace: str | None = None,
        pattern_signature: str | None = None,
        regression_severity: str | None = None,
        trigger_conditions: dict | None = None,
    ) -> None:
        """
        将方法/超参添加到黑名单。

        [改进] 新增 subspace / pattern_signature / regression_severity / trigger_conditions
        参数，支持子空间级黑名单匹配。
        """
        assert self.state is not None
        entry = BlacklistEntry(
            method_name_normalized=normalize_method_name(method_name),
            hyperparam_constraint=hp_constraint,
            failed_at_version=failed_at_version,
            reason=reason,
            subspace=subspace,
            pattern_signature=pattern_signature,
            regression_severity=regression_severity,
            trigger_conditions=trigger_conditions,
        )
        self.state.blacklist.append(entry)
        self._save()
        logger.info(
            "Blacklisted: %s (subspace=%s, pattern=%s, severity=%s)",
            entry.method_name_normalized, subspace, pattern_signature, regression_severity,
        )

    def update_best(self, version_id: str, iter_dir: Path | None = None) -> None:
        """更新 current best。"""
        assert self.state is not None
        self.state.current_best_id = version_id
        if iter_dir and self.run_dir:
            self.persistence.update_best_symlink(self.run_dir, iter_dir)
        self._save()

    def mark_done(self) -> None:
        """标记运行完成。"""
        assert self.state is not None
        self.state.status = RunStatus.DONE
        self._save()

    def mark_failed(self) -> None:
        """标记运行失败。"""
        assert self.state is not None
        self.state.status = RunStatus.FAILED
        self._save()

    def should_stop(self) -> tuple[bool, str]:
        """
        检查是否满足终止条件。

        强制跑满 max_iterations；连续 reject、correctness 失败、catastrophic
        regression、tiny kernel 等信号只作为后续 prompt 的上下文，不再触发早停。

        Returns:
            (should_stop, reason)
        """
        assert self.state is not None
        s = self.state
        c = self.config

        if len(s.iterations) >= c.max_iterations:
            return True, f"Reached maximum iterations ({c.max_iterations})"

        return False, ""

    def get_best_iteration(self) -> IterationRecord | None:
        """获取当前 best 的迭代记录。"""
        if self.state is None or not self.state.current_best_id:
            return None
        return self.state.iter_by_id(self.state.current_best_id)

    def create_iteration_dir(self, version_id: str) -> Path:
        """创建迭代目录。"""
        assert self.run_dir is not None
        return self.persistence.create_iteration_dir(self.run_dir, version_id)

    def _save(self) -> None:
        """保存当前状态。"""
        if self.state and self.run_dir:
            self.persistence.save_state(self.state, self.run_dir)
