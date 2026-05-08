"""
持久化管理 —— 运行目录创建、state.json / history.jsonl 的读写。
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import orjson

from ..models.data import IterationRecord, RunState, RunStatus

logger = logging.getLogger(__name__)
_NATIVE_PATH = type(Path("."))


class PersistenceManager:
    """管理单次运行的产物目录与持久化。"""

    def __init__(self, runs_dir: str = "runs"):
        self.runs_dir = _NATIVE_PATH(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def create_run_dir(self, operator_name: str) -> Path:
        """创建新的运行目录: {op_name}_run_{timestamp}。"""
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        run_id = f"{operator_name}_run_{ts}"
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def find_latest_unfinished_run(self, operator_name: str) -> Path | None:
        """查找同名算子最近的未完成运行。"""
        candidates = []
        for d in self.runs_dir.iterdir():
            if d.is_dir() and d.name.startswith(f"{operator_name}_run_"):
                state_file = d / "state.json"
                if state_file.exists():
                    try:
                        state = self.load_state(d)
                        if state.status in (RunStatus.RUNNING, RunStatus.PAUSED):
                            candidates.append((d, state.updated_at))
                    except Exception:
                        continue
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    # ── State 读写 ──

    def save_state(self, state: RunState, run_dir: Path) -> None:
        """保存 state.json (原子写入)。"""
        state.touch()
        state_path = run_dir / "state.json"
        tmp_path = run_dir / "state.json.tmp"
        data = orjson.dumps(state.model_dump(), option=orjson.OPT_INDENT_2)
        with open(tmp_path, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        tmp_path.replace(state_path)

        # Directory fsync is POSIX-only; Windows rejects opening directories.
        if os.name != "nt":
            fd = os.open(str(run_dir), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)

    def load_state(self, run_dir: Path) -> RunState:
        """加载 state.json。"""
        state_path = run_dir / "state.json"
        if not state_path.exists():
            raise FileNotFoundError(f"state.json not found: {state_path}")
        data = orjson.loads(state_path.read_bytes())
        return RunState.model_validate(data)

    # ── History 追加 ──

    def append_history(self, record: IterationRecord, run_dir: Path) -> None:
        """向 history.jsonl 追加一行。"""
        history_path = run_dir / "history.jsonl"
        line = orjson.dumps(record.model_dump()).decode() + "\n"
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(line)

    def load_history(self, run_dir: Path) -> list[IterationRecord]:
        """从 history.jsonl 重建迭代记录(降级模式)。"""
        history_path = run_dir / "history.jsonl"
        if not history_path.exists():
            return []
        records = []
        for line in history_path.read_text(encoding="utf-8").strip().split("\n"):
            if line.strip():
                data = orjson.loads(line)
                records.append(IterationRecord.model_validate(data))
        return records

    # ── 迭代目录 ──

    def create_iteration_dir(self, run_dir: Path, version_id: str) -> Path:
        """创建迭代子目录。"""
        iter_dir = run_dir / f"iter{version_id}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        return iter_dir

    def save_code(self, code: str, iter_dir: Path, filename: str = "code.cu") -> Path:
        """保存 kernel 代码。"""
        code_path = iter_dir / filename
        code_path.write_text(code, encoding="utf-8")
        return code_path

    def save_reasoning(self, reasoning: str, iter_dir: Path) -> Path:
        """保存推理过程。"""
        reasoning_path = iter_dir / "reasoning.md"
        reasoning_path.write_text(reasoning, encoding="utf-8")
        return reasoning_path

    def save_benchmark(self, result: dict, iter_dir: Path) -> Path:
        """保存 benchmark 结果。"""
        bm_path = iter_dir / "benchmark.json"
        bm_path.write_bytes(orjson.dumps(result, option=orjson.OPT_INDENT_2))
        return bm_path

    def save_config(self, config: dict, run_dir: Path) -> Path:
        """保存运行配置。"""
        config_path = run_dir / "config.json"
        config_path.write_bytes(orjson.dumps(config, option=orjson.OPT_INDENT_2))
        return config_path

    def update_best_symlink(self, run_dir: Path, iter_dir: Path) -> None:
        """更新 best 指针。

        best.txt 是跨平台主机制。Windows 默认不创建目录 symlink,避免普通
        用户权限下触发 WinError 1314;如确实需要 symlink,设置
        ENABLE_BEST_SYMLINK=1。
        """
        if not iter_dir.exists():
            raise FileNotFoundError(f"best target directory not found: {iter_dir}")
        best_dir = run_dir / "best"
        best_target = iter_dir.relative_to(run_dir)
        (run_dir / "best.txt").write_text(str(best_target), encoding="utf-8")

        enable_symlink = os.getenv("ENABLE_BEST_SYMLINK", "").lower() in {"1", "true", "yes", "on"}
        if os.name == "nt" and not enable_symlink:
            if best_dir.is_symlink():
                best_dir.unlink()
            return

        if best_dir.is_symlink():
            best_dir.unlink()
        elif best_dir.exists():
            logger.info("best path exists and is not a symlink; best.txt was written: %s", best_dir)
            return

        try:
            best_dir.symlink_to(best_target, target_is_directory=True)
        except OSError as e:
            if os.name == "nt" and getattr(e, "winerror", None) == 1314:
                logger.debug("Windows lacks permission to create best symlink; best.txt was written")
            else:
                logger.info("Could not create best symlink; best.txt was written: %s", e)

    def save_reasoning_log(self, text: str, run_dir: Path) -> None:
        """向全局 reasoning_log.md 追加。"""
        log_path = run_dir / "reasoning_log.md"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(text + "\n\n---\n\n")

    # ── 状态恢复 ──

    def try_recover_state(self, run_dir: Path) -> RunState | None:
        """
        尝试从多源恢复状态:
        1. state.json (权威源)
        2. history.jsonl (降级)
        3. 目录结构 (最后手段)
        """
        # 1) state.json
        state_path = run_dir / "state.json"
        if state_path.exists():
            try:
                return self.load_state(run_dir)
            except Exception as e:
                logger.warning("state.json is corrupted: %s; trying fallback recovery", e)

        # 2) history.jsonl
        records = self.load_history(run_dir)
        if records:
            logger.info("Rebuilding state from history.jsonl (%d records)", len(records))
            state = RunState(
                run_id=run_dir.name,
                iterations=records,
                status=RunStatus.PAUSED,
            )
            # 找到最后一个 accepted 的作为 best
            for rec in reversed(records):
                if rec.accepted:
                    state.current_best_id = rec.version_id
                    break
            return state

        logger.error("Could not recover state: %s", run_dir)
        return None
