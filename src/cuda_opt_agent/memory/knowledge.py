"""
跨运行知识库 —— 按算子大类组织,提供软提示。
哲学:初始为空,随运行积累,永远以软提示形式注入 Prompt。
"""

from __future__ import annotations

import logging
from pathlib import Path

import orjson

from ..models.data import KnowledgeEntry, Outcome, _now_iso

logger = logging.getLogger(__name__)

# EWMA 衰减因子
EWMA_ALPHA = 0.3


class KnowledgeBase:
    """跨运行知识库管理。"""

    def __init__(self, kb_dir: str | Path = "knowledge_base"):
        self.kb_dir = Path(kb_dir)
        self.kb_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, list[KnowledgeEntry]] = {}

    def _file_for(self, operator_class: str) -> Path:
        """返回某算子类别对应的文件路径。"""
        safe_name = operator_class.lower().replace(" ", "_").replace("/", "_")
        return self.kb_dir / f"{safe_name}.json"

    def _load_file(self, path: Path) -> list[KnowledgeEntry]:
        """加载单个 JSON 文件。"""
        if not path.exists():
            return []
        try:
            data = orjson.loads(path.read_bytes())
            return [KnowledgeEntry.model_validate(entry) for entry in data]
        except Exception as e:
            logger.warning("Failed to load knowledge file %s: %s", path, e)
            return []

    def _save_file(self, path: Path, entries: list[KnowledgeEntry]) -> None:
        """保存到 JSON 文件。"""
        data = [entry.model_dump() for entry in entries]
        path.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))

    def query(
        self,
        operator_class: str,
        hardware_signature: str,
        top_k: int = 10,
    ) -> list[KnowledgeEntry]:
        """
        查询与当前 (算子, 硬件) 匹配的知识条目。

        同时附上全局知识 (_global.json) 的高分条目。
        """
        results = []

        # 1) 算子特定知识
        op_entries = self._load_file(self._file_for(operator_class))
        matched = [e for e in op_entries if e.hardware_signature == hardware_signature]
        # 按 aggregate_speedup * confidence 排序
        matched.sort(key=lambda e: e.aggregate_speedup * e.confidence, reverse=True)
        results.extend(matched[:top_k])

        # 2) 全局知识
        global_entries = self._load_file(self._file_for("_global"))
        global_matched = [e for e in global_entries if e.hardware_signature == hardware_signature]
        global_matched.sort(key=lambda e: e.aggregate_speedup * e.confidence, reverse=True)
        results.extend(global_matched[:3])

        return results

    def format_hints_for_prompt(self, entries: list[KnowledgeEntry]) -> str:
        """格式化知识条目为 Prompt 软提示。"""
        if not entries:
            return "（暂无历史经验）"

        lines = []
        for i, entry in enumerate(entries, 1):
            speedup = entry.aggregate_speedup
            direction = "加速" if speedup > 1.0 else "减速"
            hp_info = ""
            if entry.hyperparams_pattern:
                hp_info = f", 超参模式: {entry.hyperparams_pattern}"
            lines.append(
                f"{i}. [{entry.operator_class}] 方法: {entry.method_name}{hp_info}\n"
                f"   观测: {direction} {speedup:.2f}x, 置信度: {entry.confidence:.2f}\n"
                f"   备注: {entry.notes}"
            )
        return "\n".join(lines)

    def write_entry(
        self,
        operator_class: str,
        hardware_signature: str,
        method_name: str,
        run_id: str,
        version_id: str,
        speedup_vs_parent: float,
        operator_shape_signature: str = "",
        hyperparams_pattern: dict | None = None,
        notes: str = "",
    ) -> None:
        """
        写入或更新一条知识。
        使用 EWMA 更新 aggregate_speedup。
        """
        path = self._file_for(operator_class)
        entries = self._load_file(path)

        outcome = Outcome(
            run_id=run_id,
            version_id=version_id,
            speedup_vs_parent=speedup_vs_parent,
            operator_shape_signature=operator_shape_signature,
        )

        # 查找是否已有同方法 + 同硬件 + 同超参模式的条目
        existing = None
        for entry in entries:
            if (
                entry.hardware_signature == hardware_signature
                and entry.method_name == method_name
                and entry.hyperparams_pattern == hyperparams_pattern
            ):
                existing = entry
                break

        if existing:
            existing.observed_outcomes.append(outcome)
            # EWMA 更新
            existing.aggregate_speedup = (
                EWMA_ALPHA * speedup_vs_parent
                + (1 - EWMA_ALPHA) * existing.aggregate_speedup
            )
            # 置信度随观测次数增加
            n = len(existing.observed_outcomes)
            existing.confidence = min(1.0, n * 0.2)
            if notes:
                existing.notes = notes
            existing.last_updated = _now_iso()
        else:
            new_entry = KnowledgeEntry(
                operator_class=operator_class,
                hardware_signature=hardware_signature,
                method_name=method_name,
                hyperparams_pattern=hyperparams_pattern,
                observed_outcomes=[outcome],
                aggregate_speedup=speedup_vs_parent,
                confidence=0.2,
                notes=notes,
            )
            entries.append(new_entry)

        self._save_file(path, entries)

    def write_global_entry(self, **kwargs) -> None:
        """写入全局知识。"""
        kwargs["operator_class"] = "_global"
        self.write_entry(**kwargs)
