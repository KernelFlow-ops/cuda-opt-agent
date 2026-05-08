"""
跨运行知识库 —— 按算子大类组织,提供软提示。
哲学:初始为空,随运行积累,永远以软提示形式注入 Prompt。

[改进]:
  - 支持 negative polarity 条目（失败教训）
  - format_hints_for_prompt 区分正面/负面经验,负面用醒目标记
  - write_entry 支持 polarity 参数
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
        [改进] 负面条目也会返回,排序时负面在前(作为警告)。
        """
        results = []

        # 1) 算子特定知识
        op_entries = self._load_file(self._file_for(operator_class))
        matched = [e for e in op_entries if e.hardware_signature == hardware_signature]

        # [改进] 分开排序：负面条目优先（按回归严重程度）,正面按加速比
        negative = [e for e in matched if e.polarity == "negative"]
        positive = [e for e in matched if e.polarity != "negative"]

        # 负面按回归程度排序 (aggregate_speedup 越小 = 回归越严重,排在前面)
        negative.sort(key=lambda e: e.aggregate_speedup)
        # 正面按加速比 * 置信度排序
        positive.sort(key=lambda e: e.aggregate_speedup * e.confidence, reverse=True)

        results.extend(negative[:top_k])
        results.extend(positive[:top_k])

        # 2) 全局知识
        global_entries = self._load_file(self._file_for("_global"))
        global_matched = [e for e in global_entries if e.hardware_signature == hardware_signature]
        global_matched.sort(key=lambda e: e.aggregate_speedup * e.confidence, reverse=True)
        results.extend(global_matched[:3])

        return results

    def format_hints_for_prompt(self, entries: list[KnowledgeEntry]) -> str:
        """
        格式化知识条目为 Prompt 软提示。

        [改进] 区分正面/负面经验,负面用醒目标记。
        """
        if not entries:
            return "（暂无历史经验）"

        negative = [e for e in entries if e.polarity == "negative"]
        positive = [e for e in entries if e.polarity != "negative"]

        lines = []

        if negative:
            lines.append("### ⚠️ 已知反模式（在相似硬件/算子上曾导致显著回归，**强烈建议避免**）")
            for i, entry in enumerate(negative, 1):
                regression = 1.0 / entry.aggregate_speedup if entry.aggregate_speedup > 0 else float("inf")
                hp_info = ""
                if entry.hyperparams_pattern:
                    hp_info = f", 超参模式: {entry.hyperparams_pattern}"
                lines.append(
                    f"  {i}. ❌ [{entry.operator_class}] 方法: {entry.method_name}{hp_info}\n"
                    f"     回归: {regression:.2f}x 变慢, 置信度: {entry.confidence:.2f}\n"
                    f"     教训: {entry.notes}"
                )

        if positive:
            if negative:
                lines.append("")
            lines.append("### ✓ 历史有效经验")
            for i, entry in enumerate(positive, 1):
                speedup = entry.aggregate_speedup
                hp_info = ""
                if entry.hyperparams_pattern:
                    hp_info = f", 超参模式: {entry.hyperparams_pattern}"
                lines.append(
                    f"  {i}. ✅ [{entry.operator_class}] 方法: {entry.method_name}{hp_info}\n"
                    f"     加速: {speedup:.2f}x, 置信度: {entry.confidence:.2f}\n"
                    f"     备注: {entry.notes}"
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
        polarity: str = "positive",
    ) -> None:
        """
        写入或更新一条知识。
        使用 EWMA 更新 aggregate_speedup。

        [改进] 新增 polarity 参数:
          - "positive": 成功经验 (speedup > 1.0)
          - "negative": 失败教训 (speedup < 1.0, 表示回归)
        """
        path = self._file_for(operator_class)
        entries = self._load_file(path)

        outcome = Outcome(
            run_id=run_id,
            version_id=version_id,
            speedup_vs_parent=speedup_vs_parent,
            operator_shape_signature=operator_shape_signature,
        )

        # 查找是否已有同方法 + 同硬件 + 同超参模式 + 同极性的条目
        existing = None
        for entry in entries:
            if (
                entry.hardware_signature == hardware_signature
                and entry.method_name == method_name
                and entry.hyperparams_pattern == hyperparams_pattern
                and entry.polarity == polarity
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
                polarity=polarity,
            )
            entries.append(new_entry)

        self._save_file(path, entries)
        logger.info(
            "KB %s entry: %s/%s on %s (speedup=%.3f, polarity=%s)",
            "updated" if existing else "wrote new",
            operator_class, method_name, hardware_signature,
            speedup_vs_parent, polarity,
        )

    def write_global_entry(
        self,
        hardware_signature: str,
        method_name: str,
        run_id: str,
        version_id: str,
        speedup_vs_parent: float,
        operator_shape_signature: str = "",
        hyperparams_pattern: dict | None = None,
        notes: str = "",
        polarity: str = "positive",
    ) -> None:
        """写入适用于任意算子的全局知识条目。"""
        self.write_entry(
            operator_class="_global",
            hardware_signature=hardware_signature,
            method_name=method_name,
            run_id=run_id,
            version_id=version_id,
            speedup_vs_parent=speedup_vs_parent,
            operator_shape_signature=operator_shape_signature,
            hyperparams_pattern=hyperparams_pattern,
            notes=notes,
            polarity=polarity,
        )
