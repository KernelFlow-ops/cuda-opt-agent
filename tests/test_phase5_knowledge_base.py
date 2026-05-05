"""
Phase 5 测试 —— 跨运行知识库。
"""

import pytest
from pathlib import Path


class TestKnowledgeBase:
    def test_empty_query(self, tmp_dir):
        from cuda_opt_agent.memory.knowledge import KnowledgeBase
        kb = KnowledgeBase(tmp_dir / "kb")
        results = kb.query("gemm", "sm_80_a100")
        assert results == []

    def test_write_and_query(self, tmp_dir):
        from cuda_opt_agent.memory.knowledge import KnowledgeBase
        kb = KnowledgeBase(tmp_dir / "kb")

        kb.write_entry(
            operator_class="gemm",
            hardware_signature="sm_80_a100",
            method_name="shared_memory_tiling",
            run_id="test_run_1",
            version_id="v3",
            speedup_vs_parent=1.71,
            notes="L2 hit rate 从 41% 升到 73%",
        )

        results = kb.query("gemm", "sm_80_a100")
        assert len(results) == 1
        assert results[0].method_name == "shared_memory_tiling"
        assert results[0].aggregate_speedup == 1.71

    def test_ewma_update(self, tmp_dir):
        from cuda_opt_agent.memory.knowledge import KnowledgeBase, EWMA_ALPHA
        kb = KnowledgeBase(tmp_dir / "kb")

        # 第一次写入
        kb.write_entry(
            operator_class="gemm",
            hardware_signature="sm_80_a100",
            method_name="tiling",
            run_id="run1",
            version_id="v1",
            speedup_vs_parent=2.0,
        )

        # 第二次写入同方法
        kb.write_entry(
            operator_class="gemm",
            hardware_signature="sm_80_a100",
            method_name="tiling",
            run_id="run2",
            version_id="v2",
            speedup_vs_parent=1.5,
        )

        results = kb.query("gemm", "sm_80_a100")
        assert len(results) == 1

        # EWMA: 0.3 * 1.5 + 0.7 * 2.0 = 1.85
        expected = EWMA_ALPHA * 1.5 + (1 - EWMA_ALPHA) * 2.0
        assert abs(results[0].aggregate_speedup - expected) < 0.01

        # 观测数增加
        assert len(results[0].observed_outcomes) == 2

    def test_cross_hardware_isolation(self, tmp_dir):
        from cuda_opt_agent.memory.knowledge import KnowledgeBase
        kb = KnowledgeBase(tmp_dir / "kb")

        kb.write_entry(
            operator_class="gemm",
            hardware_signature="sm_80_a100",
            method_name="tiling",
            run_id="run1",
            version_id="v1",
            speedup_vs_parent=2.0,
        )

        # 不同硬件查不到
        results = kb.query("gemm", "sm_90_h100")
        assert len(results) == 0

    def test_global_knowledge(self, tmp_dir):
        from cuda_opt_agent.memory.knowledge import KnowledgeBase
        kb = KnowledgeBase(tmp_dir / "kb")

        kb.write_global_entry(
            hardware_signature="sm_80_a100",
            method_name="register_tiling",
            run_id="run1",
            version_id="v1",
            speedup_vs_parent=1.3,
            notes="通用优化方法",
        )

        # 查询任何算子都能看到全局知识
        results = kb.query("gemm", "sm_80_a100")
        assert len(results) == 1
        assert results[0].method_name == "register_tiling"

    def test_format_hints(self, tmp_dir):
        from cuda_opt_agent.memory.knowledge import KnowledgeBase
        kb = KnowledgeBase(tmp_dir / "kb")

        kb.write_entry(
            operator_class="gemm",
            hardware_signature="sm_80_a100",
            method_name="tiling",
            run_id="run1",
            version_id="v1",
            speedup_vs_parent=2.0,
            notes="有效",
        )

        results = kb.query("gemm", "sm_80_a100")
        hints = kb.format_hints_for_prompt(results)
        assert "tiling" in hints
        assert "2.00x" in hints

    def test_empty_hints_format(self, tmp_dir):
        from cuda_opt_agent.memory.knowledge import KnowledgeBase
        kb = KnowledgeBase(tmp_dir / "kb")
        hints = kb.format_hints_for_prompt([])
        assert "暂无" in hints

    def test_failure_recording(self, tmp_dir):
        from cuda_opt_agent.memory.knowledge import KnowledgeBase
        kb = KnowledgeBase(tmp_dir / "kb")

        # 记录失败
        kb.write_entry(
            operator_class="gemm",
            hardware_signature="sm_80_a100",
            method_name="bad_method",
            run_id="run1",
            version_id="v1",
            speedup_vs_parent=0.8,  # 减速
            notes="导致 register spilling",
        )

        results = kb.query("gemm", "sm_80_a100")
        assert results[0].aggregate_speedup < 1.0  # 记录了减速

    def test_persistence(self, tmp_dir):
        from cuda_opt_agent.memory.knowledge import KnowledgeBase

        # 写入
        kb1 = KnowledgeBase(tmp_dir / "kb")
        kb1.write_entry(
            operator_class="softmax",
            hardware_signature="sm_80_a100",
            method_name="warp_reduction",
            run_id="run1",
            version_id="v1",
            speedup_vs_parent=1.5,
        )

        # 新实例读取
        kb2 = KnowledgeBase(tmp_dir / "kb")
        results = kb2.query("softmax", "sm_80_a100")
        assert len(results) == 1
        assert results[0].method_name == "warp_reduction"
