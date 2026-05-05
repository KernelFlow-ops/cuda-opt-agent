"""
Phase 4 测试 —— 超参搜索 + 黑名单逻辑。
"""

import pytest
from cuda_opt_agent.models.data import (
    BlacklistEntry,
    BenchmarkResult,
    HyperparamCandidate,
    IterationRecord,
    MethodDecision,
    RunState,
)
from cuda_opt_agent.models.enums import make_blacklist_key, normalize_method_name


class TestBlacklist:
    def test_basic_blacklist(self, sample_run_state):
        state = sample_run_state
        state.blacklist.append(BlacklistEntry(
            method_name_normalized="tiling",
            reason="no speedup",
        ))
        assert state.is_method_blacklisted("tiling")
        assert state.is_method_blacklisted("Tiling")
        assert not state.is_method_blacklisted("vectorization")

    def test_blacklist_with_hp_constraint(self, sample_run_state):
        state = sample_run_state
        state.blacklist.append(BlacklistEntry(
            method_name_normalized="tiling",
            hyperparam_constraint={"tile_m": "<=32"},
            reason="small tiles ineffective",
        ))
        # 无超参约束的 tiling 不在黑名单
        assert not state.is_method_blacklisted("tiling")
        # 带相同约束的 tiling 在黑名单
        assert state.is_method_blacklisted("tiling", {"tile_m": "<=32"})
        # 不同约束不在黑名单
        assert not state.is_method_blacklisted("tiling", {"tile_m": ">=64"})

    def test_blacklist_key_generation(self):
        key1 = make_blacklist_key("Shared Memory Tiling")
        key2 = make_blacklist_key("shared_memory_tiling")
        assert key1 == key2

        key3 = make_blacklist_key("tiling", {"tile_m": 128})
        assert "::" in key3

    def test_multiple_blacklist_entries(self, sample_run_state):
        state = sample_run_state
        methods = ["tiling", "vectorization", "loop_unrolling", "double_buffer", "warp_shuffle"]
        for m in methods:
            state.blacklist.append(BlacklistEntry(
                method_name_normalized=m,
                reason="failed",
            ))
        assert len(state.blacklist) == 5
        for m in methods:
            assert state.is_method_blacklisted(m)


class TestHyperparamCandidate:
    def test_create(self):
        cand = HyperparamCandidate(
            index=0,
            hyperparams={"tile_m": 128, "tile_n": 128, "tile_k": 32},
            rationale="经典平衡配置",
        )
        assert cand.hyperparams["tile_m"] == 128

    def test_diverse_candidates(self):
        """验证候选应该多样化。"""
        candidates = [
            HyperparamCandidate(index=i, hyperparams={"tile": 2**i * 16}, rationale=f"size {2**i * 16}")
            for i in range(5)
        ]
        tile_sizes = [c.hyperparams["tile"] for c in candidates]
        assert len(set(tile_sizes)) == 5  # 所有值不同
        assert max(tile_sizes) / min(tile_sizes) >= 8  # 跨度足够大


class TestMethodDecision:
    def test_give_up(self):
        d = MethodDecision(
            method_name="none",
            give_up=True,
            rationale="已尝试所有主要方向",
        )
        assert d.give_up is True

    def test_with_hyperparams(self):
        d = MethodDecision(
            method_name="tiling",
            has_hyperparams=True,
            hyperparams_schema={
                "tile_m": {"type": "int", "range": [16, 256]},
                "tile_n": {"type": "int", "range": [16, 256]},
            },
            rationale="memory bound, need tiling",
            confidence=0.85,
        )
        assert d.has_hyperparams
        assert "tile_m" in d.hyperparams_schema


class TestAcceptRejectCriteria:
    def test_accept_when_faster(self):
        epsilon = 0.005
        best_lat = 1.0
        trial_lat = 0.9  # 10% faster

        threshold = best_lat * (1 - epsilon)
        accepted = trial_lat < threshold
        assert accepted

    def test_reject_when_noise(self):
        epsilon = 0.005
        best_lat = 1.0
        trial_lat = 0.998  # only 0.2% faster, within noise

        threshold = best_lat * (1 - epsilon)
        accepted = trial_lat < threshold
        assert not accepted

    def test_reject_when_slower(self):
        epsilon = 0.005
        best_lat = 1.0
        trial_lat = 1.05  # 5% slower

        threshold = best_lat * (1 - epsilon)
        accepted = trial_lat < threshold
        assert not accepted

    def test_consecutive_rejects(self, sample_run_state):
        """连续回退计数测试。"""
        state = sample_run_state
        bm = BenchmarkResult(latency_ms_median=2.0, latency_ms_p95=2.5)

        for i in range(5):
            state.iterations.append(IterationRecord(
                version_id=f"v{i+1}",
                parent_id="v0",
                method_name=f"method_{i}",
                benchmark=bm,
                accepted=False,
            ))

        assert state.consecutive_rejects() == 5
