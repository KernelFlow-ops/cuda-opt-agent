"""
Phase 2 测试 —— 持久化层 + 状态管理 + CLI。
"""

import json
import pytest
from pathlib import Path


class TestPersistenceManager:
    def test_create_run_dir(self, tmp_dir):
        from cuda_opt_agent.memory.persistence import PersistenceManager
        pm = PersistenceManager(str(tmp_dir / "runs"))
        run_dir = pm.create_run_dir("gemm")
        assert run_dir.exists()
        assert "gemm_run_" in run_dir.name

    def test_save_load_state(self, tmp_dir, sample_run_state):
        from cuda_opt_agent.memory.persistence import PersistenceManager
        pm = PersistenceManager(str(tmp_dir / "runs"))
        run_dir = pm.create_run_dir("gemm")

        pm.save_state(sample_run_state, run_dir)
        assert (run_dir / "state.json").exists()

        loaded = pm.load_state(run_dir)
        assert loaded.run_id == sample_run_state.run_id
        assert loaded.current_best_id == "v0"
        assert len(loaded.iterations) == 1

    def test_save_load_state_roundtrip(self, tmp_dir, sample_run_state):
        from cuda_opt_agent.memory.persistence import PersistenceManager
        pm = PersistenceManager(str(tmp_dir / "runs"))
        run_dir = pm.create_run_dir("gemm")

        # 保存
        pm.save_state(sample_run_state, run_dir)

        # 加载
        loaded = pm.load_state(run_dir)

        # 验证关键字段完整
        assert loaded.operator_spec.name == "gemm"
        assert loaded.hardware_spec.compute_capability == "sm_80"
        assert loaded.config.max_iterations == 5

    def test_append_and_load_history(self, tmp_dir, sample_iteration_record):
        from cuda_opt_agent.memory.persistence import PersistenceManager
        pm = PersistenceManager(str(tmp_dir / "runs"))
        run_dir = pm.create_run_dir("test")

        pm.append_history(sample_iteration_record, run_dir)
        pm.append_history(sample_iteration_record, run_dir)

        records = pm.load_history(run_dir)
        assert len(records) == 2
        assert records[0].version_id == "v0"

    def test_text_outputs_are_utf8(self, tmp_dir, sample_iteration_record):
        from cuda_opt_agent.memory.persistence import PersistenceManager
        pm = PersistenceManager(str(tmp_dir / "runs"))
        run_dir = pm.create_run_dir("test")
        iter_dir = pm.create_iteration_dir(run_dir, "v1")

        sample_iteration_record.method_name = "切换到 Tensor Core"
        pm.append_history(sample_iteration_record, run_dir)
        code_path = pm.save_code("// 中文注释\n__global__ void k() {}", iter_dir)
        reasoning_path = pm.save_reasoning("中文推理", iter_dir)
        pm.save_reasoning_log("### Method: 中文方法", run_dir)

        assert "切换到 Tensor Core" in (run_dir / "history.jsonl").read_text(encoding="utf-8")
        assert "中文注释" in code_path.read_text(encoding="utf-8")
        assert reasoning_path.read_text(encoding="utf-8") == "中文推理"
        assert "中文方法" in (run_dir / "reasoning_log.md").read_text(encoding="utf-8")

    def test_create_iteration_dir(self, tmp_dir):
        from cuda_opt_agent.memory.persistence import PersistenceManager
        pm = PersistenceManager(str(tmp_dir / "runs"))
        run_dir = pm.create_run_dir("test")
        iter_dir = pm.create_iteration_dir(run_dir, "v0")
        assert iter_dir.exists()
        assert "iterv0" in iter_dir.name

    def test_save_code(self, tmp_dir):
        from cuda_opt_agent.memory.persistence import PersistenceManager
        pm = PersistenceManager(str(tmp_dir / "runs"))
        run_dir = pm.create_run_dir("test")
        iter_dir = pm.create_iteration_dir(run_dir, "v0")

        code_path = pm.save_code("__global__ void k() {}", iter_dir)
        assert code_path.exists()
        assert code_path.read_text(encoding="utf-8") == "__global__ void k() {}"

    def test_update_best_requires_existing_iter_dir(self, tmp_dir):
        from cuda_opt_agent.memory.persistence import PersistenceManager
        pm = PersistenceManager(str(tmp_dir / "runs"))
        run_dir = pm.create_run_dir("test")

        missing_iter_dir = run_dir / "iterv_missing"
        with pytest.raises(FileNotFoundError):
            pm.update_best_symlink(run_dir, missing_iter_dir)

        assert not (run_dir / "best.txt").exists()

    def test_find_latest_unfinished(self, tmp_dir, sample_run_state):
        from cuda_opt_agent.memory.persistence import PersistenceManager
        pm = PersistenceManager(str(tmp_dir / "runs"))

        # 无运行时
        assert pm.find_latest_unfinished_run("gemm") is None

        # 创建一个运行
        run_dir = pm.create_run_dir("gemm")
        pm.save_state(sample_run_state, run_dir)

        found = pm.find_latest_unfinished_run("gemm")
        assert found is not None
        assert found == run_dir

    def test_recovery_from_history(self, tmp_dir, sample_iteration_record):
        from cuda_opt_agent.memory.persistence import PersistenceManager
        pm = PersistenceManager(str(tmp_dir / "runs"))
        run_dir = pm.create_run_dir("test")

        # 只有 history,没有 state.json
        pm.append_history(sample_iteration_record, run_dir)

        state = pm.try_recover_state(run_dir)
        assert state is not None
        assert len(state.iterations) == 1
        assert state.current_best_id == "v0"


class TestRunStateManager:
    def test_init_new_run(self, sample_agent_config, sample_operator_spec, sample_hardware_spec):
        from cuda_opt_agent.memory.run_state import RunStateManager
        sm = RunStateManager(sample_agent_config)
        state = sm.init_new_run(sample_operator_spec, sample_hardware_spec)

        assert state.run_id != ""
        assert state.operator_spec.name == "gemm"
        assert sm.run_dir is not None
        assert sm.run_dir.exists()

    def test_add_iteration(self, sample_agent_config, sample_operator_spec,
                           sample_hardware_spec, sample_iteration_record):
        from cuda_opt_agent.memory.run_state import RunStateManager
        sm = RunStateManager(sample_agent_config)
        sm.init_new_run(sample_operator_spec, sample_hardware_spec)

        sm.add_iteration(sample_iteration_record)
        assert len(sm.state.iterations) == 1

    def test_blacklist_management(self, sample_agent_config, sample_operator_spec, sample_hardware_spec):
        from cuda_opt_agent.memory.run_state import RunStateManager
        sm = RunStateManager(sample_agent_config)
        sm.init_new_run(sample_operator_spec, sample_hardware_spec)

        sm.add_to_blacklist("tiling", "no improvement")
        assert len(sm.state.blacklist) == 1
        assert sm.state.is_method_blacklisted("tiling")

    def test_should_stop_max_iters(self, sample_agent_config, sample_operator_spec,
                                    sample_hardware_spec, sample_iteration_record):
        from cuda_opt_agent.memory.run_state import RunStateManager
        sample_agent_config.max_iterations = 2
        sm = RunStateManager(sample_agent_config)
        sm.init_new_run(sample_operator_spec, sample_hardware_spec)

        sm.add_iteration(sample_iteration_record)
        stop, _ = sm.should_stop()
        assert not stop

        sm.add_iteration(sample_iteration_record)
        stop, reason = sm.should_stop()
        assert stop
        assert "maximum iterations" in reason


class TestConfig:
    def test_load_default(self, tmp_dir):
        from cuda_opt_agent.config import load_config
        config = load_config(tmp_dir / "nonexistent.env")
        assert config.max_iterations == 30
        assert config.llm_provider == "anthropic"

    def test_load_from_env_file(self, tmp_dir):
        env_file = tmp_dir / ".env"
        env_file.write_text("MAX_ITERATIONS=50\nLLM_PROVIDER=openai\n", encoding="utf-8")

        from cuda_opt_agent.config import load_config
        config = load_config(str(env_file))
        assert config.max_iterations == 50
