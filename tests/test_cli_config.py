from __future__ import annotations

import pytest
from typer.testing import CliRunner

from cuda_opt_agent.cli import app


CONFIG_ENV_KEYS = [
    "CONSOLE_ENCODING",
    "DEFAULT_DTYPE",
    "MAX_ITERATIONS",
    "CONSECUTIVE_REJECT_LIMIT",
    "ACCEPT_EPSILON",
    "DECIDE_RESELECT_MAX_RETRIES",
    "HP_CANDIDATE_COUNT",
    "HP_COMPILE_WORKERS",
    "NCU_LAUNCH_COUNT",
    "NCU_WARMUP_ROUNDS",
    "NCU_PROFILE_ROUNDS",
    "MULTI_SHAPE_AGGREGATOR",
]


class DummyTui:
    def print_welcome(self) -> None:
        pass

    def print_final_report(self, state) -> None:
        pass


class DummyStream:
    def __init__(self, is_tty: bool):
        self._is_tty = is_tty
        self.reconfigure_calls = []

    def isatty(self) -> bool:
        return self._is_tty

    def reconfigure(self, **kwargs) -> None:
        self.reconfigure_calls.append(kwargs)


def clear_config_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in CONFIG_ENV_KEYS:
        monkeypatch.setenv(key, "")
        monkeypatch.delenv(key, raising=False)


def invoke_cli(monkeypatch: pytest.MonkeyPatch, args: list[str]):
    captured = {}

    def fake_run_optimization(operator_spec, config=None, resume_dir=None):
        captured["operator_spec"] = operator_spec
        captured["config"] = config
        captured["resume_dir"] = resume_dir
        return None

    import cuda_opt_agent.agent.graph as graph_module
    import cuda_opt_agent.tui.app as tui_module

    monkeypatch.setattr(graph_module, "run_optimization", fake_run_optimization)
    monkeypatch.setattr(tui_module, "CudaOptApp", DummyTui)

    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "config" in captured
    return captured, result


def invoke_run(monkeypatch: pytest.MonkeyPatch, args: list[str]):
    return invoke_cli(monkeypatch, ["run", "gemm", *args])


def test_help_hides_deprecated_list_and_show_aliases():
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "list-runs" not in result.output
    assert "show-run" not in result.output
    assert "list" in result.output
    assert "show" in result.output


def test_auto_console_encoding_uses_utf8_for_captured_windows_streams(monkeypatch):
    import cuda_opt_agent.cli as cli_module

    stdout = DummyStream(is_tty=False)
    stderr = DummyStream(is_tty=False)
    monkeypatch.setattr(cli_module.os, "name", "nt")
    monkeypatch.setattr(cli_module.sys, "stdout", stdout)
    monkeypatch.setattr(cli_module.sys, "stderr", stderr)
    monkeypatch.delenv("CONSOLE_ENCODING", raising=False)

    cli_module._configure_console_encoding("auto")

    assert stdout.reconfigure_calls == [{"encoding": "utf-8", "errors": "replace"}]
    assert stderr.reconfigure_calls == [{"encoding": "utf-8", "errors": "replace"}]


def test_console_encoding_can_force_gbk(monkeypatch):
    import cuda_opt_agent.cli as cli_module

    stdout = DummyStream(is_tty=False)
    stderr = DummyStream(is_tty=False)
    monkeypatch.setattr(cli_module.sys, "stdout", stdout)
    monkeypatch.setattr(cli_module.sys, "stderr", stderr)

    cli_module._configure_console_encoding("gbk")

    assert stdout.reconfigure_calls == [{"encoding": "gbk", "errors": "replace"}]
    assert stderr.reconfigure_calls == [{"encoding": "gbk", "errors": "replace"}]


def test_console_encoding_default_leaves_streams_unchanged(monkeypatch):
    import cuda_opt_agent.cli as cli_module

    stdout = DummyStream(is_tty=False)
    stderr = DummyStream(is_tty=False)
    monkeypatch.setattr(cli_module.sys, "stdout", stdout)
    monkeypatch.setattr(cli_module.sys, "stderr", stderr)

    cli_module._configure_console_encoding("default")

    assert stdout.reconfigure_calls == []
    assert stderr.reconfigure_calls == []


def test_run_uses_env_defaults_when_cli_options_omitted(tmp_dir, monkeypatch):
    clear_config_env(monkeypatch)
    env_file = tmp_dir / ".env"
    env_file.write_text(
        "\n".join([
            "DEFAULT_DTYPE=bf16",
            "MAX_ITERATIONS=50",
            "CONSECUTIVE_REJECT_LIMIT=9",
            "ACCEPT_EPSILON=0.123",
            "DECIDE_RESELECT_MAX_RETRIES=6",
            "HP_CANDIDATE_COUNT=8",
            "HP_COMPILE_WORKERS=3",
            "NCU_LAUNCH_COUNT=5",
            "NCU_WARMUP_ROUNDS=2",
            "NCU_PROFILE_ROUNDS=3",
        ]),
        encoding="utf-8",
    )

    captured, result = invoke_run(monkeypatch, ["--env", str(env_file)])

    config = captured["config"]
    op_spec = captured["operator_spec"]
    assert config.default_dtype == "bf16"
    assert config.max_iterations == 50
    assert config.consecutive_reject_limit == 9
    assert config.accept_epsilon == pytest.approx(0.123)
    assert config.decide_reselect_max_retries == 6
    assert config.hp_candidate_count == 8
    assert config.hp_compile_workers == 3
    assert config.ncu_launch_count == 5
    assert config.ncu_warmup_rounds == 2
    assert config.ncu_profile_rounds == 3
    assert set(op_spec.dtypes.values()) == {"bf16"}
    assert "Max iterations: 50" in result.output


def test_run_cli_options_override_env_defaults(tmp_dir, monkeypatch):
    clear_config_env(monkeypatch)
    env_file = tmp_dir / ".env"
    env_file.write_text(
        "\n".join([
            "DEFAULT_DTYPE=bf16",
            "MAX_ITERATIONS=50",
            "CONSECUTIVE_REJECT_LIMIT=9",
            "ACCEPT_EPSILON=0.123",
            "DECIDE_RESELECT_MAX_RETRIES=6",
            "HP_CANDIDATE_COUNT=8",
            "HP_COMPILE_WORKERS=3",
        ]),
        encoding="utf-8",
    )

    captured, result = invoke_run(
        monkeypatch,
        [
            "--env",
            str(env_file),
            "--dtype",
            "fp32",
            "--max-iters",
            "7",
            "--consecutive-reject-limit",
            "2",
            "--accept-epsilon",
            "0.01",
            "--decide-reselect-max-retries",
            "2",
            "--hp-candidate-count",
            "4",
            "--hp-compile-workers",
            "2",
        ],
    )

    config = captured["config"]
    op_spec = captured["operator_spec"]
    assert config.default_dtype == "fp32"
    assert config.max_iterations == 7
    assert config.consecutive_reject_limit == 2
    assert config.accept_epsilon == pytest.approx(0.01)
    assert config.decide_reselect_max_retries == 2
    assert config.hp_candidate_count == 4
    assert config.hp_compile_workers == 2
    assert set(op_spec.dtypes.values()) == {"fp32"}
    assert "Max iterations: 7" in result.output


def test_run_shapes_parses_multi_gemm_profiles(tmp_dir, monkeypatch):
    clear_config_env(monkeypatch)

    captured, _ = invoke_cli(
        monkeypatch,
        [
            "run",
            "gemm",
            "--shapes",
            "1024^3;2048^3",
            "--multi-shape-aggregator",
            "worst",
            "--env",
            str(tmp_dir / "missing.env"),
        ],
    )

    config = captured["config"]
    op_spec = captured["operator_spec"]
    assert config.multi_shape_aggregator == "worst"
    assert op_spec.shape_profiles == [
        {"M": 1024, "N": 1024, "K": 1024},
        {"M": 2048, "N": 2048, "K": 2048},
    ]
    assert op_spec.shapes == {"A": [1024, 1024], "B": [1024, 1024], "C": [1024, 1024]}


def test_new_shape_profile_uses_operator_defaults(tmp_dir, monkeypatch):
    clear_config_env(monkeypatch)

    captured, _ = invoke_cli(
        monkeypatch,
        [
            "new",
            "softmax",
            "--task",
            "stable softmax",
            "--shape-profile",
            "sweep",
            "--env",
            str(tmp_dir / "missing.env"),
        ],
    )

    op_spec = captured["operator_spec"]
    assert op_spec.shape_profiles == [{"B": 1024, "N": 1024}, {"B": 4096, "N": 4096}]


def test_new_spec_mode_builds_operator_spec(tmp_dir, monkeypatch):
    clear_config_env(monkeypatch)
    spec_file = tmp_dir / "softmax.yaml"
    spec_file.write_text(
        """
name: softmax
signature: "y[B,N] = softmax(x[B,N], dim=-1)"
dtypes: {x: fp16, y: fp16}
task_description: "Stable softmax along the last dimension."
constraints:
  - "Use max-trick."
shape_profiles:
  - {x: [1024, 1024], y: [1024, 1024]}
""".strip(),
        encoding="utf-8",
    )

    captured, result = invoke_cli(
        monkeypatch,
        ["new", "--spec", str(spec_file), "--env", str(tmp_dir / "missing.env")],
    )

    op_spec = captured["operator_spec"]
    assert op_spec.name == "softmax"
    assert op_spec.task_description == "Stable softmax along the last dimension."
    assert op_spec.shapes == {"x": [1024, 1024], "y": [1024, 1024]}
    assert op_spec.shape_profiles == [{"x": [1024, 1024], "y": [1024, 1024]}]
    assert "Shape profiles: 1" in result.output


def test_new_task_mode_builds_free_text_spec(tmp_dir, monkeypatch):
    clear_config_env(monkeypatch)

    captured, _ = invoke_cli(
        monkeypatch,
        [
            "new",
            "softmax",
            "--task",
            "写一个 fp16 softmax,沿最后一维归一化",
            "--shape",
            "1024,1024",
            "--dtype",
            "bf16",
            "--env",
            str(tmp_dir / "missing.env"),
        ],
    )

    config = captured["config"]
    op_spec = captured["operator_spec"]
    assert config.default_dtype == "bf16"
    assert op_spec.name == "softmax"
    assert op_spec.task_description == "写一个 fp16 softmax,沿最后一维归一化"
    assert op_spec.shapes == {"input": [1024, 1024]}
    assert set(op_spec.dtypes.values()) == {"bf16"}


def test_tune_mode_uses_cuda_file_as_seed(tmp_dir, monkeypatch):
    clear_config_env(monkeypatch)
    seed = tmp_dir / "fused_attention.cu"
    seed.write_text("__global__ void fused_attention() {}\n", encoding="utf-8")

    captured, result = invoke_cli(
        monkeypatch,
        [
            "tune",
            str(seed),
            "--operator",
            "fused_attention",
            "--task",
            "Keep mask semantics intact.",
            "--env",
            str(tmp_dir / "missing.env"),
        ],
    )

    op_spec = captured["operator_spec"]
    assert op_spec.name == "fused_attention"
    assert op_spec.seed_code_path == str(seed.resolve())
    assert op_spec.task_description == "Keep mask semantics intact."
    assert "Seed code:" in result.output


def test_new_rejects_multiple_input_modes(tmp_dir, monkeypatch):
    clear_config_env(monkeypatch)
    spec_file = tmp_dir / "softmax.yaml"
    spec_file.write_text('name: softmax\nsignature: "y = softmax(x)"\n', encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "new",
            "softmax",
            "--spec",
            str(spec_file),
            "--task",
            "free text",
            "--env",
            str(tmp_dir / "missing.env"),
        ],
    )

    assert result.exit_code == 1
    assert "Use only one input mode" in result.output


def test_new_auto_requires_input_mode(tmp_dir, monkeypatch):
    clear_config_env(monkeypatch)

    result = CliRunner().invoke(
        app,
        ["new", "softmax", "--auto", "--env", str(tmp_dir / "missing.env")],
    )

    assert result.exit_code == 1
    assert "Specify --spec, --task, or --from-cu" in result.output
