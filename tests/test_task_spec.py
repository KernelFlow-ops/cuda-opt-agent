from __future__ import annotations

from pathlib import Path


def test_load_operator_spec_yaml_resolves_seed_path(tmp_dir):
    from cuda_opt_agent.task_spec import load_operator_spec

    seed = tmp_dir / "seed.cu"
    seed.write_text("__global__ void kernel() {}\n", encoding="utf-8")
    spec_file = tmp_dir / "softmax.yaml"
    spec_file.write_text(
        """
name: softmax
signature: "y[B,N] = softmax(x[B,N], dim=-1)"
dtypes: {x: fp16, y: fp16}
task_description: |
  Implement a numerically stable softmax.
constraints:
  - "Avoid overflow."
seed_code_path: seed.cu
shape_profiles:
  - {x: [1024, 1024], y: [1024, 1024]}
  - {x: [2048, 2048], y: [2048, 2048]}
""".strip(),
        encoding="utf-8",
    )

    spec = load_operator_spec(spec_file)

    assert spec.name == "softmax"
    assert spec.shapes == {"x": [1024, 1024], "y": [1024, 1024]}
    assert len(spec.shape_profiles) == 2
    assert spec.seed_code_path == str(seed.resolve())


def test_load_operator_spec_toml(tmp_dir):
    from cuda_opt_agent.task_spec import load_operator_spec

    spec_file = tmp_dir / "layernorm.toml"
    spec_file.write_text(
        """
name = "layernorm"
signature = "y[B,N] = layernorm(x[B,N])"
dtypes = { x = "fp32", y = "fp32" }
constraints = ["Use a stable variance computation."]

[[shape_profiles]]
x = [512, 1024]
y = [512, 1024]
""".strip(),
        encoding="utf-8",
    )

    spec = load_operator_spec(spec_file)

    assert spec.name == "layernorm"
    assert spec.dtypes == {"x": "fp32", "y": "fp32"}
    assert spec.shapes == {"x": [512, 1024], "y": [512, 1024]}


def test_resolve_existing_cuda_path_rejects_non_cuda(tmp_dir):
    from cuda_opt_agent.task_spec import resolve_existing_cuda_path

    path = tmp_dir / "kernel.txt"
    path.write_text("not cuda", encoding="utf-8")

    try:
        resolve_existing_cuda_path(path)
    except ValueError as e:
        assert ".cu" in str(e)
    else:
        raise AssertionError("Expected ValueError for non-.cu seed file")
