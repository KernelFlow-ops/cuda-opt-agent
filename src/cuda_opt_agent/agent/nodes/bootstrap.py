"""Bootstrap 节点 —— 生成 ref.py + baseline CUDA 代码 + benchmark_runner.py。"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any

from ...codegen.ref_generator import (
    build_extern_c_params,
    ensure_executable_harness,
    generate_benchmark_runner,
    generate_ref_py,
    wrap_with_extern_c,
)
from ...codegen.normalizer import extract_cuda_code
from ...tools.web_search import format_search_results_for_prompt, search_for_baseline_reference
from ..temperatures import TEMP_BOOTSTRAP

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "bootstrap.md"


def _load_prompt() -> str:
    if PROMPT_PATH.exists():
        return PROMPT_PATH.read_text(encoding="utf-8")
    return "Generate a baseline CUDA kernel."


async def bootstrap_node(state: dict[str, Any], *, llm_client: Any,
                          state_manager: Any = None, config: Any = None) -> dict[str, Any]:
    """生成 ref.py, benchmark_runner.py, 和 baseline CUDA 代码。"""
    op_spec = state.get("operator_spec")
    if op_spec is None:
        raise ValueError("operator_spec missing")

    op = op_spec.name if hasattr(op_spec, "name") else op_spec.get("name", "kernel")
    sig = getattr(op_spec, "signature", "") or ""
    dtypes = getattr(op_spec, "dtypes", {}) or {}
    shapes = getattr(op_spec, "shapes", {}) or {}
    profiles = getattr(op_spec, "shape_profiles", []) or []
    constraints = getattr(op_spec, "constraints", []) or []
    task_desc = getattr(op_spec, "task_description", "") or ""
    seed_path = getattr(op_spec, "seed_code_path", None)
    dtype = list(dtypes.values())[0] if dtypes else "fp16"

    run_state = state.get("run_state")
    run_dir = ""
    if run_state:
        run_dir = getattr(run_state, "run_dir", "") or ""
    if not run_dir and state_manager and state_manager.run_dir:
        run_dir = str(state_manager.run_dir)
    if not run_dir:
        run_dir = f"runs/{op}_bootstrap"
    run_dir_path = Path(run_dir)
    run_dir_path.mkdir(parents=True, exist_ok=True)

    # Step 1: ref.py
    logger.info("Generating ref.py for %s", op)
    ref_path = generate_ref_py(op, shapes=shapes, shape_profiles=profiles,
                                dtypes=dtypes, default_dtype=dtype, output_dir=run_dir_path)

    # Step 2: benchmark_runner.py
    runner_path = generate_benchmark_runner(op, output_dir=run_dir_path)

    # Step 3: 搜索外部参考
    ext_ref = ""
    web_search_enabled = getattr(config, "enable_web_search_baseline", True) if config else True
    if web_search_enabled:
        try:
            results = await search_for_baseline_reference(
                op,
                dtype=dtype,
                task_description=task_desc,
                shapes=shapes,
                shape_profiles=profiles,
                hardware_context=" ".join(str(x) for x in (
                    getattr(state.get("hardware_spec"), "gpu_name", ""),
                    getattr(state.get("hardware_spec"), "compute_capability", ""),
                    getattr(state.get("hardware_spec"), "cuda_version", ""),
                ) if x),
                max_calls=getattr(config, "bootstrap_web_search_max_calls", 20),
                max_results=getattr(config, "bootstrap_web_search_max_results", 12),
                per_query_results=getattr(config, "bootstrap_web_search_per_query_results", 3),
            )
            if results:
                ext_ref = format_search_results_for_prompt(results)
                logger.info("Injected %d baseline web search results", len(results))
            else:
                logger.info("Baseline web search returned no results")
        except Exception as e:
            logger.warning("External ref search failed: %s", e)
    else:
        logger.info("Baseline web search disabled by config")

    # Step 4: LLM 生成 baseline
    seed_section = ""
    bootstrap_mode_instruction = "当前没有已有实现,请从零生成一个正确性优先的 v0 baseline。"
    if seed_path:
        sp = Path(seed_path)
        if sp.exists():
            seed_code = sp.read_text(encoding="utf-8")
            seed_section = f'## Seed Code\n\n```cuda\n{seed_code}\n```\n\nEnsure extern "C" entry.'
            bootstrap_mode_instruction = (
                "以下代码已经实现该算子,请将其作为 v0 baseline。"
                "如果缺少 extern \"C\" 入口,请补齐;不要修改算法逻辑,"
                "不要添加 main、正确性检查或 benchmark 框架,这些由 ref.py 统一负责。"
            )

    ec_params = build_extern_c_params(op, dtypes)
    prompt_kwargs = dict(
        operator_name=op, signature=sig,
        dtypes=json.dumps(dtypes, ensure_ascii=False),
        shapes=json.dumps(shapes, ensure_ascii=False),
        shape_profiles=json.dumps(profiles, ensure_ascii=False),
        task_description=task_desc or f"Implement {op} CUDA kernel",
        seed_code_section=seed_section, external_reference=ext_ref,
        extern_c_params=ec_params,
        constraints="\n".join(f"- {c}" for c in constraints) if constraints else "None",
        bootstrap_mode_instruction=bootstrap_mode_instruction,
        gpu_name=getattr(state.get("hardware_spec"), "gpu_name", ""),
        compute_capability=getattr(state.get("hardware_spec"), "compute_capability", ""),
        sm_count=getattr(state.get("hardware_spec"), "sm_count", ""),
        shared_mem_per_block_kb=getattr(state.get("hardware_spec"), "shared_mem_per_block_kb", ""),
        l2_cache_mb=getattr(state.get("hardware_spec"), "l2_cache_mb", ""),
        has_tensor_cores=getattr(state.get("hardware_spec"), "has_tensor_cores", ""),
        cuda_version=getattr(state.get("hardware_spec"), "cuda_version", ""),
        kb_hints_section="",
    )
    if hasattr(llm_client, "format_prompt"):
        prompt = llm_client.format_prompt("bootstrap.md", **prompt_kwargs)
    else:
        prompt = _load_prompt().format(**prompt_kwargs)

    logger.info("Calling LLM for baseline CUDA code")
    response = await llm_client.ainvoke(prompt, temperature=TEMP_BOOTSTRAP, node_name="bootstrap")
    cuda_code = extract_cuda_code(response)
    cuda_code = wrap_with_extern_c(cuda_code, op, extern_c_params=ec_params, version="v0")
    cuda_code = ensure_executable_harness(cuda_code, op)

    # Save
    v0_dir = run_dir_path / "v0"
    v0_dir.mkdir(parents=True, exist_ok=True)
    code_path = v0_dir / "code.cu"
    code_path.write_text(cuda_code, encoding="utf-8")
    logger.info("Baseline saved to %s", code_path)

    return {
        **state,
        "bootstrap_code": cuda_code,
        "bootstrap_code_path": str(code_path),
        "ref_py_path": str(ref_path),
        "benchmark_runner_path": str(runner_path),
        "current_code": cuda_code,
        "new_code": cuda_code,
        "new_version_id": "v0",
        "current_best_code": cuda_code,
        "current_best_id": "v0",
        "trial_code": cuda_code,
        "trial_code_path": str(code_path),
    }
