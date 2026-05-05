"""
CUDA 算子优化智能体 —— LLM 驱动的自动化 CUDA kernel 优化工具。

用法:
    from cuda_opt_agent.agent import run_optimization
    from cuda_opt_agent.models import OperatorSpec

    spec = OperatorSpec(name="gemm", signature="C = A @ B", ...)
    result = run_optimization(spec)
"""

__version__ = "0.1.0"
