# Analyze: 分析 NCU 剖析数据

你是 CUDA 性能分析专家。请仔细分析以下 NCU (Nsight Compute) 剖析数据，判断当前 kernel 的瓶颈类型。

## 算子信息

- 名称: {operator_name}
- 当前 best: {best_id}

{operator_context}

## 硬件信息

{hardware_summary}
{hardware_spec}

## 当前最优版本代码

```cuda
{best_code}
{current_best_code}
```

## Benchmark 指标

{benchmark_metrics}

## NCU 剖析数据

{ncu_report}
{ncu_profile}

## 优化历史

{iteration_history}
{optimization_history}

## 历史经验

{kb_hints}

---

## 分析要求

### 1. 瓶颈识别

根据 NCU 指标判断主要瓶颈类型:

**Memory-bound 指标:**
- `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` > 60%
- `sm__warps_active.avg.pct_of_peak_sustained_active` < 50%
- `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum / l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum` > 4 (不合并)
- `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` 较高 (bank conflict)

**Compute-bound 指标:**
- `sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active` > 60%
- `smsp__inst_executed.avg.pct_of_peak_sustained_active` > 80%
- Memory throughput 低

**Latency-bound 指标:**
- 低占用率: `sm__warps_active.avg.pct_of_peak_sustained_active` < 30%
- 分歧: `smsp__thread_inst_executed_pred_on.avg.pct_of_peak_sustained_active` < 70%
- Stall 比例高

### 2. 子空间推荐

基于瓶颈类型，推荐最相关的优化子空间（按优先级排序，最多 5 个）。

### 3. 量化指标

列出关键的性能指标数值和改善目标。

## 输出格式（严格 JSON）

```json
{{
  "bottleneck_type": "memory_bound" | "compute_bound" | "latency_bound" | "balanced",
  "bottleneck_details": "详细瓶颈分析",
  "key_metrics": {{
    "metric_name": {{"value": "...", "assessment": "good/bad/neutral", "note": "..."}}
  }},
  "recommended_subspaces": [
    {{"name": "subspace-name", "priority": 1, "reason": "..."}}
  ],
  "code_observations": "从代码中观察到的潜在问题",
  "memory_analysis": {{
    "coalescing": "good/bad/unknown",
    "bank_conflicts": "none/few/many/unknown",
    "shared_mem_usage": "none/low/high",
    "vectorized": true
  }},
  "compute_analysis": {{
    "tensor_core_usage": false,
    "arithmetic_intensity": "low/medium/high",
    "divergence": "none/some/severe/unknown"
  }}
}}
```
