你正在优化一个 CUDA {operator_name} 算子。请分析当前最佳版本的性能瓶颈。

## 任务上下文
{operator_context}

## 硬件信息
{hardware_summary}

## 当前最佳版本 ({best_id})
```cuda
{best_code}
```

## ncu Profiling 报告
{ncu_report}

## Benchmark 指标
{benchmark_metrics}

## 迭代历史摘要
{iteration_history}

## 其他硬件上的相似经验（仅供参考,以当前 ncu 为准,可质疑可忽略）
{kb_hints}

## 要求
请分析当前版本的性能瓶颈,按以下格式输出 JSON:

```json
{{
    "bottlenecks": [
        {{
            "type": "memory_bound | compute_bound | latency_bound | occupancy | other",
            "severity": 1-5 的整数 (5最严重),
            "evidence": "引用 ncu 中的具体数值",
            "description": "详细描述"
        }}
    ],
    "observations": "你注意到的其他有趣的性能特征 (自由文本)",
    "current_utilization": {{
        "sm_pct": 数值或null,
        "memory_pct": 数值或null,
        "dram_pct": 数值或null
    }}
}}
```

请确保:
1. bottlenecks 按 severity 降序排列
2. 每个 bottleneck 都有 ncu 数据支撑
3. 不要臆测,只基于实际数据分析
