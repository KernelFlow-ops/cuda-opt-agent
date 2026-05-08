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

请按以下**严格顺序**输出 JSON。**第一步是判断可优化性，再进入瓶颈分析**：

```json
{{
    "kernel_regime": {{
        "absolute_latency_ms": 当前 best 的 latency 数值,
        "regime": "tiny (<0.01ms) | small (0.01-0.1ms) | medium (0.1-1ms) | large (>1ms)",
        "near_launch_floor": true或false,
        "rationale": "解释为什么属于这个 regime。关键原则：当 latency < 0.01ms 时, kernel 实际工作时间通常已被 launch overhead (~4-8us) 主导, ncu 的百分比利用率指标会严重失真——大量 headroom 是测量伪信号而非真实优化空间。当 latency < 0.005ms 时几乎可以断定已触底。"
    }},
    "metric_reliability": {{
        "trustworthy": true或false,
        "warnings": ["列出哪些 ncu 指标在当前 latency 规模下可能失真，例如 'SM Throughput 3.2% 在 6us kernel 上不可信, 因为 launch/teardown 开销占比 >50%'"]
    }},
    "bottlenecks": [
        {{
            "type": "memory_bound | compute_bound | latency_bound | launch_overhead | already_optimal | other",
            "severity": 1-5 的整数 (5最严重),
            "evidence": "引用 ncu 中的具体数值，同时标注 absolute time (ns/us) 而非仅仅百分比",
            "description": "详细描述。若 regime=tiny, 主要瓶颈大概率是 launch_overhead 或 already_optimal, 而非 latency_bound"
        }}
    ],
    "optimization_headroom": {{
        "estimated_max_speedup": "保守估计的可达加速比 (例: 1.0x-1.05x / 1.2x / 1.5x / 2x+)",
        "rationale": "结合 regime、ncu、iteration_history 解释。若 regime=tiny 且连续失败, 应写 1.0x-1.05x",
        "recommend_stop": true或false
    }},
    "observations": "你注意到的其他有趣的性能特征 (自由文本)",
    "current_utilization": {{
        "sm_pct": 数值或null,
        "memory_pct": 数值或null,
        "dram_pct": 数值或null
    }}
}}
```

## 关键判断准则

1. **tiny kernel 判定规则**: 若 `absolute_latency_ms < 0.01` 且 ncu 显示低 SM/DRAM 利用率，**默认结论是 launch_overhead-bound 或 already_optimal，而非 latency_bound**。
   - 不要将 "occupancy 5.8%, headroom 94%" 解读为"有 94% 优化空间"——这在 tiny kernel 上是错误推理
   - 正确解读: launch 开销 (~4-8us) 占了 kernel 的 50%-80%，GPU 硬件资源确实在被使用但时间太短无法充分摊薄
2. **连续回归检测**: 若 `iteration_history` 中最近 2-3 次尝试 latency 都 > best 的 1.5x，强制 `recommend_stop=true` 或 `bottlenecks[0].type="already_optimal"`
3. **绝对时间优先于百分比**: 分析 evidence 时优先引用绝对时间(ns)和字节数,百分比利用率只在 regime >= medium 时作为主要论据
4. 不要臆测,但也不要被百分比数字误导——基于 regime + ncu + absolute metrics 综合判断

请确保:
1. bottlenecks 按 severity 降序排列
2. 每个 bottleneck 都有数据支撑
3. kernel_regime 必须是输出的第一个字段，后续分析必须在 regime 判定的框架下进行
