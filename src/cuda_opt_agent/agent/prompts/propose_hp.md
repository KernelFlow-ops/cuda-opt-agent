你正在优化一个 CUDA {operator_name} 算子。已决定使用方法: **{method_name}**。

## 任务上下文
{operator_context}

## 方法描述
{method_rationale}

## 超参空间
{hyperparams_schema}

## 同方法历史尝试与已知超参组合
{known_hp_trials}

## 当前最佳版本 ({best_id}) 的关键 ncu 指标
{ncu_key_metrics}

## 硬件信息
{hardware_summary}

## 要求
请提出 **{hp_count} 组多样化的超参候选**,每组必须附简短理由。

**多样化要求**:
- 覆盖不同量级（例如 tile 16/32/64/128 各取代表值,而非 32/33/34/35/36）
- 至少有一组保守选择,一组激进选择
- 考虑硬件限制（共享内存、寄存器文件等）
- 避开历史中已失败的同方法同超参组合；如必须复试,必须在 rationale 中说明为什么当前上下文不同

## 输出格式 (严格 JSON 数组)
```json
[
    {{
        "index": 0,
        "hyperparams": {{"tile_m": 128, "tile_n": 128, "tile_k": 32}},
        "rationale": "经典平衡配置,适中的 tile 大小兼顾 reuse 和 occupancy"
    }},
    {{
        "index": 1,
        "hyperparams": {{"tile_m": 64, "tile_n": 64, "tile_k": 64}},
        "rationale": "增大 K 维以提升数据 reuse,适合 memory-bound 场景"
    }}
]
```


