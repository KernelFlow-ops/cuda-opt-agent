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

## 当前 Kernel Regime
{kernel_regime_info}

## 要求
请提出 **{hp_count} 组多样化的超参候选**,每组必须附简短理由。

**多样化要求**:
- 覆盖不同量级（例如 tile 16/32/64/128 各取代表值,而非 32/33/34/35/36）
- 至少有一组保守选择,一组激进选择
- 考虑硬件限制（共享内存、寄存器文件等）
- 避开历史中已失败的同方法同超参组合；如必须复试,必须在 rationale 中说明为什么当前上下文不同

**安全约束（强制执行）**:
- 当 baseline latency_median < 0.01 ms 时，以下超参组合被**禁止**：
  - `blocks_per_channel > 1` 或任何导致跨 CTA partial reduction 的设置
  - `threads_per_block < 64` 或显著降低单 CTA 线程数的设置
  - 任何引入 global atomicAdd、额外临时 buffer 或二次归约 kernel 的设置
  - 任何导致 kernel launch 数量增加的设置
- 每个候选必须声明 `predicted_regression_risk`（low/medium/high），并简述理由
- high risk 候选最多允许 1 个，且必须在 rationale 中充分论证

## 输出格式 (严格 JSON 数组)
```json
[
    {{
        "index": 0,
        "hyperparams": {{"tile_m": 128, "tile_n": 128, "tile_k": 32}},
        "rationale": "经典平衡配置,适中的 tile 大小兼顾 reuse 和 occupancy",
        "predicted_regression_risk": "low",
        "risk_rationale": "不增加 kernel 数量或全局同步，仅调整 tile 粒度"
    }},
    {{
        "index": 1,
        "hyperparams": {{"tile_m": 64, "tile_n": 64, "tile_k": 64}},
        "rationale": "增大 K 维以提升数据 reuse,适合 memory-bound 场景",
        "predicted_regression_risk": "medium",
        "risk_rationale": "较大的 K tile 可能增加寄存器压力，但不引入额外 launch 或 atomic"
    }}
]
```
