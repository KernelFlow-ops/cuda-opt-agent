你正在优化一个 CUDA {operator_name} 算子。当前最佳版本 {best_id} 的 ncu 报告显示:

## 性能指标
{benchmark_metrics}

## 瓶颈分析（上一节点产出）
{analysis_summary}

## 已尝试且无效的方向（黑名单）
{blacklist}

## 其他硬件上的相似经验（仅供参考,以当前 ncu 为准）
{kb_hints}

## 硬件信息
{hardware_summary}

请基于以上信息,选择**下一个最值得尝试的优化方法**（只选一个）。

## 要求
1. 必须针对最严重的瓶颈
2. 必须用 ncu 中的具体数据支撑你的判断
3. 不能与黑名单重叠
4. 每次只引入一个方法或一组超参(控制变量原则)
5. 如果你认为已经没有合理的下一步,设置 give_up=true 并说明原因

## 输出格式 (严格 JSON)
```json
{{
    "method_name": "你选择的方法名 (自由字符串,简洁明确)",
    "has_hyperparams": true或false,
    "hyperparams_schema": {{}} 或 null,
    "rationale": "详细解释为什么选这个方法,引用 ncu 数据",
    "expected_impact": "high/medium/low + 预期说明",
    "confidence": 0.0到1.0之间的浮点数,
    "give_up": false
}}
```

如果 has_hyperparams=true,hyperparams_schema 必须描述超参名称和可选值范围,例如:
```json
"hyperparams_schema": {{
    "tile_m": {{"type": "int", "range": [16, 256]}},
    "tile_n": {{"type": "int", "range": [16, 256]}},
    "tile_k": {{"type": "int", "range": [8, 64]}}
}}
```
