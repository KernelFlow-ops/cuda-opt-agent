优化未成功,请反思这次迭代。

## 本次方法
- 方法名: {method_name}
- 超参: {hyperparams}

## 性能变化
- best ({best_id}): latency = {best_latency_ms} ms
- 尝试 ({trial_id}): latency = {trial_latency_ms} ms (或编译/校验失败)
- 结果: {failure_reason}

## Correctness 失败详情
{correctness_failure_detail}

## ncu 关键指标 (若有)
{ncu_report}

## 要求
请输出 JSON 格式的反思:

```json
{{
    "why_ineffective": "分析为什么这个方法没有带来改善（如果是 correctness 失败,请详细分析数值错误的根因,例如:边界越界、精度累积、算法逻辑错误、共享内存 bank conflict 导致数据损坏等）",
    "avoid_similar": "描述应该避免的类似方向",
    "correctness_root_cause": "如果是 correctness 失败,说明根本原因和修复建议",
    "blacklist_extension": "建议将哪些方向/超参约束加入黑名单",
    "kb_write_suggestion": {{
        "should_write": true或false,
        "notes": "如果写入知识库,备注什么信息(失败经验也有价值)"
    }}
}}
```
