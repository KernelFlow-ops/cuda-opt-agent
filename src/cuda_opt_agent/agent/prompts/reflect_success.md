优化成功!请反思这次迭代。

## 本次方法
- 方法名: {method_name}
- 超参: {hyperparams}

## 性能变化
- 前 ({parent_id}): latency = {parent_latency_ms:.4f} ms
- 后 ({new_id}): latency = {new_latency_ms:.4f} ms
- 加速比: {speedup:.2f}x

## ncu 关键指标变化
{ncu_diff}

## 要求
请输出 JSON 格式的反思:

```json
{{
    "why_effective": "分析为什么这个方法有效,引用具体的指标变化",
    "next_suggestion": "基于当前状态,建议下一步关注什么方向",
    "kb_write_suggestion": {{
        "should_write": true或false,
        "notes": "如果写入知识库,备注什么信息"
    }}
}}
```


