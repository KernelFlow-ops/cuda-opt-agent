优化未成功,请反思这次迭代。

## 本次方法
- 方法名: {method_name}
- 超参: {hyperparams}

## 性能变化
- best ({best_id}): latency = {best_latency_ms} ms
- 尝试 ({trial_id}): latency = {trial_latency_ms} ms (或编译/校验失败)
- 回归倍数: {regression_ratio}x
- 结果: {failure_reason}

## Correctness 失败详情
{correctness_failure_detail}

## ncu 关键指标 (若有)
{ncu_report}

## 当前 kernel regime
{kernel_regime}

## 要求
请输出 JSON 格式的反思:

```json
{{
    "why_ineffective": "分析为什么这个方法没有带来改善。如果是 correctness 失败,请详细分析数值错误的根因（边界越界、精度累积、算法逻辑错误等）。如果是性能回归,重点分析引入了哪些额外开销（调度、同步、全局内存流量、线程利用率下降等）以及为什么这些开销 > 预期收益",

    "regression_severity": "catastrophic (>3x) | severe (1.5-3x) | mild (1.0-1.5x) | correctness",

    "anti_pattern": {{
        "subspace": "属于哪个子空间，从以下清单选择: reduction-restructure / cta-redistribution / fusion / warp-primitive / vectorization / shared-mem-tiling / register-blocking / algorithm-replacement / launch-overhead-mitigation / precision-conversion",
        "pattern_signature": "结构化签名，用简短的 kebab-case 标签，例如: 'multi-cta-per-output', 'cross-warp-reduction', 'half2-vectorize-tiny-kernel', 'fused-mean-var-normalize', 'warp-per-channel-packing'。不要写长句",
        "trigger_conditions": {{
            "baseline_latency_lt_ms": "当 baseline latency < 该值时触发此反模式，数值或 null",
            "reduction_elements_lt": "当每通道归约元素数 < 该值时触发，数值或 null",
            "requires_atomic_or_partial_buffer": true或false,
            "increases_kernel_count": true或false,
            "increases_block_sync_count": true或false,
            "reduces_thread_count": true或false,
            "adds_global_memory_traffic": true或false
        }},
        "applies_to_operator_class": "batchnorm / layernorm / reduction / softmax / * (通用)"
    }},

    "avoid_similar": "描述应该避免的类似方向，必须引用 anti_pattern.pattern_signature",

    "correctness_root_cause": "如果是 correctness 失败,说明根本原因和修复建议。否则写 'N/A (性能回归,correctness 通过)'",

    "blacklist_extension": "建议将哪些方向/超参约束加入黑名单。使用 subspace + pattern_signature 描述",

    "kb_write_suggestion": {{
        "should_write": true或false,
        "polarity": "positive 或 negative",
        "notes": "如果写入知识库,备注什么信息。失败经验也很有价值，应该写入（polarity=negative）以帮助未来 run 避免重蹈覆辙"
    }}
}}
```

## 关键准则
1. **regression_severity = catastrophic 时（>3x 回归）**:
   - 必须给出完整的 `anti_pattern.trigger_conditions`，让系统能自动拦截
   - `kb_write_suggestion.should_write` 必须 = true，`polarity` = "negative"
   - `anti_pattern.pattern_signature` 必须是简短、可被字符串匹配的标签
2. **pattern_signature 命名规范**: 使用 kebab-case，3-5 个词，描述"做了什么"而非"叫什么名字"
   - 好: `multi-cta-per-channel`, `warp-shuffle-replace-smem`, `fused-reduction-normalize`
   - 坏: `按通道多CTA拆分归约以扩大grid并行度`（太长，不可匹配）
3. **regime-aware 分析**: 如果 kernel_regime 是 tiny/small，优先从 launch overhead / 固定开销角度解释回归，而非从 "还需要更复杂的优化" 角度
