# Reflect: 成功优化总结

优化迭代 #{iteration} 成功！

## 方法: {method}
## 加速比: {speedup}x
## 延迟: {latency_ms} ms

## 代码变更摘要

{code_diff_summary}

## Kernel Regime

{kernel_regime}

## Risk Signals

{risk_signals}

## 下一步建议

请简要总结:
1. 这次优化为什么有效
2. 下一步最有前景的优化方向
3. 需要注意的潜在风险

输出 JSON:
```json
{{
  "success_reason": "...",
  "next_direction": "...",
  "risks": "..."
}}
```
