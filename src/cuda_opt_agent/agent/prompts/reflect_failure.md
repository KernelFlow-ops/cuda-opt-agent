# Reflect: 优化失败分析

优化迭代 #{iteration} 未通过。

## 方法: {method}
## 失败原因: {failure_reason}
## 连续失败次数: {consecutive_rejects}

## 失败详情

{failure_details}

## 外部知识参考（来自 Web 搜索）

{external_knowledge}

## Kernel Regime

{kernel_regime}

## Risk Signals

{risk_signals}

## 分析要求

1. 分析失败的根因
2. 判断是否应该换一个完全不同的方向
3. 如果有外部知识参考，分析其中有用的启发

输出 JSON:
```json
{{
  "root_cause": "...",
  "should_change_direction": true,
  "recommended_next_subspace": "...",
  "lessons_learned": "...",
  "external_insights": "..."
}}
```
