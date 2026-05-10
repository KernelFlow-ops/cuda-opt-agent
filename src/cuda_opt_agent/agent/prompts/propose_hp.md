# Propose: 超参数候选方案

你是 CUDA 内核优化专家。请为以下优化方法提出超参数候选方案。

## 优化方法

- **子空间**: {subspace}
- **方法名**: {method_name}
- **描述**: {subspace_description}
- **瓶颈分析**: {bottleneck_analysis}
- **方法理由**: {method_rationale}

## 算子上下文

- 名称: {operator_name}

{operator_context}

## 可用超参数

{hyperparams_spec}
{hyperparams_schema}

## 当前代码

```cuda
{best_code}
{current_code}
```

## NCU 数据

{ncu_key_metrics}
{ncu_profile}

## 硬件信息

{hardware_summary}
{hardware_spec}

## Kernel Regime

{kernel_regime_info}

## 历史超参数结果（如有）

{hp_history}
{known_hp_trials}

## 外部知识参考

{external_knowledge}

## 要求

提出 {candidate_count}{hp_count} 个不同的超参数组合候选方案。每个方案应该:
1. 在合理范围内
2. 与硬件约束兼容
3. 互相有差异性（覆盖搜索空间）
4. 附带选择理由

## 输出格式（严格 JSON）

```json
{{
  "candidates": [
    {{
      "index": 1,
      "hyperparams": {{"param1": "value1", "param2": "value2"}},
      "rationale": "选择理由",
      "predicted_regression_risk": "low|medium|high",
      "risk_rationale": "为什么该候选风险可控"
    }}
  ]
}}
```
