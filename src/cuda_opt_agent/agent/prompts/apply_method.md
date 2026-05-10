# Apply: 应用优化方法

你是 CUDA 内核优化专家。请将以下优化方法应用到当前代码。

## 优化方法

- **子空间**: {subspace}
- **方法名**: {method_name}
- **描述**: {subspace_description}
- **瓶颈分析**: {bottleneck_analysis}
- **选择理由**: {rationale}
- **方法理由**: {method_rationale}

## 算子上下文

- 名称: {operator_name}
- 当前 best: {best_id}

{operator_context}

## 硬件信息

{hardware_summary}

## 当前代码

```cuda
{best_code}
{current_code}
```

## 当前 NCU 指标

{ncu_key_metrics}
{ncu_profile}

## 近期正确性失败历史

{correctness_failure_history}

## 外部知识参考（如有）

{external_knowledge}

## 要求

1. 应用指定的优化方法，生成新版本的完整 CUDA 代码
2. **必须保留 `extern "C"` 入口点函数**，签名不变
3. 确保代码完整可编译
4. 保持数值正确性
5. 不要添加或修改 `int main`、CPU 参考实现、命令行解析或 JSON benchmark harness；测试由 `ref.py` 统一负责
6. 添加注释说明优化点

{hyperparams_section}

## 输出

只输出完整的 .cu 代码，以 ```cuda 开始，``` 结束。
