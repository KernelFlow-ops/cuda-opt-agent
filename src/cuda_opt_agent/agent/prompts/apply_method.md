你正在优化一个 CUDA {operator_name} 算子。请在当前最佳版本基础上应用以下优化方法。

## 任务上下文
{operator_context}

## 优化方法
- 方法名: {method_name}
- 理由: {method_rationale}
{hyperparams_section}

## 硬件信息
{hardware_summary}

## 当前最佳版本 ({best_id}) 完整代码
```cuda
{best_code}
```

## 当前版本的 ncu 关键指标
{ncu_key_metrics}

## ⚠️ 近期 Correctness 失败记录 (务必避免同类错误)
{correctness_failure_history}

## 要求
1. 在当前最佳版本的基础上,**只应用上述一个方法**
2. 输出**完整的新 .cu 文件**（不要用 patch 或 diff,给完整代码）
3. 保持正确性校验和 benchmark 框架不变（main 函数、命令行参数等）；必须继续支持 `--warmup N` 和 `--rounds N`，不要只改成 `--iters`
4. 在代码开头用注释标注:本次的方法名和超参
5. 确保边界条件正确
6. 代码必须可直接用 nvcc 编译
7. 校验 JSON 必须包含字段: `correct`, `max_abs_error`, `max_rel_error`, `message`
8. Benchmark JSON 必须包含字段: `latency_ms_median`, `latency_ms_p95`, `throughput_gflops`; 或输出 `latencies_ms` 数组
9. `--check` 必须在用户请求的完整 shape 上运行 kernel,然后抽样输出元素并只为这些元素计算 CPU reference；不要只用 reduced/small shape 判断正确性
10. 保持并支持 `--shape key=value [key=value ...]` 参数,不要重新硬编码单一尺寸；例如 `--shape M=4096 N=4096 K=4096` 或 `--shape B=4096 N=4096`
11. **特别注意**：不要修改 main() 函数中的命令行参数解析和 JSON 输出格式，不要修改 CPU reference 的计算逻辑，只修改 kernel 实现部分

请直接输出完整的 .cu 代码,用 ```cuda 包裹。
