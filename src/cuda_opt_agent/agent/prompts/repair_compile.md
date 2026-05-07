以下 CUDA 代码编译或校验失败,请修复。

## 错误信息 (包含所有历史修复尝试的错误)
```
{compile_error}
```
{cumulative_hint}

## 当前代码
```cuda
{code}
```

## 硬件信息
- 计算能力: {compute_capability}
- CUDA 版本: {cuda_version}

## 要求
1. 修复错误,输出**完整的修复后 .cu 文件**
2. 不要改变算法逻辑和优化策略,只修复编译问题
3. 如果错误与硬件不兼容相关,做最小改动使其兼容
4. 如果错误是校验超时,让 `--check` 在完整 benchmark shape 上运行 kernel,然后抽样输出元素并只为这些元素计算 CPU reference；不要把 reduced/small shape 作为唯一正确性依据
5. 校验 JSON 必须包含字段: `correct`, `max_abs_error`, `max_rel_error`, `message`
6. Benchmark JSON 必须包含字段: `latency_ms_median`, `latency_ms_p95`, `throughput_gflops`; 或输出 `latencies_ms` 数组
7. 必须继续支持 benchmark 参数 `--warmup N` 和 `--rounds N`; 如果同时支持 `--iters N` 也可以,但不能删除 `--rounds`
8. 代码用 ```cuda 包裹
9. 如果上方的错误信息包含多次修复尝试,请仔细分析所有错误的共性和根因,从源头解决问题

请直接输出修复后的完整代码。


