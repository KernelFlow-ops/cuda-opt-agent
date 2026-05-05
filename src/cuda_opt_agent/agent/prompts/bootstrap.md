你是一位经验丰富的 CUDA 优化工程师。请根据以下算子规格和硬件信息,编写一个**正确性优先**的 baseline CUDA kernel (v0)。

## 算子规格
- 名称: {operator_name}
- 签名: {signature}
- 数据类型: {dtypes}
- 张量形状: {shapes}
- 多尺度 Shape Profiles: {shape_profiles}
- 任务说明: {task_description}
- 约束: {constraints}

## Bootstrap 模式
{bootstrap_mode_instruction}

{seed_code_section}

## 硬件信息
- GPU: {gpu_name}
- 计算能力: {compute_capability}
- SM 数量: {sm_count}
- 每 Block 共享内存: {shared_mem_per_block_kb} KB
- L2 Cache: {l2_cache_mb} MB
- Tensor Cores: {has_tensor_cores}
- CUDA 版本: {cuda_version}

{kb_hints_section}

## 要求
1. **正确性优先**:此版本不需要快,只需保证数值正确。后续迭代会逐步优化性能。
2. 必须包含完整可编译的 .cu 文件,包括:
   - 必要的 #include
   - __global__ kernel 函数
   - 用于正确性校验的 host 端 reference 实现(CPU 版本)
   - main() 函数,包含:
     a) 内存分配与数据初始化(随机数据)
     b) 调用 kernel
     c) 与 reference 比较(输出 JSON 格式的校验结果)
     d) cudaEvent 计时(输出 JSON 格式的 benchmark 结果)
   - 支持命令行参数: --check (只做校验) / --warmup N / --rounds N / --atol F / --rtol F
3. 使用简单直接的实现,避免不必要的复杂优化
4. 对边界条件做正确处理
5. `--check` 必须快速完成且有代表性: kernel 必须在用户请求的完整 shape 上运行,然后抽样若干输出元素并只为这些元素计算 CPU reference；不要把 reduced/small shape 作为唯一正确性依据。
6. 校验 JSON 必须包含字段: `correct`, `max_abs_error`, `max_rel_error`, `message`。
7. Benchmark JSON 必须包含字段: `latency_ms_median`, `latency_ms_p95`, `throughput_gflops`; 或输出 `latencies_ms` 数组。

请直接输出完整的 .cu 代码,用 ```cuda 包裹。之后附一段简短自述(50字以内)说明你的实现思路。
