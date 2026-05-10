# Bootstrap: 生成基准实现

你是 CUDA 内核优化专家。请为以下算子生成**初始基准实现**。

{bootstrap_mode_instruction}

## 算子信息

- **名称**: {operator_name}
- **签名**: {signature}
- **数据类型**: {dtypes}
- **形状**: {shapes}
- **Shape profiles**: {shape_profiles}
- **任务描述**: {task_description}

## 硬件信息

- GPU: {gpu_name}
- Compute capability: {compute_capability}
- SM count: {sm_count}
- Shared memory/block: {shared_mem_per_block_kb} KB
- L2 Cache: {l2_cache_mb} MB
- Tensor Cores: {has_tensor_cores}
- CUDA version: {cuda_version}

{kb_hints_section}

{seed_code_section}

{external_reference}

## 要求

### 1. CUDA Kernel 代码

生成一个**正确**的 CUDA kernel 实现，满足:
- 代码完整可编译（包含所有必要的 #include）
- 正确性优先；v0 baseline 必须先保证可编译和数值正确，再考虑性能
- 数值正确性（结果与 PyTorch 参考一致，误差在 atol=1e-2, rtol=1e-2 以内）
- **必须包含 `extern "C"` 入口点函数**，函数签名为:
  ```c
  extern "C" void {operator_name}_kernel({extern_c_params})
  ```
- 入口点函数内部调用实际的 CUDA kernel，配置好 grid/block 参数
- 不要提供 `int main`、CPU 参考实现、命令行解析或 JSON benchmark harness；正确性和测速由生成的 `ref.py` 统一负责
- 这是初始基准版本，不需要极致优化，但应该是合理的基础实现

### 2. 代码结构规范

```cuda
#include <cuda_runtime.h>
#include <cuda_fp16.h>
// ... other includes

// 实际的 CUDA kernel
__global__ void {operator_name}_kernel_impl(...) {{
    // kernel implementation
}}

// extern "C" 入口点（统一 benchmark 接口）
extern "C" void {operator_name}_kernel({extern_c_params}) {{
    // 配置 grid, block
    dim3 block(...);
    dim3 grid(...);
    {operator_name}_kernel_impl<<<grid, block>>>(...);
}}
```

### 3. 约束

{constraints}

## 输出格式

只输出完整的 .cu 代码，不要有多余的解释。代码块以 ```cuda 开始，``` 结束。
