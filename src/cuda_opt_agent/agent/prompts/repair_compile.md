# Repair: 编译错误修复

你是 CUDA 编译调试专家。以下代码编译失败，请修复。

## 编译错误

```
{compile_error}
```

## 当前代码

```cuda
{code}
{current_code}
```

## 环境

- Compute capability: {compute_capability}
- CUDA version: {cuda_version}

{cumulative_hint}

## 要求

1. 修复所有编译错误
2. **保留 `extern "C"` 入口点函数签名不变**
3. 不要添加或修改 `int main`、CPU 参考实现、命令行解析或 JSON benchmark harness；测试由 `ref.py` 统一负责
4. 不要改变代码的功能逻辑
5. 确保修复后代码完整可编译

## 输出

只输出修复后的完整 .cu 代码，以 ```cuda 开始，``` 结束。
