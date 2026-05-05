"""
代码规范化与提取 —— 从 LLM 输出中提取 CUDA 代码。
"""

from __future__ import annotations

import re


def extract_cuda_code(llm_output: str) -> str:
    """
    从 LLM 输出中提取 CUDA/C++ 代码块。

    支持:
    - ```cuda ... ``` 块
    - ```cpp ... ``` 块
    - ```c ... ``` 块
    - 裸代码(无 markdown 包裹)
    """
    # 尝试提取 fenced code block
    patterns = [
        r"```cuda\s*\n(.*?)```",
        r"```cpp\s*\n(.*?)```",
        r"```c\s*\n(.*?)```",
        r"```\s*\n(.*?)```",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, llm_output, re.DOTALL)
        if matches:
            # 取最长的匹配(通常是完整代码)
            code = max(matches, key=len)
            return code.strip()

    # 如果没有 code block,检查是否整个输出就是代码
    if _looks_like_cuda(llm_output):
        return llm_output.strip()

    return llm_output.strip()


def _looks_like_cuda(text: str) -> bool:
    """启发式判断文本是否是 CUDA 代码。"""
    indicators = [
        "#include",
        "__global__",
        "__device__",
        "__shared__",
        "cudaMalloc",
        "<<<",
        "threadIdx",
        "blockIdx",
        "blockDim",
    ]
    return any(ind in text for ind in indicators)


def normalize_code_formatting(code: str) -> str:
    """统一代码格式:去除多余空行,确保换行符一致。"""
    lines = code.split("\n")
    # 去除末尾空白
    lines = [line.rstrip() for line in lines]
    # 合并连续空行(最多保留一个)
    result = []
    prev_empty = False
    for line in lines:
        if not line.strip():
            if not prev_empty:
                result.append("")
            prev_empty = True
        else:
            result.append(line)
            prev_empty = False
    return "\n".join(result).strip() + "\n"
