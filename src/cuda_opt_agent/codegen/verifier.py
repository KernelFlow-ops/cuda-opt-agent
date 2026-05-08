"""
代码结构验证 —— 在编译前做基础检查。
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class VerifyResult:
    valid: bool = True
    warnings: list[str] = None
    errors: list[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.errors is None:
            self.errors = []


def verify_code_structure(code: str) -> VerifyResult:
    """
    对 CUDA 代码做基础结构检查。
    不替代编译器,只做快速预筛。
    """
    result = VerifyResult()

    if not code.strip():
        result.valid = False
        result.errors.append("代码为空")
        return result

    # 1) 检查是否有 __global__ kernel
    if "__global__" not in code:
        result.warnings.append("未发现 __global__ kernel 定义")

    # 2) 检查括号匹配
    _check_brackets(code, result)

    # 3) 检查常见错误模式
    _check_common_errors(code, result)

    # 4) 检查是否包含必要的 include
    if "#include" not in code:
        result.warnings.append("未发现任何 #include")

    result.valid = len(result.errors) == 0
    return result


def _check_brackets(code: str, result: VerifyResult) -> None:
    """检查大括号、圆括号、方括号是否匹配。"""
    # 去掉字符串和注释
    clean = _strip_strings_and_comments(code)

    pairs = {"{": "}", "(": ")", "[": "]"}
    stack = []
    for i, ch in enumerate(clean):
        if ch in pairs:
            stack.append((ch, i))
        elif ch in pairs.values():
            if not stack:
                result.errors.append(f"多余的闭合括号 '{ch}' 在位置 {i}")
                return
            open_ch, open_pos = stack.pop()
            expected = pairs[open_ch]
            if ch != expected:
                result.errors.append(
                    f"括号不匹配: '{open_ch}' (位置 {open_pos}) "
                    f"对应 '{ch}' (位置 {i}), 期望 '{expected}'"
                )
                return

    if stack:
        for open_ch, open_pos in stack:
            result.errors.append(f"未闭合的 '{open_ch}' 在位置 {open_pos}")


def _check_common_errors(code: str, result: VerifyResult) -> None:
    """检查常见的 CUDA 编程错误。"""
    lines = code.split("\n")

    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # 检查 __global__ 函数返回类型不是 void
        if "__global__" in stripped and "void" not in stripped.split("__global__")[0] + stripped.split("__global__")[1].split("(")[0]:
            if "void" not in stripped:
                result.warnings.append(f"第 {i} 行: __global__ 函数应返回 void")

        # 检查 __shared__ 变量在全局作用域
        if stripped.startswith("__shared__") and not any(
            stripped.startswith(f"{prefix} __shared__") for prefix in ["//", "/*", "*"]
        ):
            # 这可能是正常的,只是一个提示
            pass


def _strip_strings_and_comments(code: str) -> str:
    """去掉字符串字面量和注释,用于括号匹配检查。"""
    # 去掉单行注释
    code = re.sub(r"//[^\n]*", "", code)
    # 去掉多行注释
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    # 去掉字符串
    code = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', '""', code)
    code = re.sub(r"'[^'\\]*(?:\\.[^'\\]*)*'", "''", code)
    return code


def generate_diff(old_code: str, new_code: str) -> str:
    """生成两版代码的简单 diff。"""
    import difflib
    old_lines = old_code.splitlines(keepends=True)
    new_lines = new_code.splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines, new_lines,
        fromfile="best.cu", tofile="new.cu",
        lineterm="",
    )
    return "".join(diff)
