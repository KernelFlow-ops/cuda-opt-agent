"""Per-node LLM temperature policy.

[修复] 降低 TEMP_PROPOSE_HP 和 TEMP_APPLY_METHOD,
减少高随机性导致的 correctness 失败概率。

[改进] 新增:
  - TEMP_DECIDE_AFTER_REGRESSION: 连续回归后升温 decide,鼓励换思路
  - get_dynamic_decide_temperature(): 根据 regression streak 动态调整
"""

TEMP_BOOTSTRAP = 0.2
TEMP_ANALYZE = 0.1
TEMP_DECIDE = 0.1
TEMP_DECIDE_AFTER_REGRESSION = 0.4   # [改进] 连续 ≥ 2 次回归时升温, 鼓励换思路
TEMP_REPAIR = 0.1
TEMP_PROPOSE_HP = 0.5       # [修复] 从 0.8 降至 0.5, 减少无效探索
TEMP_APPLY_METHOD = 0.15    # [修复] 从 0.25 降至 0.15, 代码生成更确定性
TEMP_REFLECT = 0.5


def get_dynamic_decide_temperature(regression_streak: int = 0) -> float:
    """
    [改进] 根据连续回归次数动态调整 decide 节点的 temperature。

    - 0-1 次回归: 使用默认 TEMP_DECIDE (0.1), 保持确定性
    - ≥ 2 次回归: 升温至 TEMP_DECIDE_AFTER_REGRESSION (0.4),
      鼓励 LLM 跳出当前思维模式，考虑更不同的方向或 give_up

    原理：连续回归说明当前推理方向可能有系统性偏差,
    低温会让 LLM 沿相同偏差更"自信"地重复错误,
    升温可以增加多样性和跳出局部最优的概率。
    """
    if regression_streak >= 2:
        return TEMP_DECIDE_AFTER_REGRESSION
    return TEMP_DECIDE
