"""Per-node LLM temperature policy.

[修复] 降低 TEMP_PROPOSE_HP 和 TEMP_APPLY_METHOD,
减少高随机性导致的 correctness 失败概率。
"""

TEMP_BOOTSTRAP = 0.2
TEMP_ANALYZE = 0.1
TEMP_DECIDE = 0.1
TEMP_REPAIR = 0.1
TEMP_PROPOSE_HP = 0.5       # [修复] 从 0.8 降至 0.5, 减少无效探索
TEMP_APPLY_METHOD = 0.15    # [修复] 从 0.25 降至 0.15, 代码生成更确定性
TEMP_REFLECT = 0.5
