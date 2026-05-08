你正在优化一个 CUDA {operator_name} 算子。当前最佳版本 {best_id} 的 ncu 报告显示:

## 任务上下文
{operator_context}

## 性能指标
{benchmark_metrics}

## 瓶颈分析（上一节点产出）
{analysis_summary}

## 已尝试且无效的方向（黑名单）
{blacklist}

## 历史方法/超参尝试
{method_history}

## 本轮已驳回的重复候选
{rejected_methods}

## 其他硬件上的相似经验（仅供参考,以当前 ncu 为准）
{kb_hints}

## 硬件信息
{hardware_summary}

## Library 基线参照（若有）
{library_baseline}

## 强制迭代策略
{forced_continue}

请基于以上信息，先做**元决策（meta decision）**，再选具体方法。

---

## 第一步：元决策

在输出 JSON 之前，请内部回答以下 5 个问题来判断风险和收益，但不要把低收益直接当作停止理由:

1. **analysis 中 `optimization_headroom.recommend_stop` 是 true 吗？** 如果 yes → 视为低收益提示，但仍必须继续选择一个候选方法
2. **kernel_regime 是 tiny 或 near_launch_floor=true 吗？** 如果 yes → 优先 launch-overhead-mitigation、algorithm-replacement、register-blocking 或减法优化
3. **最近连续 2+ 次尝试是否 ≥ best × 1.5？（catastrophic regression streak）** 如果 yes → 视为高风险提示，但仍必须换子空间继续尝试
4. **已失败方法的子空间覆盖了多少个？** 使用下方子空间清单逐一对照
5. **Library baseline 是否 ≥ best latency × 0.9？** 如果 yes → 记录为接近库实现，但未达到 max_iterations 前仍必须继续

硬性规则: 若 `当前迭代记录数 < 最大迭代数`，`meta_decision.should_continue` 必须为 true，`give_up` 必须为 false。`recommend_stop`、near launch floor、接近库基线、连续失败都只能降低 expected_impact/confidence 或改变子空间选择，不能提前停止。

### 优化子空间清单
- `reduction-restructure`: 重组归约拓扑（multi-CTA-per-channel, warp-per-channel, cross-warp reduction 变体）
- `cta-redistribution`: 改变 CTA↔数据映射（blocks_per_channel, channels_per_block, persistent CTA）
- `fusion`: Kernel 合并（fuse mean/var/normalize, fuse with activation, fuse with residual）
- `warp-primitive`: 使用 warp shuffle / vote / match 替代 shared-memory 操作
- `vectorization`: 数据向量化（half2, float4, __ldg, vectorized load/store）
- `shared-mem-tiling`: 共享内存 tile / double-buffer / async copy
- `register-blocking`: 寄存器阻塞 / 增加 ILP
- `algorithm-replacement`: 算法级替换（用库函数、换公式、online welford → two-pass 等）
- `launch-overhead-mitigation`: 减少 launch 开销（CUDA Graph, persistent kernel, 合并多次调用）
- `precision-conversion`: 精度相关优化（混合精度、TF32、FP8 等）

已失败的方法分别落在哪个子空间？请逐一标注。

若以上 5 个问题中有 2 个以上答 yes，在 rationale 中注明“收益可能很低”，但仍必须选择一个与黑名单不重叠的候选方法继续。

---

## 第二步：选择方法（仅当不放弃时）

要求：
1. **必须验证不与黑名单语义重叠**：在 `differentiation_from_failed` 字段中先列出"我的方法属于子空间 X，与已失败方法 [Y, Z] 的本质区别是 ABC"。
   - 反例：v1 (multi-CTA reduction split) 与 v4 (warp-per-channel packing) 表面不同，但都在重组归约拓扑——这就是"换皮"，必须拒绝
   - 如果某子空间已有 ≥ 2 次失败，该子空间在当前 run 内视为已穷尽，除非你能给出极强的差异论据
2. **必须给出否证条件**：`falsification_condition` 字段写"如果该方法不奏效，最可能的原因是什么"。这迫使你在选之前考虑失败模式
3. **expected_impact 校准**：若 baseline 已 < 0.01 ms，`expected_impact` 不允许填 "high"；最多 "medium"，且必须解释为什么能突破 launch floor
4. 控制变量原则：每次只引入一个方法或一组超参
5. 不能与黑名单重叠（包括子空间级重叠）
6. 参考历史方法/超参尝试，避免重复推荐
7. 不能再次选择"本轮已驳回的重复候选"

### 当 baseline < 0.01 ms 时的优先方向（如果继续）
以下方向对 launch-bound 场景更友好（若尚未尝试）：
- **algorithm-replacement**: 用 cuDNN/cuBLAS/CUTLASS 的等价实现、或换更简洁的算法公式
- **launch-overhead-mitigation**: persistent kernel / CUDA Graph capture / 减少 kernel 数量
- **register-blocking**: 寄存器内完成全部计算，消除 shared memory 访问和 __syncthreads
- **减法优化**: 不是"加新东西"，而是"砍掉不必要的索引计算、边界分支、冗余 __syncthreads"

---

## 输出格式 (严格 JSON)
```json
{{
    "meta_decision": {{
        "should_continue": true或false,
        "reasons": ["列出影响决策的关键因素"],
        "regression_streak": 最近连续回归次数(整数),
        "exhausted_subspaces": ["已有 >=1 次失败的子空间列表"],
        "remaining_promising_subspaces": ["尚未尝试或仅温和回归的子空间"],
        "near_library_baseline": true或false
    }},
    "method_name": "你选择的方法名 (自由字符串,简洁明确)",
    "subspace": "从上方子空间清单中选一个",
    "differentiation_from_failed": "明确说明本方法与已失败方法的本质差异（非表面差异）",
    "falsification_condition": "本方法失败的最可能原因是什么",
    "has_hyperparams": true或false,
    "hyperparams_schema": {{}} 或 null,
    "rationale": "详细解释为什么选这个方法,引用 ncu 数据和 analysis 结论",
    "expected_impact": "high/medium/low + 预期说明（baseline<0.01ms 时不允许 high）",
    "confidence": 0.0到1.0之间的浮点数,
    "give_up": false,
    "give_up_reason_type": "optimal_reached | exhausted_search | catastrophic_streak | near_library_baseline | null"
}}
```

如果 has_hyperparams=true,hyperparams_schema 必须描述超参名称和可选值范围,例如:
```json
"hyperparams_schema": {{
    "tile_m": {{"type": "int", "range": [16, 256]}},
    "tile_n": {{"type": "int", "range": [16, 256]}},
    "tile_k": {{"type": "int", "range": [8, 64]}}
}}
```

只有在 `当前迭代记录数 >= 最大迭代数` 时，才允许 meta_decision.should_continue=false，并设 give_up=true 填写 give_up_reason_type:
- `optimal_reached`: baseline 已在 launch floor 或接近库基线
- `exhausted_search`: 所有合理子空间已穷尽
- `catastrophic_streak`: 连续灾难性回归
- `near_library_baseline`: 与 cuDNN/cuBLAS 相当，无需继续
