# Decide: 选择下一个优化子空间

你是 CUDA 内核优化专家。根据以下信息，选择**下一个最有前景的优化子空间**。

---

## 当前硬件信息

{hardware_spec}

## 当前最优版本代码

```cuda
{current_best_code}
```

## 当前 NCU 剖析数据

{ncu_profile}

## 优化历史记录

{optimization_history}

## Runtime Risk Signals

{runtime_signals}

## 黑名单（已尝试无效的子空间，本轮不要再选）

{blacklist}

## 已有效使用的方法（已采纳的优化，不要重复使用相同策略）

{effective_methods}

## 外部知识参考（如有）

{external_knowledge}

---

## 可选优化子空间（共 20 个，分 4 层级）

### Layer 1 — 内存层级优化（最高优先级）
1. **memory-coalescing** [No-HP] — 全局内存合并访问: AoS→SoA, corner turning, alignment padding
2. **shared-mem-tiling** [HP] — 共享内存分块与重用: tiling, double/multi-stage buffering
3. **bank-conflict-resolution** [HP] — 共享内存 Bank 冲突消除: padding, swizzling
4. **vectorized-memory-access** [HP] — 向量化加载/存储: float4/int4, __ldg, half2
5. **async-memory-pipeline** [HP] — 异步内存流水线: cp.async (SM80+), TMA (SM90+)
6. **l2-cache-tuning** [HP] — L2 缓存窗口调优: persistence policy (SM80+)
7. **register-optimization** [HP] — 寄存器压力管理: __launch_bounds__, -maxrregcount
8. **texture-constant-memory** [No-HP] — 纹理/常量内存利用

### Layer 2 — 执行配置优化
9. **occupancy-tuning** [HP] — 占用率与 Launch 配置调优
10. **cta-redistribution** [HP] — CTA-数据映射重组: persistent CTA, split-K, tile scheduling
11. **thread-coarsening** [HP] — 线程粗化: elements_per_thread, grid-stride loop
12. **warp-specialization** [HP] — Warp 特化: producer/consumer 分工 (SM90+)

### Layer 3 — 指令级优化
13. **instruction-optimization** [No-HP] — fast math, FMA, rsqrt, pragma unroll
14. **control-flow-divergence** [No-HP] — 控制流分歧消除
15. **precision-conversion** [HP] — 混合精度, TF32, FP8, half2 打包
16. **warp-primitive** [No-HP] — Warp 级原语替代: shuffle, vote, cooperative groups

### Layer 4 — 算法与架构级优化
17. **reduction-restructure** [HP] — 归约拓扑重组
18. **algorithm-replacement** [No-HP] — 算法级替换: cuBLAS/CUTLASS/FlashAttention
19. **fusion** [No-HP] — Kernel 合并
20. **launch-overhead-mitigation** [HP] — Launch 开销消减: CUDA Graphs

### 子空间间的依赖与互斥关系

**互相增强（建议组合）：**
- memory-coalescing + vectorized-memory-access
- shared-mem-tiling + bank-conflict-resolution
- shared-mem-tiling + async-memory-pipeline
- occupancy-tuning + register-optimization
- thread-coarsening + vectorized-memory-access
- warp-specialization + async-memory-pipeline

**互相冲突（避免同时）：**
- occupancy-tuning ↔ register-optimization (寄存器数量权衡)
- thread-coarsening ↔ occupancy-tuning (线程数权衡)
- warp-primitive ↔ shared-mem-tiling (共享内存使用冲突)
- split-k (cta-redistribution) ↔ launch-overhead-mitigation

**架构门控（检查 compute capability）：**
- async-memory-pipeline (cp.async) → SM80+
- async-memory-pipeline (TMA) → SM90+
- warp-specialization → SM90+
- precision-conversion (FP8) → SM89+
- precision-conversion (TF32) → SM80+
- l2-cache-tuning → SM80+

---

## 决策要求

1. 仔细分析 NCU 数据，找到当前代码的**主要瓶颈**（memory-bound / compute-bound / latency-bound）
2. 结合历史记录，避免重复选择已失败的方向
3. 避免选择已有效使用过的相同策略（除非有明确的新角度）
4. 检查架构兼容性（SM 版本门控）
5. 考虑子空间间的增强/冲突关系
6. 优先选择高优先级、与当前瓶颈匹配的子空间

## 输出格式（严格 JSON）

```json
{{
  "bottleneck_analysis": "当前代码的主要瓶颈分析（基于 NCU 数据和代码）",
  "chosen_subspace": "子空间名称（必须是上面 20 个之一）",
  "has_hyperparams": true,
  "rationale": "选择理由：为什么这个子空间最适合当前瓶颈",
  "expected_improvement": "预期改善方向和幅度",
  "synergy_note": "与已有优化的协同效应说明（如有）",
  "conflict_check": "与已有优化的冲突检查结果"
}}
```
