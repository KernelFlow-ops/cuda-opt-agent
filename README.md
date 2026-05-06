# CUDA 算子优化智能体

LLM 驱动的 CUDA 算子迭代优化 Agent。用户提供结构化任务、自然语言描述或已有 `.cu` 文件后，Agent 会自动完成 **生成/接入 v0 baseline -> 编译 -> 正确性校验 -> Benchmark -> ncu Profile -> 瓶颈分析 -> 选择优化方法 -> 生成新版本 -> 接受/回退 -> 沉淀经验** 的闭环。

## 核心能力

| 能力 | 说明 |
|------|------|
| 多输入模式 | 支持 spec 文件、自然语言任务、已有 `.cu` 文件作为 v0 起点 |
| LLM 自主优化 | 不预设固定优化菜单，由 LLM 基于 ncu 和历史结果选择下一步 |
| 实测闭环 | 每个版本都经过编译、正确性校验、Benchmark 和 Profile |
| 断点续跑 | 中断后可按算子名、run id 或目录续跑 |
| 过程落盘 | 代码、状态、历史、日志、benchmark、reasoning 全部保存到 run 目录 |
| 超参历史感知 | 实际 HP 组合会写入迭代记录,并注入 Decide / Propose HP prompts 避免重复失败组合 |
| 经验沉淀 | 跨运行知识库会作为软提示注入后续优化 |

## 系统要求

| 依赖 | 要求 |
|------|------|
| Python | `>=3.10` |
| CUDA Toolkit | 需要 `nvcc` 可用 |
| Nsight Compute | 需要 `ncu` 可用 |
| NVIDIA Driver | 需要能访问目标 GPU |
| LLM API | Anthropic 或 OpenAI 兼容接口 |

## 安装

```bash
pip install -e ".[dev]"
```

验证 CLI 是否可用:

```bash
cuda-opt --help
```

## 配置

在项目根目录创建 `.env`，填入至少一个 provider 的 API key。

```dotenv
# LLM provider: anthropic 或 openai
LLM_PROVIDER=anthropic

# Anthropic
ANTHROPIC_API_KEY=your_key_here
ANTHROPIC_MODEL=claude-sonnet-4-20250514

# OpenAI 或 OpenAI-compatible gateway
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o
# OPENAI_BASE_URL=https://your-gateway.example.com/v1

# 运行默认值
DEFAULT_DTYPE=fp16
MAX_ITERATIONS=30
CONSECUTIVE_REJECT_LIMIT=5
ACCEPT_EPSILON=0.005
COMPILE_REPAIR_MAX_RETRIES=3
DECIDE_RESELECT_MAX_RETRIES=3
HP_CANDIDATE_COUNT=5
HP_COMPILE_WORKERS=0
BENCHMARK_WARMUP_ROUNDS=10
BENCHMARK_MEASURE_ROUNDS=100
MULTI_SHAPE_AGGREGATOR=mean

# 产物目录
RUNS_DIR=runs
KNOWLEDGE_BASE_DIR=knowledge_base

# Windows/终端行为
CONSOLE_ENCODING=auto
ENABLE_BEST_SYMLINK=0
```

配置优先级为: `CLI 显式参数 > 环境变量 > .env > 代码默认值`。

## 快速开始

### 结构化任务文件

适合复杂算子、多人协作、可复现任务。

```bash
cuda-opt new softmax --spec tasks/softmax_fp16.yaml
```

### 自然语言任务

适合快速试验。

```bash
cuda-opt new softmax --task "写一个 fp16 softmax，沿最后一维归一化" --shape 1024,1024
```

多尺度 sweep:

```bash
cuda-opt run gemm --shapes "1024^3;2048^3;4096^3" --multi-shape-aggregator worst
```

### 优化已有 CUDA 文件

适合从现有实现继续调优。

```bash
cuda-opt tune kernels/fused_attention.cu --operator fused_attention --task "保持 mask 语义不变"
```

### 兼容旧入口

`run` 仍可用，推荐新任务优先使用 `new` 或 `tune`。

```bash
cuda-opt run gemm --shape 4096,4096,4096 --dtype fp16 --max-iters 30
```

## 任务输入模式

| 模式 | 命令 | 适用场景 |
|------|------|----------|
| Spec 文件 | `cuda-opt new <operator> --spec <file>` | 复杂算子、多 shape、可版本化任务 |
| 自然语言 | `cuda-opt new <operator> --task "..."` | 快速描述算子需求 |
| 已有代码 | `cuda-opt tune <file.cu>` 或 `cuda-opt new <operator> --from-cu <file.cu>` | 以现有 `.cu` 作为 v0 baseline |
| 交互向导 | `cuda-opt new <operator>` | 未传 `--spec` / `--task` / `--from-cu` 时逐项询问 |

`--spec`、`--task`、`--from-cu` 互斥。启用 `--auto` 时必须显式传入其中一种输入模式。

## Spec 文件格式

支持 `.yaml`、`.yml`、`.toml`、`.json`。推荐 YAML。

```yaml
name: softmax
signature: "y[B,N] = softmax(x[B,N], dim=-1)"
dtypes: {x: fp16, y: fp16}
task_description: |
  实现一个数值稳定的 softmax，沿最后一维做归一化。
  要求：使用 online softmax 算法或 max-trick 防溢出。
constraints:
  - "B、N 都可能很大，要避免单 block 处理超长行"
shape_profiles:
  - {x: [1024, 1024], y: [1024, 1024]}
  - {x: [2048, 2048], y: [2048, 2048]}
  - {x: [4096, 4096], y: [4096, 4096]}
```

### Spec 字段

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `name` | string | 是 | 算子名称，如 `gemm`、`softmax`、`fused_attention` |
| `signature` | string | 是 | 算子语义签名 |
| `dtypes` | object | 否 | 张量名到 dtype 的映射 |
| `shapes` | object | 否 | 单一 shape 描述 |
| `constraints` | string list | 否 | 用户约束，会注入 prompt |
| `task_description` | string | 否 | 自由文本任务说明，会贯穿 bootstrap/analyze/decide/apply prompts |
| `seed_code_path` | string | 否 | 已有 `.cu` 文件路径，作为 v0 起点 |
| `shape_profiles` | object list | 否 | 多尺度 shape；当 `shapes` 为空时，第一个 profile 会作为默认 `shapes` |

### 使用已有代码作为 seed

```yaml
name: fused_attention
signature: "out = fused_attention(q, k, v, mask)"
dtypes: {q: fp16, k: fp16, v: fp16, mask: bool, out: fp16}
seed_code_path: ../kernels/fused_attention.cu
task_description: |
  优化已有 fused attention 实现。
  必须保持 mask、causal 语义和数值行为不变。
shape_profiles:
  - {q: [16, 16, 128, 64], k: [16, 16, 128, 64], v: [16, 16, 128, 64], out: [16, 16, 128, 64]}
```

当 `seed_code_path` 非空时，Agent 会把该 `.cu` 注入 bootstrap prompt，并要求 LLM 将其作为 v0 baseline。若缺少 correctness check 或 benchmark 框架，LLM 只做必要补齐，不应改变原始算法逻辑。

## CLI 参考

### 新建任务

```bash
cuda-opt new [operator] [OPTIONS]
```

常用选项:

| 选项 | 说明 |
|------|------|
| `--spec <file>` | 从 YAML/TOML/JSON spec 创建任务 |
| `--task <text>` | 自然语言任务说明 |
| `--from-cu <file.cu>` | 以已有 `.cu` 作为 v0 seed |
| `--sig <text>` | 算子签名 |
| `--shape <dims>` | 逗号分隔 shape，如 `1024,1024` |
| `--shapes <profiles>` | 分号分隔多 shape，如 `1024^3;2048^3` 或 `M=1024,N=1024,K=1024;M=2048,N=2048,K=2048` |
| `--shape-profile <name>` | 使用内置 profile: `small`、`medium`、`large`、`sweep` |
| `--dtype <dtype>` | 覆盖默认 dtype |
| `--max-iters <n>` | 覆盖最大迭代数 |
| `--multi-shape-aggregator <mode>` | 多 shape latency 聚合方式: `mean`、`worst`、`weighted` |
| `--auto` | 禁用交互向导，缺少输入模式时报错 |

### 优化已有文件

```bash
cuda-opt tune <file.cu> [OPTIONS]
```

常用选项:

| 选项 | 说明 |
|------|------|
| `--operator <name>` | 算子名；默认使用文件名 stem |
| `--task <text>` | 补充任务语义 |
| `--sig <text>` | 算子签名 |
| `--shape <dims>` | 逗号分隔 shape |
| `--shapes <profiles>` | 分号分隔多 shape profiles |
| `--shape-profile <name>` | 使用内置 profile: `small`、`medium`、`large`、`sweep` |
| `--dtype <dtype>` | 覆盖默认 dtype |
| `--max-iters <n>` | 覆盖最大迭代数 |
| `--multi-shape-aggregator <mode>` | 多 shape latency 聚合方式: `mean`、`worst`、`weighted` |

### 多尺度验证

`--shape` 保持旧单点行为；`--shapes` 和 `--shape-profile` 会对每个 shape profile 分别执行 correctness 与 benchmark。接受新版本的前提是所有 shape 都正确，且聚合 latency 相比 best 满足 `ACCEPT_EPSILON`。

内置 profiles:

| 算子 | profile | profiles |
|------|---------|----------|
| `gemm` | `small` / `medium` / `large` | `1024^3` / `2048^3` / `4096^3` |
| `gemm` | `sweep` | `1024^3;2048^3;4096^3` |
| `softmax` | `small` / `medium` / `large` | `B,N = 1024,1024` / `2048,2048` / `4096,4096` |
| `softmax` | `sweep` | `B,N = 1024,1024;4096,4096` |

生成的 `.cu` 必须支持通用命令行参数 `--shape key=value [key=value ...]`，例如:

```bash
./kernel --check --shape M=4096 N=4096 K=4096
./kernel --warmup 10 --rounds 100 --shape B=4096 N=4096
```

### HP 候选并发

HP search 阶段会先串行让 LLM 生成各候选代码，然后并行执行 nvcc 编译。编译成功后，correctness 和 benchmark 仍然串行执行，避免同一张 GPU 上并发运行导致校验/测速互相干扰。

`HP_COMPILE_WORKERS=0` 表示自动使用 `min(候选数, CPU 核数)`；设置为 `1` 可关闭并发，便于调试。

### 节点温度策略

不同节点显式使用不同 LLM temperature，不再使用全局固定值。

| 节点 | temperature | 目的 |
|------|-------------|------|
| bootstrap | `0.2` | 生成稳定、可编译的 baseline |
| analyze | `0.1` | 降低瓶颈分析发散 |
| decide | `0.1` | 稳定选择下一步方法并减少黑名单重选噪声 |
| repair_compile | `0.1` | 修复编译/校验问题时优先严谨 |
| propose_hp | `0.8` | 提高超参候选多样性，避免候选挤在相同区域 |
| apply_method | `0.25` | 应用方法时保持稳定代码生成 |
| reflect | `0.5` | 允许适度自由文本归因和经验沉淀 |

### 续跑

```bash
# 按算子名查找最近未完成 run
cuda-opt resume gemm

# 按 run id 查找 RUNS_DIR 下目录
cuda-opt resume gemm_run_20260501T120000

# 指定目录
cuda-opt resume --run-dir runs/gemm_run_20260501T120000

# 续跑时追加迭代预算
cuda-opt resume gemm --extra-iters 20
```

### 浏览与对比

```bash
# 查看所有运行
cuda-opt list

# 查看单次运行报告
cuda-opt show runs/gemm_run_20260501T120000

# 对比两个版本代码
cuda-opt diff runs/gemm_run_20260501T120000 v0 v3
```

兼容别名:

```bash
cuda-opt list-runs
cuda-opt show-run runs/gemm_run_20260501T120000
```

## CLI 与 .env 双通道配置

`new`、`tune`、`run` 支持以下显式 CLI 覆盖项。

| .env | CLI | 默认值 | 说明 |
|------|-----|--------|------|
| `DEFAULT_DTYPE` | `--dtype` | `fp16` | 默认 dtype |
| `MAX_ITERATIONS` | `--max-iters` | `30` | 最大迭代次数 |
| `CONSECUTIVE_REJECT_LIMIT` | `--consecutive-reject-limit` | `5` | 连续拒绝后停止 |
| `ACCEPT_EPSILON` | `--accept-epsilon` | `0.005` | 接受新版本所需相对提升 |
| `DECIDE_RESELECT_MAX_RETRIES` | `--decide-reselect-max-retries` | `3` | decide 选中黑名单方法后的重选次数 |
| `HP_CANDIDATE_COUNT` | `--hp-candidate-count` | `5` | 每轮超参候选数量 |
| `HP_COMPILE_WORKERS` | `--hp-compile-workers` | `0` | HP 候选并行编译 worker 数；`0` 自动，`1` 串行 |
| `MULTI_SHAPE_AGGREGATOR` | `--multi-shape-aggregator` | `mean` | 多 shape latency 聚合方式 |

仅 `.env` 配置项:

| .env | 默认值 | 说明 |
|------|--------|------|
| `LLM_PROVIDER` | `anthropic` | LLM provider |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-20250514` | Anthropic 模型 |
| `OPENAI_MODEL` | `gpt-4o` | OpenAI 模型 |
| `LLM_MODEL` | provider 默认值 | provider 专用模型未设置时的通用模型名 |
| `COMPILE_REPAIR_MAX_RETRIES` | `3` | 编译/校验失败后的 LLM 修复重试次数 |
| `BENCHMARK_WARMUP_ROUNDS` | `10` | benchmark warmup 轮数 |
| `BENCHMARK_MEASURE_ROUNDS` | `100` | benchmark 测量轮数 |
| `RUNS_DIR` | `runs` | 运行产物目录 |
| `KNOWLEDGE_BASE_DIR` | `knowledge_base` | 跨运行知识库目录 |
| `CONSOLE_ENCODING` | `auto` | 终端输出编码；可选 `auto`、`utf-8`、`gbk`、`default` |
| `ENABLE_BEST_SYMLINK` | 空/关闭 | Windows 默认不创建 `best` symlink；设为 `1` 时尝试创建 |

开发约定: 凡是“环境可配 + CLI 可覆盖”的字段，Typer 选项默认值必须为 `None`，只能在用户显式传入时覆盖 `load_config()` 的结果，避免 CLI 默认值吞掉 `.env`。

## 运行产物

默认保存在 `runs/<operator>_run_<timestamp>/`。

常见文件:

| 路径 | 说明 |
|------|------|
| `state.json` | 当前完整运行状态 |
| `history.jsonl` | 迭代历史追加日志 |
| `config.json` | 本次运行配置快照 |
| `reasoning_log.md` | 全局推理过程记录 |
| `iter<version>/code.cu` | 某次迭代生成的 CUDA 代码 |
| `iter<version>/compile.log` | 编译输出 |
| `iter<version>/benchmark.json` | benchmark 结果 |
| `best.txt` | 当前 best 版本目录引用 |

HP 搜索路径会把实际选中的 `hyperparams` 写入 `history.jsonl` 和 `state.json`。后续 Decide 会看到历史方法/超参尝试，Propose HP 会看到同方法已尝试组合，并被要求避开已失败的同方法同超参配置。

## 测试

```bash
# 不依赖 GPU/API 的测试
pytest tests/ -v -m "not api and not gpu"

# API 连通性测试，需要 API key
pytest tests/ -v -m api

# GPU 相关测试，需要 CUDA 环境
pytest tests/ -v -m gpu

# 覆盖率报告
pytest tests/ -v --cov=cuda_opt_agent --cov-report=html
```

## 项目结构

```text
src/cuda_opt_agent/
├── cli.py                 # CLI 入口 (Typer)
├── config.py              # 配置加载 (.env)
├── task_spec.py           # 任务规格文件加载 (YAML/TOML/JSON)
├── agent/                 # Agent 核心
│   ├── graph.py           # LangGraph 状态机
│   ├── nodes.py           # 所有节点实现
│   ├── state.py           # GraphState 定义
│   ├── llm_client.py      # LLM 调用封装
│   └── prompts/           # Prompt 模板
├── tools/                 # 工具层
│   ├── compile.py         # nvcc 编译
│   ├── profile.py         # ncu Profiling
│   ├── benchmark.py       # cudaEvent 测速
│   ├── correctness.py     # 数值正确性校验
│   ├── hardware.py        # GPU 硬件信息采集
│   └── web_search.py      # 可选 Web 检索
├── memory/                # 持久化层
│   ├── persistence.py     # 运行目录与 state 管理
│   ├── run_state.py       # RunState 高层封装
│   └── knowledge.py       # 跨运行知识库
├── codegen/               # 代码生成辅助
│   ├── normalizer.py      # 代码提取与格式化
│   └── verifier.py        # 代码结构预检
├── models/                # 数据模型 (Pydantic)
│   ├── data.py            # 所有核心数据结构
│   └── enums.py           # 枚举与归一化
└── tui/                   # TUI 界面 (Rich + Textual)
    ├── app.py             # TUI 主应用
    ├── widgets.py         # 面板组件
    └── live.py            # 实时推理流
```

## 设计原则

| 原则 | 说明 |
|------|------|
| LLM 自主决策 | 优化方法由 LLM 自行提出，不预设方法菜单 |
| 硬件感知 | GPU 信息注入所有 Prompt |
| 实测为准 | 只以真实硬件上的 latency 判断好坏 |
| 正确性优先 | v0 与每次迭代都必须通过 correctness check |
| 控制变量 | 每次只引入一个方法或一组超参，便于归因 |
| 过程透明 | 每次迭代的代码、ncu 报告、推理过程落盘 |
| 断点续跑 | Ctrl+C 后可用 `resume` 继续 |
| 经验沉淀 | 跨运行知识库以软提示形式注入 |

## 常见问题

### `nvcc` 找不到

确认 CUDA Toolkit 已安装，并且 `nvcc` 所在目录在 `PATH` 中。

```bash
nvcc --version
```

### `ncu` 找不到

确认 Nsight Compute 已安装，并且 `ncu` 在 `PATH` 中。

```bash
ncu --version
```

### `.env` 配置没有生效

检查是否通过 CLI 显式传入了同名覆盖参数。优先级为 `CLI 显式参数 > 环境变量 > .env > 代码默认值`。

### Windows 提示 `[WinError 1314] 客户端没有所需的特权`

这是 Windows 创建目录软链接所需权限不足导致的。当前版本默认把 `best.txt` 作为跨平台主机制，并在 Windows 上跳过 `best` symlink 创建，所以普通用户权限不会再刷出这个错误。

如果确实需要 `runs/<run>/best` 软链接，可以开启 Windows Developer Mode 或以管理员身份运行，并设置:

```dotenv
ENABLE_BEST_SYMLINK=1
```

即使 symlink 创建失败，`best.txt` 仍会记录当前 best 版本目录，例如 `iterv3`。

### Seed 代码路径无效

`seed_code_path` 必须指向存在的 `.cu` 文件。相对路径会按 spec 文件所在目录解析。

### Windows 控制台乱码

CLI 支持通过 `CONSOLE_ENCODING` 控制 stdout/stderr 编码，文件产物始终使用 UTF-8。

```dotenv
# 默认: Windows TTY 使用 GBK；IDE/CI/捕获输出使用 UTF-8
CONSOLE_ENCODING=auto

# Agent、IDE、CI 或日志捕获环境推荐
CONSOLE_ENCODING=utf-8

# 传统 cmd.exe 中文环境可选
CONSOLE_ENCODING=gbk

# 不修改 Python 默认 stream 编码
CONSOLE_ENCODING=default
```

## License

MIT
