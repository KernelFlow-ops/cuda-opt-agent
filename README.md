# CUDA 算子优化智能体

> LLM 自主驱动的 CUDA 算子迭代优化 Agent。
> 用户只需说一句"帮我优化 GEMM"，Agent 自动完成 **调研 → 生成 v0 → Profile → 分析瓶颈 → 选优化方法 → 生成 vN → 验证 → 接受/回退 → 沉淀经验** 的全流程。

## 快速开始

### 1. 环境准备

```bash
# 系统要求
# - Python >= 3.10
# - CUDA Toolkit (nvcc)
# - Nsight Compute (ncu)
# - NVIDIA GPU

# 安装
pip install -e ".[dev]"

# 配置
cp .env.example .env
# 编辑 .env，填入 ANTHROPIC_API_KEY 或 OPENAI_API_KEY
```

### 2. 运行优化

```bash
# 新建优化运行
cuda-opt run gemm --shape 4096,4096,4096 --dtype fp16 --max-iters 30

# 续跑（自动找最近的同名未完成 run）
cuda-opt resume gemm

# 续跑（指定目录）
cuda-opt resume --run-dir runs/gemm_run_20260501T120000

# 续跑并增加迭代数
cuda-opt resume gemm --extra-iters 20

# 查看所有运行
cuda-opt list-runs

# 查看运行详情
cuda-opt show-run runs/gemm_run_20260501T120000
```

### 3. 运行测试

```bash
# 运行所有不依赖 GPU/API 的测试
pytest tests/ -v -m "not api and not gpu"

# 运行 API 连通性测试（需要 API key）
pytest tests/ -v -m api

# 运行带覆盖率报告的测试
pytest tests/ -v --cov=cuda_opt_agent --cov-report=html
```

## 项目结构

```
src/cuda_opt_agent/
├── cli.py                 # CLI 入口 (Typer)
├── config.py              # 配置加载 (.env)
├── agent/                 # Agent 核心
│   ├── graph.py           # LangGraph 状态机
│   ├── nodes.py           # 所有节点实现
│   ├── state.py           # GraphState 定义
│   ├── llm_client.py      # LLM 调用封装
│   └── prompts/           # 8 个 Prompt 模板
├── tools/                 # 工具层
│   ├── compile.py         # nvcc 编译
│   ├── profile.py         # ncu Profiling
│   ├── benchmark.py       # cudaEvent 测速
│   ├── correctness.py     # 数值正确性校验
│   ├── hardware.py        # GPU 硬件信息采集
│   └── web_search.py      # (可选) Web 检索
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
| 过程透明 | 每次迭代的代码、ncu 报告、推理过程 100% 落盘 |
| 控制变量 | 每次只引入一个改动，便于归因 |
| 断点续跑 | Ctrl+C 后可无缝续跑 |
| 经验沉淀 | 跨运行知识库，以软提示形式注入 |

## 技术栈

- **Agent 框架**: LangGraph + LangChain
- **LLM**: Anthropic Claude / OpenAI GPT-4o（可切换）
- **数据模型**: Pydantic v2
- **CLI**: Typer
- **TUI**: Rich + Textual
- **配置**: python-dotenv
- **CUDA 工具**: nvcc, ncu (Nsight Compute), nvidia-smi

## License

MIT
