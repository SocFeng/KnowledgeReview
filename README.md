# KnowledgeReview · 大模型应用实战合集

> 三个独立可跑、互相参考的 **LLM 应用样板**，覆盖当下两大主流范式：
> **RAG**（检索增强生成，"问知识库"）与 **Agent**（工具调用，"调外部世界"）。
>
> 全部基于阿里云百炼 **Qwen / DashScope**，全部在 **Windows + Python 3.11** 下踩过坑，
> 每个子项目都附 `README.md`（怎么用）+ `LEARNING.md`（为什么这样设计）。
>
> 适合作为：**学习样本** / **框架对比** / **可二次开发的脚手架**。

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            KnowledgeReview                              │
│                                                                         │
│   ┌────────────────────┐  ┌────────────────────┐  ┌──────────────────┐  │
│   │  llamaIndex_rag/   │  │   langchain_rag/   │  │ langChain_       │  │
│   │  RAG · LlamaIndex  │  │  RAG · LangChain   │  │ langGraph_agent/ │  │
│   │                    │  │                    │  │ Agent · LangGraph│  │
│   └─────────┬──────────┘  └─────────┬──────────┘  └────────┬─────────┘  │
│             │ 同一份需求 · 两套实现 │                       │            │
│             └──────── RAG ──────────┘                       │            │
│                                                             ↓            │
│                                                       Agent + Tools      │
│                                                                         │
│              共同底座：Qwen via DashScope · Streamlit UI                │
│              RAG 专用：ChromaDB · jieba BM25 · gte-rerank-v2            │
│              Agent 专用：LangGraph StateGraph · 自定义 Tools            │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 1. 项目特点

✅ **覆盖两大 LLM 应用范式**：RAG（封闭世界问答）+ Agent（开放世界工具调用）
✅ **同一份需求、两套 RAG 实现**：直接对比 LangChain vs LlamaIndex 两个主流框架的工程取舍
✅ **生产级细节**：去重、联动删除、流式 fallback、API key 全局补丁、checkpointer 持久化、可观测 trace …… 全都已踩过坑
✅ **完整的学习材料**：每个项目都有用法 README + 深度 LEARNING.md（含实战练习题）
✅ **国产模型友好**：直接走阿里云百炼 DashScope（Qwen LLM、text-embedding、gte-rerank）；Embedding 可换本地 HuggingFace BGE
✅ **零隐藏依赖**：Agent 项目优先免费无 key 数据源（Open-Meteo / Wikipedia / OSM），高德 Key 可选

---

## 2. 三个子项目一览

| 子项目 | 范式 | 框架 | 核心能力 | 适合谁读 |
|---|---|---|---|---|
| `llamaIndex_rag/` | **RAG** | LlamaIndex 0.11+ | 文档问答 / 多轮对话 / 混合检索 + Rerank | 想快速搭原型、喜欢"高抽象、少代码"的开发者 |
| `langchain_rag/` | **RAG** | LangChain 0.3+ | 同上，功能 1:1 对齐 | 想深入理解 RAG 内部、喜欢"低抽象、全可控"的开发者 |
| `langChain_langGraph_agent/` | **Agent** | LangChain + LangGraph | 智能旅游规划：自动调用天气 / 地理 / 景点 / 文化 / 路线工具 | 想学会写 Agent、ReAct loop、Tool calling 的开发者 |

> 三个项目**完全独立**：各自的 venv、`.env`、`storage/` / `data/` 都不共享。
> 想看哪个直接 `cd` 进去即可。

---

## 3. RAG 子项目能力清单

`llamaIndex_rag/` 和 `langchain_rag/` 两个 RAG 项目功能完全对齐：

| 类别 | 能力 |
|---|---|
| 📚 文档格式 | PDF / Markdown / TXT / DOCX |
| 🧠 LLM | Qwen via DashScope（`qwen-plus` / `qwen-max` 等可配） |
| 🔢 Embedding | DashScope `text-embedding-v3` 或 本地 HuggingFace BGE，`.env` 切换 |
| 🗄️ 向量库 | ChromaDB 本地持久化 |
| 🔍 检索 | 向量检索 + BM25（jieba 中文分词）+ RRF 融合 |
| 🎯 Rerank | DashScope `gte-rerank-v2` 精排（失败自动降级） |
| 💬 多轮对话 | 自动浓缩历史，独立检索问句改写 |
| 🔗 引用溯源 | 答案中 `[n]` 可点击跳转到对应来源片段 |
| 📊 召回可视化 | vector / BM25 / 融合 / rerank 四阶段命中对比 |
| ✨ 追问建议 | 每轮回答后自动生成 3 个相关追问按钮 |
| 🎛️ 回答风格 | 简洁 / 详细 / 对比表格 / 步骤化 四种切换 |
| 🗂️ 文档管理 | 按内容 hash 去重；删原始文件联动清理向量库；文件预览 |
| 📌 会话管理 | 搜索 / 重命名 / 固定置顶 / 导入导出（JSON & Markdown） |
| 🌐 REST API | FastAPI，自带 `/docs` Swagger 页面 |
| 🖥️ Web UI | Streamlit，上传 → 解析 → 问答 全流程 |

### 两套 RAG 实现的差异

| 维度 | `llamaIndex_rag/` | `langchain_rag/` |
|---|---|---|
| 检索流程编排 | LlamaIndex 内置 `QueryFusionRetriever` + `RetrieverQueryEngine` | 手工组合 `_ChromaVectorRetriever` + `_BM25Retriever` + 自实现 RRF |
| 多轮对话 | 内置 `CondensePlusContextChatEngine` | 自实现"浓缩 + 上下文" 两步流程 |
| BM25 数据来源 | `SimpleDocumentStore`（LlamaIndex 内置） | 自维护 `nodes.json`，独立 JSON 快照 |
| LLM / Embedding 接入 | 走 `llama-index-llms-dashscope` / `llama-index-embeddings-dashscope` | 自实现 `_DashScopeSDKEmbeddings`，绕开 `langchain-community` 的 401 坑 |
| 全局配置 | `Settings.llm = ...` 全局单例（LlamaIndex 风格） | 工厂函数 + 显式 `get_llm()` / `get_embedding()` |
| Chroma 调用 | 走 `llama-index-vector-stores-chroma` 包装 | 直接用原生 `chromadb.PersistentClient` |
| 抽象层数 | 高，少量代码胶合多个内置组件 | 低，多写胶水换全程可控 |

**简单结论**：

- **想快速搭原型** → 看 `llamaIndex_rag/`，行数少、抽象高
- **想深入了解 RAG 内部机制 / 想对每一步都可控** → 看 `langchain_rag/`，几乎每行代码都能讲清"为什么"

---

## 4. Agent 子项目（`langChain_langGraph_agent/`）

> 用 **LangChain + LangGraph + 自定义工具** 实现的中文**智能旅游规划助手**。
> 用户用聊天对话告诉 Agent 出发地、目的地、出行天数、偏好，
> Agent 会自动调用天气 / 地理 / 景点 / 文化 / 路线等工具，
> 整合成可随时修改的 Markdown 行程方案。

### 4.1 能力清单

| 类别 | 能力 |
|---|---|
| 🤖 Agent 框架 | **LangGraph StateGraph** 手写 ReAct loop（不是 prebuilt 的 `create_react_agent` 黑盒） |
| 🛠 自定义工具 | 6 个**全部自实现**：地理编码 / 天气 / 距离 / 景点 / 文化 / 路线 |
| 🌐 数据源策略 | 优先**免费无 key**（Open-Meteo / Wikipedia / OSM），可选高德地图增强 |
| 💬 类聊天 UI | Streamlit 多会话切换、重命名、删除 |
| ⌨️ 真·token 流式 | 前端**打字机效果**：双 `stream_mode=["messages", "updates"]`，首 token ~2s 抵达 |
| 💰 Token / 成本统计 | 按轮次 / 按会话累计 `input_tokens` / `output_tokens` / 人民币费用，侧边栏实时显示 |
| 🔌 LLM Provider 可切换 | 默认 **DashScope OpenAI 兼容端点 + `ChatOpenAI`**（流式 + tool_calls 稳定），可切回 `ChatTongyi` |
| 🧠 持久化记忆 | LangGraph **SqliteSaver checkpointer**，跨进程也能续聊 |
| 🔧 可观测 | 工具调用 trace 实时展开（参数 + 返回 JSON） |
| 🔁 上下文修改 | 用户随时改需求（"再加一天"、"去掉博物馆"），Agent 基于历史增量调整 |
| 🇨🇳 中文 LLM | DashScope Qwen 系列（`qwen-plus` / `qwen-max` / ...） |

### 4.2 核心知识点（`LEARNING.md` 已详解）

- **Agent vs Chain 的本质区别**：流程是写死还是 LLM 决策
- **ReAct 框架**：Reason → Act → Observe 循环，靠 tool calling 协议落地
- **LangGraph StateGraph**：节点 / 条件边 / cycle，区别于 LangChain 普通 DAG
- **`@tool` 装饰器**：函数 → JSON Schema → LLM 可见
- **`bind_tools()` 跨 Provider 兼容**：同一套业务代码配 `ChatOpenAI` / `ChatTongyi` / Claude 都能跑
- **双 `stream_mode` 流式**：`["messages", "updates"]` 一次拿 token 增量 + 节点级工具 trace
- **Token/成本聚合**：按 `request_id` 去重 + 兼容 `usage_metadata` / `response_metadata.token_usage` 两种上报渠道
- **State + add_messages reducer**：多节点写同字段时合并而非覆盖
- **Checkpointer 持久化**：thread_id 隔离会话、跨进程续聊
- **Tool 失败哲学**：永不抛异常，返回 `{"error": "..."}` 让 LLM 自行换路
- **多源 fallback 排序**：怎么避免"北京"被 Open-Meteo 解析到重庆某村镇

---

## 5. 仓库地图

```
KnowledgeReview/
│
├── README.md                            ← 本文（总览导航）
│
├── llamaIndex_rag/                      ← 实现 1：RAG · LlamaIndex
│   ├── README.md                        用法说明 / 快速启动
│   ├── docs/项目学习指南.md              原理讲解 / 设计决策
│   ├── src/                             核心模块
│   ├── scripts/                         命令行入口
│   ├── streamlit_app.py                 Web UI
│   ├── requirements.txt
│   └── .env.example
│
├── langchain_rag/                       ← 实现 2：RAG · LangChain
│   ├── README.md                        用法说明 / 快速启动
│   ├── LEARNING.md                      ⭐ 学习路径文档（10 节、~700 行）
│   ├── src/                             核心模块（config / settings / doc_store / ingest / retriever / chat / sessions / api）
│   ├── scripts/                         命令行入口（含 diagnose 体检脚本）
│   ├── streamlit_app.py                 Web UI
│   ├── requirements.txt
│   └── .env.example
│
└── langChain_langGraph_agent/           ← 实现 3：Agent · LangChain + LangGraph
    ├── README.md                        用法说明 / 快速启动
    ├── LEARNING.md                      ⭐ 学习路径文档（12 节、~1100 行，标注全部 Agent 知识点）
    ├── src/
    │   ├── config.py                    配置中心（支持 LLM_PROVIDER 切换）
    │   ├── llm.py                       LLM 工厂（ChatOpenAI 兼容模式 / ChatTongyi 双轨）
    │   ├── prompts.py                   System Prompt（含工作流约束）
    │   ├── state.py                     LangGraph State + add_messages
    │   ├── usage.py                     ⭐ Token 用量 + 成本估算 + 按 thread 持久化
    │   ├── agent.py                     ⭐⭐ StateGraph + 手写 ReAct + 双 stream_mode + checkpointer
    │   └── tools/                       6 个自定义工具
    │       ├── geocode.py / weather.py / distance.py
    │       └── attractions.py / culture.py / route.py
    ├── scripts/
    │   ├── diagnose.py                  ⭐ 一键体检：配置 → LLM → 6 个工具
    │   └── chat_cli.py                  命令行版对话
    ├── streamlit_app.py                 Web UI（多会话 + 工具 trace）
    ├── requirements.txt
    └── .env.example
```

---

## 6. 快速开始

> 三个子项目相互独立，各自有自己的虚拟环境、`.env`、数据目录。

### 6.1 选择子项目

```powershell
# 选 RAG · LangChain（推荐学习 RAG 用）
cd langchain_rag

# 或选 RAG · LlamaIndex
cd llamaIndex_rag

# 或选 Agent · LangGraph（推荐学习 Agent 用）
cd langChain_langGraph_agent
```

### 6.2 创建虚拟环境（Python 3.11 推荐）

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 6.3 配置 API Key

```powershell
copy .env.example .env
# 编辑 .env，填入你的 DASHSCOPE_API_KEY
```

> 在 [阿里云百炼](https://bailian.console.aliyun.com/) 控制台申请 DashScope API Key。
> Agent 项目还可选填 [高德地图](https://console.amap.com/dev/key/app) Web 服务 Key（不填也能跑，会 fallback 到免费数据源）。

### 6.4 自检（langchain_rag 和 Agent 项目都有）

```powershell
python -m scripts.diagnose
```

会逐项验证 配置 → LLM → Embedding/Rerank（RAG）或 6 个工具（Agent）是否可用。任何一项失败都给出可操作的提示。

### 6.5 起 Web UI

```powershell
streamlit run streamlit_app.py
```

打开浏览器 `http://localhost:8501`：

- **RAG 项目**：📤 文档管理 → 上传 PDF/MD/TXT/DOCX；💬 知识问答 → 基于已入库内容多轮对话
- **Agent 项目**：💬 直接聊天，告诉它出发地/目的地/天数/偏好，看它自动调工具规划行程

### 6.6 起 REST API（仅 RAG 项目，可选）

```powershell
python -m scripts.run_api
```

打开 `http://localhost:8000/docs` 即可看到 Swagger 接口文档。

---

## 7. 推荐学习路径

### 路径 A：想学懂 RAG

```
Step 1  先跑通 langchain_rag
        ├─ 按 §6 起 Streamlit
        ├─ 上传一份你熟悉的资料
        └─ 提几个问题，观察"召回详情"面板里 4 阶段的命中变化

Step 2  阅读 langchain_rag/LEARNING.md
        ├─ §3 第一性原理：什么是 RAG
        ├─ §4 模块走读（按依赖顺序逐个文件读）
        └─ §6 关键概念深入

Step 3  对比 llamaIndex_rag 的同名模块
        ├─ 同样的功能在 LlamaIndex 里怎么写？
        ├─ 哪些地方变短了？哪些地方"看不见"了？
        └─ 哪种你更喜欢？为什么？

Step 4  做 LEARNING.md §8 的实战练习题（Level 1 → Level 4）
```

### 路径 B：想学懂 Agent

```
Step 1  先跑通 langChain_langGraph_agent
        ├─ 按 §6 起 Streamlit
        ├─ 输入"我想周末从北京去天津玩 1 天，喜欢历史"
        └─ 观察工具 trace，看 Agent 怎么自己决定调哪些工具

Step 2  阅读 langChain_langGraph_agent/LEARNING.md
        ├─ §3 从 Chain 到 Agent + ReAct 框架
        ├─ §4 模块走读：尤其是 agent.py 的图结构
        └─ §5 Tool Calling 协议深入 + §6 记忆机制

Step 3  做 LEARNING.md §9 实战练习
        ├─ Level 1：改 should_continue / state / prompt 看会发生什么
        ├─ Level 2：自己加一个 currency 工具
        └─ Level 4：尝试 Agent + RAG 融合
```

### 路径 C：想全面掌握 LLM 工程

```
RAG（路径 A） → Agent（路径 B） → 阅读 Agent LEARNING.md §10 与 RAG 的对比

最后挑战 §9 Level 4：把 Agent 的"文化背景"工具改成调 langchain_rag 项目的检索器
        → 这就是真实生产里 Agent + RAG 的标准融合方式
```

### 路径 D：只想用，不想学

直接按 §6 跑任意一个 Web UI 即可。三个 Streamlit 都开箱可用。

---

## 8. 技术栈一览

| 类别 | RAG 项目选型 | Agent 项目选型 |
|---|---|---|
| 语言 | Python 3.11 | Python 3.11 |
| 主框架 | LlamaIndex 0.11+ / LangChain 0.3+ | LangChain 0.3+ + **LangGraph 0.2+** |
| LLM 平台 | 阿里云百炼 DashScope | 同左 |
| LLM 模型 | Qwen 系列（qwen-plus / qwen-max） | 同左 |
| Embedding | DashScope `text-embedding-v3` 或 HuggingFace BGE | — |
| Rerank | DashScope `gte-rerank-v2` | — |
| 向量库 | ChromaDB | — |
| 关键词检索 | rank-bm25 + jieba | — |
| 工具数据源 | — | Open-Meteo / Wikipedia / OSM Nominatim / 高德（可选） |
| 状态 / 记忆 | 自维护 chat_history JSON | **LangGraph SqliteSaver checkpointer** |
| API | FastAPI + Uvicorn | — |
| Web UI | Streamlit | Streamlit |
| 配置管理 | pydantic-settings | pydantic-settings |
| 文档加载 | pypdf / python-docx / docx2txt | — |

---

## 9. 项目意图

这个仓库的目标**不是**"再造一套开源工具"——LangChain、LlamaIndex、LangGraph 自己的官方 examples 已经够多。

它的目标是：

1. **作为学习样本**：帮助理解一个真正能用的 RAG / Agent 系统由哪些模块组成、各模块边界在哪
2. **作为框架对比**：在同一份需求下看 LangChain vs LlamaIndex 的工程取舍；在 RAG vs Agent 的世界观差异中体会"封闭世界 vs 开放世界"
3. **作为可二次开发的脚手架**：所有"生产级细节"都已踩过坑、代码里有注释说明，可以直接 fork 改成你自己的内部应用：
   - RAG → 公司内部知识库 / 法律咨询助手 / 产品手册问答
   - Agent → 智能客服 / 报销助手 / 数据分析 bot / 代码评审

---

## 10. 常见问题

**Q: RAG 和 Agent 有什么区别？我应该用哪个？**
A: RAG 适合**封闭世界**——答案在你给的资料里就能找到（公司文档、产品手册）；Agent 适合**开放世界**——答案需要"实时调外部世界"才能拿到（天气、汇率、订单状态）。**生产里复杂应用通常是 Agent 嵌套调 RAG**：Agent 把检索器当成它的工具之一。

**Q: 三个子项目的数据/索引能共用吗？**
A: 不能。它们各自维护 `storage/` / `data/`，相互独立。

**Q: 必须用阿里云百炼吗？**
A: 当前是。LLM 部分换成 OpenAI / Claude 改动量很小（每个子项目都有 `llm.py` 或 `settings.py` 工厂函数）；Embedding 已支持本地 HuggingFace BGE；Rerank 唯一依赖 DashScope，可在 `.env` 里把 `RERANK_MODEL` 设空字符串绕过（已有降级逻辑）。

**Q: Agent 项目不申请高德 Key 能用吗？**
A: 能。地理编码、景点、路线都有免费数据源 fallback（Open-Meteo / Wikipedia / OSM），质量稍差但完整可用。

**Q: Windows 下能跑吗？**
A: 能，本仓库就是在 Windows + PowerShell 下开发测试的。所有命令示例都是 PowerShell 语法。

**Q: 切换 embedding 模型后 RAG 召回质量很差？**
A: 不同 embedding 模型生成的向量在不同语义空间，必须**清空 `storage/chroma/` 重新 ingest**。直接复用旧索引会得到一堆噪音。

**Q: Agent 项目偶尔超时？**
A: DashScope 在调用大上下文（带 6 个工具 schema）时偶发 read timeout；切换到 `qwen-turbo` 更快。项目启用了真·token 流式，首字 ~2s 即可抵达，整体体感不卡。

**Q: 想看本次对话花了多少 token？**
A: Agent 项目的 Streamlit 侧边栏有"📊 Token & 费用"面板，实时显示 ↑ 输入 / ↓ 输出 / 人民币费用的累计值和逐轮明细。原始数据在 `langChain_langGraph_agent/data/usage/{thread_id}.json`。

**Q: 为什么 Agent 项目默认不直接用 `ChatTongyi`？**
A: `langchain_community.ChatTongyi` 在 `streaming=True` + `tool_calls` 场景下有上游 bug（`subtract_client_response` 抛 `IndexError`）。本项目默认改用 **DashScope 的 OpenAI 兼容端点 + `ChatOpenAI`**，后端依然是 Qwen，但流式 + 工具调用稳定得多，并且能走 LangChain 标准的 `usage_metadata` 统计 token。需要切回 ChatTongyi 做对照时，把 `.env` 里 `LLM_PROVIDER=tongyi` 即可。

---

## 11. 许可

本仓库仅作学习用途，欢迎 fork、改造、用于内部项目搭建。
依赖的第三方库（LangChain / LlamaIndex / LangGraph / DashScope 等）请遵循它们各自的许可协议。

---

> 🌟 如果这个仓库对你有帮助，欢迎 Star。
> 🐛 发现 bug 或想加新功能（比如把 OpenAI / Claude 接进去），欢迎提 Issue / PR。
