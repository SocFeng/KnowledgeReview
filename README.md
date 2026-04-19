# KnowledgeReview · RAG 双实现对比项目

> 一个用 **同一份需求** 分别基于 **LlamaIndex** 与 **LangChain** 完整复刻一遍的
> RAG（Retrieval-Augmented Generation）项目。两套实现功能完全对齐，便于对比
> 两个主流框架在工程化、可控性、抽象代价上的差异，是学习 RAG 的好素材。

```
┌────────────────────────────────────────────────────────────────┐
│                          KnowledgeReview                       │
│                                                                │
│   ┌──────────────────────┐        ┌──────────────────────┐     │
│   │   llamaIndex_rag/    │   vs   │    langchain_rag/    │     │
│   │  （LlamaIndex 实现）  │        │   （LangChain 实现）   │     │
│   └──────────────────────┘        └──────────────────────┘     │
│             │                              │                   │
│             └─────────── 同一套 ───────────┘                    │
│                  Qwen + DashScope                              │
│                  ChromaDB + jieba BM25                         │
│                  gte-rerank-v2                                 │
│                  FastAPI + Streamlit                           │
└────────────────────────────────────────────────────────────────┘
```

---

## 1. 项目特点

✅ **同一份需求、两套实现**：可以直接对比两个框架在同一问题上的代码差异
✅ **功能完全对齐**：上传、检索、对话、引用、召回可视化、追问建议、会话管理…… 一个都不少
✅ **生产级细节**：去重、联动删除、流式 fallback、API key 全局补丁、可观测 trace
✅ **完整的学习材料**：每个项目都有 README（用法）+ LEARNING.md / docs（原理）
✅ **国产模型友好**：直接走阿里云百炼 DashScope（Qwen LLM、text-embedding、gte-rerank）

---

## 2. 共同的功能清单

下面这些能力两个子项目都实现了：

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

---

## 3. 两套实现的差异

> 功能 1:1，但因为框架抽象不同，代码长相和工程权衡有差异。

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

## 4. 仓库地图

```
KnowledgeReview/
│
├── README.md                        ← 本文（总览导航）
│
├── llamaIndex_rag/                  ← 实现 1：基于 LlamaIndex
│   ├── README.md                    用法说明 / 快速启动
│   ├── docs/项目学习指南.md          原理讲解 / 设计决策
│   ├── src/                         核心模块
│   ├── scripts/                     命令行入口
│   ├── streamlit_app.py             Web UI
│   ├── requirements.txt
│   └── .env.example
│
└── langchain_rag/                   ← 实现 2：基于 LangChain
    ├── README.md                    用法说明 / 快速启动
    ├── LEARNING.md                  ⭐ 学习路径文档（10 节、~700 行）
    ├── src/                         核心模块
    │   ├── config.py                配置中心
    │   ├── settings.py              LLM / Embedding 工厂
    │   ├── doc_store.py             文档级元数据 + nodes.json
    │   ├── ingest.py                摄入管线
    │   ├── retriever.py             混合检索 + RRF + Rerank
    │   ├── chat.py                  多轮对话
    │   ├── sessions.py              会话持久化
    │   └── api.py                   FastAPI 路由
    ├── scripts/                     命令行入口
    │   ├── ingest.py
    │   ├── run_api.py
    │   ├── ask_once.py
    │   └── diagnose.py              ⭐ 强烈推荐先跑这个体检
    ├── streamlit_app.py             Web UI
    ├── requirements.txt
    └── .env.example
```

---

## 5. 快速开始（任选其一）

> 两个子项目相互独立，各自有自己的虚拟环境、`.env`、`storage/`。

### 5.1 选择子项目

```powershell
# 选 LangChain 版（推荐学习用）
cd langchain_rag

# 或选 LlamaIndex 版
cd llamaIndex_rag
```

### 5.2 创建虚拟环境（Python 3.11 推荐）

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 5.3 配置 API Key

```powershell
copy .env.example .env
# 编辑 .env，填入你的 DASHSCOPE_API_KEY
```

> 在 [阿里云百炼](https://bailian.console.aliyun.com/) 控制台申请 DashScope API Key。

### 5.4 自检（仅 langchain_rag 提供）

```powershell
python scripts\diagnose.py
```

会逐项验证 配置 → Embedding → LLM → Rerank 是否可用。任何一项失败都会给出可操作的提示。

### 5.5 起 Web UI

```powershell
streamlit run streamlit_app.py
```

打开浏览器访问 `http://localhost:8501`：

1. **📤 文档管理** 页：上传 PDF/MD/TXT/DOCX → 解析入库 → 可点击文件名预览
2. **💬 知识问答** 页：基于已入库内容多轮对话，附引用、召回 trace、追问建议

### 5.6 起 REST API（可选）

```powershell
python -m scripts.run_api
```

打开 `http://localhost:8000/docs` 即可看到 Swagger 接口文档。

---

## 6. 推荐学习路径

如果你是来"学习 RAG 是怎么实现的"：

```
Step 1  先跑通 langchain_rag
        ├─ 按 §5 起 Streamlit
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

Step 4  做 LEARNING.md §8 的实战练习题
        从 Level 1（读懂）→ Level 4（深入）渐进
```

如果你只是想"用一个本地知识库"：

- 直接按 §5 跑 Web UI 即可，挑哪个版本都行

---

## 7. 技术栈一览

| 类别 | 选型 |
|---|---|
| 语言 | Python 3.11 |
| RAG 框架 | LlamaIndex 0.11+ / LangChain 0.3+（两个独立实现） |
| LLM 平台 | 阿里云百炼 DashScope |
| LLM 模型 | Qwen 系列（qwen-plus / qwen-max 等） |
| Embedding | DashScope `text-embedding-v3` 或 HuggingFace BGE |
| Rerank | DashScope `gte-rerank-v2` |
| 向量库 | ChromaDB |
| 关键词检索 | rank-bm25 + jieba |
| API | FastAPI + Uvicorn |
| Web UI | Streamlit |
| 配置管理 | pydantic-settings |
| 文档加载 | pypdf / python-docx / docx2txt |

---

## 8. 项目意图

这个仓库的目标**不是**"再造一个开源 RAG 工具"——市面上 LangChain、LlamaIndex 自己的官方 examples 已经够多。

它的目标是：

1. **作为学习样本**：帮助理解一个真正能用的 RAG 系统由哪些模块组成、各模块边界在哪
2. **作为框架对比**：在同一份需求下看两个主流框架的工程取舍
3. **作为可二次开发的脚手架**：所有"生产级细节"都已踩过坑、代码里有注释说明，可以直接 fork 改成你自己的内部知识库

---

## 9. 常见问题

**Q: 两个子项目的数据/索引能共用吗？**
A: 不能。它们各自维护 `storage/chroma/` 和 `data/`，相互独立。即便切到同一个 embedding 模型，存储格式细节也有差异。

**Q: 必须用阿里云百炼吗？**
A: 当前是。LLM 部分换成 OpenAI / Claude 改动量很小（替换 `_build_llm` 工厂即可）；Embedding 已经支持本地 HuggingFace BGE；Rerank 唯一依赖 DashScope，可以在 `.env` 里把 `RERANK_MODEL` 设成空字符串绕过（已有降级逻辑）。

**Q: Windows 下能跑吗？**
A: 能，本项目就是在 Windows + PowerShell 下开发测试的。本 README 的命令示例都是 PowerShell 语法。

**Q: 切换 embedding 模型后召回质量很差？**
A: 不同 embedding 模型生成的向量在不同语义空间，必须**清空 `storage/chroma/` 重新 ingest**。直接复用旧索引会得到一堆噪音。

**Q: 上传同一份文件没反应？**
A: 是按内容 hash 去重的有意行为；改名上传也会被识别为同一文件。如果想强制重入，删掉 `storage/` 重来即可。

---

## 10. 许可

本项目仅作学习用途，欢迎 fork、改造、用于内部知识库搭建。
依赖的第三方库（LangChain / LlamaIndex / DashScope 等）请遵循它们各自的许可协议。
