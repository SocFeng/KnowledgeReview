# LangChain RAG 学习指南

> 这是一份**学习路径文档**，目标读者是想把这个项目读懂、改造、二次开发的工程师。
> 它不是 README 的复述，而是按"为什么这样写"的顺序，把整个项目拆开讲一遍。
>
> 阅读建议：先把整体地图过一遍（第 2 节），再按推荐顺序逐模块走读（第 4 节）。
> 每读完一个模块，去做一次配套的"实战练习"（第 8 节）。

---

## 1. 学习目标

读完本文档你应该能回答以下问题：

1. RAG 的本质是什么？为什么需要"检索 + 生成"两阶段？
2. 一段 PDF 文本是怎么从磁盘走到 LLM 的 prompt 里去的？经历了哪些处理？
3. 为什么仅用向量检索不够，要再叠加 BM25 + Rerank？这三者各自补什么短板？
4. RRF（Reciprocal Rank Fusion）为什么这么简单却好用？
5. 多轮对话里，"历史浓缩"（condense）和"上下文注入"（context）分别在解决什么问题？
6. 删除一个原始文件时，向量库里的残留怎么联动清理？为什么需要 `file_hash`？
7. 为什么本项目几乎不用 LangChain 的高级抽象（`EnsembleRetriever`、`Chroma` 包装等），而直接调原生 SDK？

如果上面任何一个问题答不出来，就值得在对应章节多停留一会。

---

## 2. 项目地图

```
langchain_rag/
├── .env                  # 实际配置（API Key、模型名、参数）
├── .env.example          # 模板
├── .streamlit/config.toml
├── requirements.txt
├── README.md             # 给"使用者"看的：怎么跑起来
├── LEARNING.md           # 本文，给"学习者"看的：为什么这样设计
│
├── data/
│   ├── uploads/          # Streamlit 上传的原始文件
│   └── chat_history/     # 持久化的会话记录（每个会话一个 .json）
│
├── storage/
│   ├── chroma/           # ChromaDB 持久化目录（向量索引）
│   └── docstore/
│       └── nodes.json    # 节点快照，BM25 重启时直接复活
│
├── src/                  # ⭐ 核心代码全在这里
│   ├── config.py         # 1) 配置中心：pydantic-settings 读 .env
│   ├── settings.py       # 2) LLM/Embedding/TextSplitter 工厂 + 单例
│   ├── doc_store.py      # 3) 文档级元数据：去重、联动删除、节点快照
│   ├── ingest.py         # 4) 摄入管线：load → split → embed → write
│   ├── retriever.py      # 5) 混合检索 + RRF + DashScope rerank + trace
│   ├── chat.py           # 6) 多轮对话：浓缩 + 检索 + 风格 + 追问
│   ├── sessions.py       # 7) 会话持久化（JSON 格式）
│   └── api.py            # 8) FastAPI 路由
│
├── scripts/              # 命令行入口
│   ├── ingest.py         # 全量 ingest data/ 目录
│   ├── run_api.py        # 起 uvicorn
│   ├── ask_once.py       # 单次问答（调试用）
│   └── diagnose.py       # 自检 LLM / Embedding / Rerank 是否可用
│
└── streamlit_app.py      # Web UI（文档管理 + 知识问答两个 tab）
```

**推荐阅读顺序**（也是各文件功能的依赖顺序）：

```
config.py → settings.py → doc_store.py → ingest.py
                                          ↓
                  retriever.py ← 复用 doc_store/nodes.json
                                          ↓
                                       chat.py
                                          ↓
                          sessions.py + api.py + streamlit_app.py
```

---

## 3. 第一性原理：什么是 RAG

### 3.1 LLM 的两个根本性短板

1. **知识截止**：模型训练完就定格了，新发生的、私有的、内部的知识它根本不知道。
2. **幻觉**：当问到未学过的事时，LLM 会"编"，编得还像模像样。

**RAG（Retrieval-Augmented Generation）的核心想法**：把"知道什么"和"会说话"解耦——
让 LLM 只负责把别人给它的资料加工成自然语言答案，不负责"记住事实"。

### 3.2 一次 RAG 问答的完整链条

```
用户问题
   ↓
[检索 Retrieval]  → 在你的知识库里找出最相关的 N 段
   ↓
[拼 Prompt]       → "你只能用下面这些资料回答……" + 资料 + 问题
   ↓
[LLM 生成]        → 自然语言答案 + 引用编号 [1][2]
```

实现一个 RAG 系统的关键问题就是：**"怎么把最相关的资料找出来"**。这就是为什么本项目大半的代码都在围绕"检索"打转（`ingest.py` 准备数据、`retriever.py` 做检索）。

### 3.3 本项目的技术选型

| 环节 | 选型 | 原因 |
|---|---|---|
| 切片 | `RecursiveCharacterTextSplitter` + 中文分隔符 | 控制粒度、对中文友好 |
| Embedding | DashScope `text-embedding-v3` 或本地 BGE | 在线快，本地省钱 |
| 向量库 | ChromaDB | 本地持久化、零运维 |
| 关键词检索 | BM25 (`rank_bm25`) + jieba | 补 embedding 召不全的实体词 |
| 融合 | Reciprocal Rank Fusion | 简单、无需调参 |
| 精排 | DashScope `gte-rerank-v2` | 让最相关的几条排到最前 |
| LLM | Qwen via DashScope (`ChatTongyi`) | 中文能力强、支持流式 |
| 多轮 | 自实现"浓缩 + 上下文" | 等价 LlamaIndex 的 `CondensePlusContextChatEngine` |

---

## 4. 模块走读

### 4.1 `config.py`：配置中心

**用什么**：`pydantic_settings.BaseSettings`，自动读 `.env` 并做类型校验。

**为什么用 pydantic 而不是 `os.getenv`**：

- 类型转换免费：`CHUNK_SIZE=512` 会自动变成 `int`
- 缺字段直接报错（`dashscope_api_key: str = Field(...)` 没传就启动失败）
- 配合 IDE 有补全和类型提示，比 `cfg["chunk_size"]` 这种字典访问安全得多

**派生路径用 `@property`**：

```python
@property
def chroma_dir(self) -> Path:
    return self.storage_dir / "chroma"
```

这样所有路径都从 `storage_dir` 派生，用户只配一个根目录即可。

**单例 `get_settings()`**：第一次调用时还会顺手把所有目录 `mkdir`，避免运行时 `FileNotFoundError`。

> 🧠 **学习要点**：所有"运行时不该改的状态"都集中在 `config.py`，下游模块不读 `.env`，只 `from .config import get_settings`。这是单一职责。

---

### 4.2 `settings.py`：LLM / Embedding 工厂

这个文件做了三件事：
1. 工厂函数：`_build_llm` / `_build_embedding` / 创建 `RecursiveCharacterTextSplitter`
2. 单例缓存：`get_llm()` / `get_embedding()` / `get_text_splitter()`
3. 一个看似不起眼但救命的全局副作用：`_ensure_dashscope_global_key()`

#### 4.2.1 为什么要把 LLM 和 Embedding 包成工厂

不同人跑这个项目，可能用的是 DashScope 在线 embedding，也可能用本地 BGE 模型。下游 `retriever.py` 不应该 care 你用的是哪种——它只调 `embed.embed_query(text)`。所以用工厂模式：

- `_build_dashscope_embedding()` 返回我们自实现的 `_DashScopeSDKEmbeddings`
- `_build_hf_embedding()` 返回 `HuggingFaceEmbeddings`，并在外层加一层 `_QueryInstructionEmbeddings` wrapper（给 BGE/E5 系列查询加固定前缀）

下游永远只看到 `Embeddings` 这个抽象基类。

#### 4.2.2 为什么不用 `langchain_community.DashScopeEmbeddings`

血泪经验：在某些版本下，那个 wrapper 不会把构造器传入的 `dashscope_api_key` 写回 `dashscope` SDK 的全局变量，导致底层 HTTP 请求 401。我们直接调 `dashscope.TextEmbedding.call(api_key=...)` 完全绕开这个坑：

```5:5:src/settings.py
# 见 _DashScopeSDKEmbeddings._call
```

而且我们的实现比 wrapper 多了：
- 显式区分 `text_type=document` / `text_type=query`（DashScope 对此敏感，不区分会拉低召回质量）
- 自动按 16 条一批切分，超长输入不会越界

#### 4.2.3 `_ensure_dashscope_global_key` 是什么坑

LLM (`ChatTongyi`)、Embedding、Rerank 三者底层都依赖 `dashscope.api_key` 这个全局变量。某些 wrapper 会"忘记下发"，导致明明设了 key 还是 401。这个函数在 `init_settings()` 里做兜底：

```python
dashscope.api_key = api_key
os.environ["DASHSCOPE_API_KEY"] = api_key
```

> 🧠 **学习要点**：第三方 SDK 经常用全局变量传配置（不是好实践，但事实如此）。封装第三方时，**把全局副作用集中到一个明显的位置**比埋进各个 builder 更可控。

---

### 4.3 `doc_store.py`：文档级元数据管理

#### 4.3.1 为什么需要这一层？

ChromaDB 是按 chunk（节点）粒度组织的：每条记录是一个文本片段 + 它的 embedding + metadata。但用户面对的是**文件**，不是 chunk。

所以中间需要一层抽象，能回答这些问题：
- "这个文件之前传过吗？" → 按 `file_hash` 查
- "这个文件占了多少向量节点？" → 按 `file_hash` 聚合 `count`
- "我把这个文件删了，向量库里的残留怎么办？" → 按 `file_hash` 批量 delete

`DocStore` 类就是干这个的。它本质上是 ChromaDB collection 之上的一层"按 file_hash 聚合的视图"。

#### 4.3.2 `file_hash` 为什么用 SHA-1 而不是 SHA-256

```python
def file_sha1(path: Path, block_size: int = 1 << 20) -> str:
```

- SHA-1 比 SHA-256 快约 30%
- 我们要的是"区分文件是否相同"，不是密码学安全
- 碰撞概率对单用户知识库完全可以忽略
- 便宜、足够、不阻塞主流程

#### 4.3.3 `NodeSnapshotStore` 为什么单独存 `nodes.json`

这是本项目最值得学习的一个设计。

**问题**：BM25 是 in-memory 算法，需要拿到所有节点的原始文本才能建索引。
但 ChromaDB 的设计目标是"快速 KNN 查询"，全量拉所有 chunk 文本回来重建 BM25 既慢又浪费。

**方案**：在 ingest 时，每写一条节点到 Chroma，**同时**把 `{id, text, metadata}` 三元组追加到一个独立的 JSON 文件 `storage/docstore/nodes.json`。BM25 重启时直接 `json.load()` 这个文件，毫秒级复活。

这个文件本质上是 ChromaDB 文档内容的**冗余备份**，但它换来了：
- BM25 启动时间从"几秒拉数据 + 几秒分词"降到"毫秒级 load"
- 即便切换 embedding provider（重新生成向量），BM25 索引也不需要重建

#### 4.3.4 写 JSON 用 `tmp -> replace` 模式

```python
tmp = self.file_path.with_suffix(".tmp")
tmp.write_text(json.dumps(...))
tmp.replace(self.file_path)
```

避免半写入造成 JSON 损坏。这是写文件时的通用最佳实践（POSIX 系统下 `replace` 是原子的）。

> 🧠 **学习要点**：当一个数据需要在两套系统中保持一致（这里是 Chroma + nodes.json），要么用事务（重），要么承担最终一致性 + 提供"清孤儿"的兜底命令（`streamlit_app.py` 里的 🧹 按钮）。

---

### 4.4 `ingest.py`：摄入管线

这是一个经典的 ETL 管线，按下面四步走：

```
[加载文件] → [按 hash 去重] → [切片 + 注入元数据] → [embed + 写两个存储]
```

#### 4.4.1 加载：`_load_one()` 按后缀分发

```python
if suffix == ".pdf":
    from langchain_community.document_loaders import PyPDFLoader
elif suffix == ".docx":
    from langchain_community.document_loaders import Docx2txtLoader
elif suffix in (".md", ".txt"):
    return path.read_text(...)
```

注意 `.md` 是直接读纯文本而**不**用 `UnstructuredMarkdownLoader`——因为 markdown 的结构（标题、列表）保留下来对后续切片和 BM25 召回都更友好，转换成所谓的"结构化对象"反而是损失。

#### 4.4.2 去重：`precheck_files()` + `file_sha1`

上传前先按内容哈希查一遍 ChromaDB，已存在的直接跳过，不重复入库。这就是为什么 UI 里换个名字上传同一份文件不会产生两份索引。

#### 4.4.3 切片：`RecursiveCharacterTextSplitter` 配中文分隔符

```python
separators=[
    "\n\n", "\n",
    "。", "！", "？", "；", "，",
    ".", "!", "?", ";", ",",
    " ", "",
]
```

工作机制：splitter 会**按顺序尝试**这些分隔符，先按段落分（`\n\n`），段太长就改按行分（`\n`），还太长就按句号分……一直退化到字符级。这样可以最大可能保持语义完整。

`chunk_size=512, chunk_overlap=64` 是平衡**召回粒度**和**上下文连续性**的折中：
- 太大：embedding 失焦，相似度下降
- 太小：跨段落的语义被切断
- overlap：让相邻 chunk 之间有点重叠，避免"刚好被切在关键句中间"

#### 4.4.4 元数据：`make_node_metadata()` 注入了什么

```python
{
    "file_hash": ...,       # 联动删除用
    "file_name": ...,       # 落盘后的文件名（带 timestamp 前缀）
    "original_name": ...,   # 用户上传时的原始名
    "file_path": ...,       # 绝对路径
    "file_size": ...,
    "uploaded_at": ...,
}
```

这些字段会**冗余地**写到每个 chunk 的 metadata 里。看起来浪费，但换来了：
- 检索结果可以直接显示"这段来自哪个文件"
- 按 file_hash 批量删除时不需要额外索引表
- `doc_filter` 限定"只在某些文件里检索"时直接过滤 metadata

#### 4.4.5 写入：分批 embed + 双写

```python
for start in range(0, total_nodes, embed_batch_size):
    batch = chunks[start: start + embed_batch_size]
    embeddings = embed_model.embed_documents(texts)  # ← 一次 API 调用
    collection.add(ids=..., embeddings=..., documents=..., metadatas=...)
    snapshot_buffer.extend(batch)
get_doc_store().append_snapshot(snapshot_buffer)
```

为什么 batch size = 10？太大单次请求慢、超时风险高；太小请求次数多、进度条更新太频繁。10 是个经验值。

> 🧠 **学习要点**：管线里的每一步都有可能失败（网络、文件损坏、格式不支持），代码里几乎每个 `try` 都对应一种现实里碰过的错误。读 `ingest.py` 时不要跳过那些 `except`——它们是"已知会出的问题"的清单。

---

### 4.5 `retriever.py`：混合检索 + RRF + Rerank

整个项目最值得细读的一个文件。一次检索的完整流程：

```
query
  ├─→ [向量检索] 召回 vec_pool 条（pool 比 top_k 大 3 倍）
  └─→ [BM25 检索] 召回 bm25_pool 条
        ↓
   [RRF 融合] 不同来源、不同分数体系的两个列表融合成一个排序
        ↓
   [doc_filter 过滤] 如果用户限定了文件范围
        ↓
   [DashScope Rerank] 让最相关的几条排到最前
        ↓
   返回最终 top_n
```

#### 4.5.1 为什么要混合检索？

| 问题类型 | 向量检索的表现 | BM25 的表现 |
|---|---|---|
| "羊毛衣物的护理方式" | 强，能匹配同义/近义表达 | 弱，需要原文出现"羊毛""护理" |
| "型号 ABC-123 的参数" | 弱，型号代码语义稀疏 | 强，关键字精准匹配 |
| "什么时候提交了 PR #4567" | 弱 | 强 |

向量擅长"语义"，BM25 擅长"关键词/实体"。两个加起来覆盖面最广。

#### 4.5.2 RRF 为什么这么简单还好用

```python
score = Σ 1/(k + rank_i)        # k=60 是经验常数
```

为什么有效：
- **不依赖原始分数尺度**：BM25 分数和向量相似度不在同一个量级（一个可能是 12.5，一个是 0.83），直接相加没意义。RRF 只看排名（rank），完全规避了这个问题
- **位置敏感**：rank=1 贡献 `1/61`，rank=10 贡献 `1/70`，差距明显但不悬殊
- **天然处理重复**：同一个 doc 在两个排序中都出现，分数累加，鼓励"被多个检索器都召回"

不需要任何训练、不需要调参。这就是为什么我们没用 LangChain 的 `EnsembleRetriever`（它有些版本要求显式权重）——RRF 就够了。

#### 4.5.3 `_ChromaVectorRetriever` 为什么不用 `langchain_chroma.Chroma`

直接用 `chromadb.PersistentClient` 原生客户端，原因：
- LangChain 的 `Chroma` wrapper 在不同版本之间 API 不兼容（参数名换、返回值结构变）
- 我们只需要 `query()` 一个方法，包一层 wrapper 收益太低
- 出问题时栈帧清晰，方便定位

#### 4.5.4 `cosine distance` → `相似度` 的坑

```python
score = (1.0 - d) if isinstance(d, (int, float)) else None
```

ChromaDB 默认返回的是 cosine **distance**（越小越像），但 UI 上希望"分数越大越相似"。简单地 `1 - distance` 就转成了相似度。这一步如果忘了，UI 上 score 排序就反了。

#### 4.5.5 `set_doc_filter()` 为什么用 `contextvars`

UI 端可能勾了"只在这 3 个文件里检索"。最直接的做法是给 `retrieve()` 加个参数。但这意味着 `chat.py` 里 `_do_retrieve` 也要加、要往下传……整条调用链都要改。

`contextvars` 是 Python 标准库的"隐式上下文变量"，专门用来传递"逻辑请求级"的数据（在异步场景下比 `threading.local` 更安全）：

```python
token = set_doc_filter(["file_a.pdf"])
try:
    docs = retriever.retrieve_scored(query)   # 内部 _doc_filter_var.get() 读到 ["file_a.pdf"]
finally:
    reset_doc_filter(token)  # 用完一定要 reset，否则会污染下一次
```

> 🧠 **学习要点**：`contextvars` 适合"传递面向横切关注点（cross-cutting concern）的轻量配置"。日志的 trace_id、限流的 user_id 都是同类用法。

#### 4.5.6 Rerank 的本质

Embedding + cosine 给出的是"句子级别的语义相似度"。但相似 ≠ 真的能回答问题。Rerank 模型是一种 **cross-encoder**：它会把 query 和 doc **拼在一起**送进模型，对每一对单独打分。

代价：比向量检索贵 N 倍（要算 N 次模型推理）。所以套路是：
1. 用便宜的向量+BM25 召回 ~10 条候选
2. 把这 10 条全部送 rerank，挑最好的 4 条
3. 只把这 4 条放进 LLM 的 context

Rerank 是 RAG 系统从 60 分提升到 85 分的关键一步。

#### 4.5.7 `RetrievalTrace`：可观测性

```python
@dataclass
class RetrievalTrace:
    vector_hits: List[Dict]
    bm25_hits: List[Dict]
    fused_hits: List[Dict]
    rerank_hits: List[Dict]
    rerank_used: bool
```

每次检索后把四个阶段的命中都记下来，挂在 `retriever.last_trace`。Streamlit 里直接读这个字段做"召回可视化"，方便排查"为什么没召回到我想要的那段"。

> 🧠 **学习要点**：复杂的 pipeline 一定要有 trace。不知道中间发生了什么的系统是不可调试的。

---

### 4.6 `chat.py`：多轮对话引擎

#### 4.6.1 多轮对话最大的难题

考虑这个对话：

```
用户：苹果公司的 CEO 是谁？
助手：是蒂姆·库克。
用户：他什么时候上任的？
```

如果直接拿"他什么时候上任的"去检索向量库，"他"是个空指代，根本检索不到任何相关内容。这就是为什么需要 **condense（浓缩）**：先用 LLM 把上下文嵌进当前问题里，得到独立可检索的版本：

```
"蒂姆·库克什么时候上任苹果公司 CEO 的？"
```

`_condense_question()` 干的就是这件事。注意几个细节：
- 只取最近 8 轮（防 prompt 过长）
- 每条消息截 400 字符（同上）
- LLM 偶尔会回 `"xxx"` 带引号、`「xxx」`带书名号，做一次 `.strip()` 清理
- 浓缩失败时降级用原问题，永远不让对话断掉

#### 4.6.2 `_build_messages` 的 prompt 结构

```
SystemMessage:
   ├─ BASE_SYSTEM_PROMPT（你是知识库助手，只用资料回答……）
   ├─ STYLE_PROMPTS[style]（输出格式：简洁/详细/表格/步骤化）
   └─ 【参考资料】[1] 来源：xxx \n ... [2] 来源：yyy \n ...

HumanMessage（历史轮次的用户消息）
AIMessage（历史轮次的助手回答）
...
HumanMessage（当前轮的问题）
```

为什么把检索到的资料放在 system prompt 里、不放在 user 消息里？因为 system 在很多模型里有"更高的指令优先级"，能更好地约束模型不脱离资料。

#### 4.6.3 流式生成的 fallback 机制

```python
def _gen():
    try:
        for chunk in get_llm().stream(msgs):
            yield chunk.content
    except Exception:
        yield "[流式生成中断]"
        return

    if not collected:
        # 流没崩，但一字未吐 → 部分模型会静默失败
        resp = get_llm().invoke(msgs)  # 改用非流式重试
        yield resp.content
```

这是踩坑后加上的：DashScope 偶尔会返回成功的流但不吐 token（一个 0 长度的 stream）。监测到这种情况后立刻降级为非流式重试，保证用户至少看到答案。

#### 4.6.4 `suggest_followups`：稳健解析 LLM 的 JSON 输出

让 LLM 返回 JSON 数组听起来简单，实际上 LLM 经常会：
- 在 JSON 前后加解释文字
- 用单引号
- 用 `「」` 中文引号
- 加 markdown 代码围栏 ` ```json `

`_parse_followups()` 的三段式解析体现了"防御性编程"：

1. 先试 `json.loads`（顺利时最快）
2. 再用正则 `\[.*?\]` 抠出第一个数组（处理前后有解释的情况）
3. 最后兜底：按行拆，去掉行首的 `1.`、`-`、`*`、`(` 之类的列表标记

> 🧠 **学习要点**：和 LLM 交互的所有代码都要写得"宽容"，假设它会出格、会偷懒、会发明语法。

---

### 4.7 `sessions.py`：会话持久化

设计很朴素：每个会话一个 JSON 文件，文件名是 session_id。

为什么不用数据库？
- 单用户场景，JSON 够用
- 文件级锁定简单（写入用 `tmp -> replace` 模式）
- 直接打开能看，便于调试
- 用户可以随手 `cp xxx.json /backup/` 备份

注意 `list()` 是扫整个目录的——如果会话数量上千，需要换 SQLite。但本项目预期是"单人 + 几十个会话"，目前足矣。

---

### 4.8 `api.py`：FastAPI 路由

按资源分组：

```
/health                       自检
/chat                         同步对话
/chat/stream                  SSE 流式对话
/chat/followups               追问建议
/sessions (GET/POST)          列表 / 新建
/sessions/{sid}               读 / 改 / 删 / 导出
/sessions/import              导入
/documents (GET)              列表
/documents/upload (POST)      上传 + 入库
/documents/{file_hash}        删除（联动）
```

重点看两件事：

#### 4.8.1 SSE 流式协议

```python
def _format_event(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
```

SSE（Server-Sent Events）协议要求每条消息以 `data: <内容>\n\n` 结尾。我们定义了三种事件：

```
{"event": "token", "text": "今"}   ← 每次新 token
{"event": "token", "text": "天"}
...
{"event": "done", "session_id": ..., "answer": ..., "citations": [...], "trace": {...}}
{"event": "error", "message": "..."}
```

前端拿 `EventSource` 或 `fetch` 流式读响应即可。

#### 4.8.2 入库/删除后必须重置三个单例

```python
reset_doc_store()
reset_retriever()
reset_chat_service()
```

因为：
- `DocStore` 缓存了 chroma 的 count（统计陈旧）
- `Retriever` 在内存里持有 BM25 索引（节点列表过时）
- `ChatService` 持有 retriever 引用（间接陈旧）

不重置会导致用户上传新文件后问问题搜不到刚上传的内容。这种"单例缓存失效"的坑很经典。

---

### 4.9 `streamlit_app.py`：Web UI

不展开讲，几个值得注意的 Streamlit 技巧：

- 用 `st.sidebar` + `st.radio` 切换页面，**不用 `st.tabs`**——后者会让 `st.chat_input` 不固定到屏幕底部
- `st.session_state` 是页面级状态容器，刷新页面（rerun）后保留
- `st.write_stream(generator)` 自动把流式 token 渲染成打字机效果
- 文件预览用 "session_state 标志位 + rerun + 完整宽度 panel" 模式，比 `st.popover` 更适合大段内容

---

## 5. 关键设计决策（Why Not）

### 5.1 为什么不用 `langchain.chains` 系列

LangChain 的 `RetrievalQA` / `ConversationalRetrievalChain` 把整套流程封装得很高，但带来的问题是：
- 改任何一步（比如想插入 rerank）都得拆开
- 不同版本之间 API 大改，升级困难
- 出错时栈深、定位难

我们手工拼 messages、自己调 retriever 和 LLM，**多写了一些胶水代码，换来全局可控**。这是一个工程权衡。

### 5.2 为什么不用 `langchain.retrievers.EnsembleRetriever`

它要求每个子 retriever 都是 LangChain `BaseRetriever` 的子类，接口对不上时要写适配器。RRF 自己手写就 20 行，没必要绕一圈。

### 5.3 为什么用 ChromaDB 而不是 FAISS

| | ChromaDB | FAISS |
|---|---|---|
| 持久化 | ✅ 内置 | ❌ 自己存/加载 |
| 元数据过滤 | ✅ where 子句 | ❌ 要自己维护 |
| 客户端 | ✅ 原生 Python | C++ 库，部署稍麻烦 |
| 性能 | 单机够用 | 大规模更快 |

本项目目标是"单机本地知识库"，ChromaDB 完胜。规模到亿级再考虑 Milvus/Weaviate。

### 5.4 为什么 BM25 单独维护一份 `nodes.json`

详见 4.3.3。核心是"避免每次启动都全量拉 Chroma 数据"。

---

## 6. 关键概念深入

### 6.1 Embedding 模型选型对比

| 模型 | 维度 | 中文 | 速度 | 备注 |
|---|---|---|---|---|
| `text-embedding-v3` (DashScope) | 1024 | ✅ 强 | 中（API） | 在线、按量计费 |
| `text-embedding-v4` (DashScope) | 1024 | ✅ 强 | 中（API） | 比 v3 略好 |
| `BAAI/bge-large-zh-v1.5` (HF) | 1024 | ✅ 强 | 快（本地） | 中文 SOTA 之一 |
| `BAAI/bge-m3` (HF) | 1024 | ✅ 强 | 中 | 多语言、多向量 |

切换时需要 **重新 ingest**，因为不同模型生成的向量在不同的语义空间，无法兼容。

### 6.2 chunk_size 的影响

| chunk_size | 优点 | 缺点 |
|---|---|---|
| 256 | 召回精准 | 上下文短，LLM 看不到全貌 |
| 512（默认） | 平衡 | / |
| 1024 | 上下文丰富 | embedding 失焦，召回掉粒度 |
| 2048 | 适合长篇连续问题 | 召回严重失焦 |

经验：中文文档 500-800 字符是甜点区。

### 6.3 BM25 的局限

BM25 是 1970 年代的算法，本质是 **TF-IDF 的改进**。它假设：
- 词与词之间独立
- 词序不重要
- 没有同义词概念

这就是为什么"羊毛护理"查不到"如何洗毛衣"——它们没有共同词。**BM25 必须配合 embedding 才能补全这个短板**。

### 6.4 Rerank 模型 vs Embedding 模型

| | Embedding（bi-encoder） | Rerank（cross-encoder） |
|---|---|---|
| 输入 | 单条文本 | (query, doc) 对 |
| 输出 | 一个向量 | 一个相关度分数 |
| 速度 | 极快（向量索引） | 慢（每对都要算） |
| 精度 | 中 | 高 |
| 用法 | 召回 | 精排 |

Embedding 的"召回 N=10"和 Rerank 的"精排取 4"是天作之合：召回靠速度、精排靠质量。

### 6.5 CondensePlusContext 模式

LlamaIndex 里这是一个内置 chat engine。我们手实现等价的两步：

```
[Condense] 把"对话历史 + 当前问题"交给 LLM → 输出独立可检索问句
[Context]  用独立问句去检索 → 拼到 system prompt 的【参考资料】里
```

替代方案：
- "Window memory"：直接把最近 N 轮全塞 prompt（简单但浪费 token）
- "Summary memory"：把历史压缩成总结（适合超长对话）
- "Vector memory"：把历史也存向量库，按需检索（复杂）

我们选 Condense+Context 是因为它**和检索 RAG 流程天然契合**——反正都要做检索，那就把改写问题当前置步骤。

---

## 7. 调试与观测

### 7.1 三层日志

```python
# 进入文件时已经配过：
logging.basicConfig(level=logging.INFO, ...)
logging.getLogger("dashscope").setLevel(logging.DEBUG)  # SDK 详细日志
logging.getLogger("httpx").setLevel(logging.WARNING)    # 抑制噪音
```

按需要看：
- `[INFO] src.chat`：每轮问题、检索分数、风格切换
- `[INFO] src.retriever`：BM25 加载情况、降级提示
- `[DEBUG] dashscope`：HTTP 请求体（看 API key 有没有正确下发）

### 7.2 `RetrievalTrace`

UI 的"召回详情"展开后能看到四个阶段每条命中：

```
🔎 向量    🧮 BM25    🔀 融合    🏆 Rerank
#1 0.835   #1 12.4    #1 0.0285  #1 0.998
来源 a.pdf 来源 b.md  来源 a.pdf 来源 c.txt
预览...     预览...    预览...    预览...
```

排查"为什么没召回到 X"的标准操作：
1. 看 vector_hits 里有 X 吗？没有 → embedding 模型对该领域不灵
2. 看 bm25_hits 里有 X 吗？没有 → 关键词没匹配上
3. fused 里有但 rerank 没有？→ rerank 觉得不够相关
4. 都没有？→ 可能 ingest 时根本没切到那段，去 `nodes.json` 搜原文

### 7.3 `scripts/diagnose.py`

跑通这个脚本相当于做了一次系统体检：

```bash
python scripts\diagnose.py
```

它会逐步检查：配置 → embedding → LLM → rerank。任何一步失败都会给出可操作的错误信息。

---

## 8. 实战练习

按照难度递增：

### Level 1：读懂

1. **追踪一个 chunk 的一生**：往 `data/uploads/` 放一个 `test.txt`，跑 `python scripts\ingest.py`。打开 `storage/docstore/nodes.json` 找到这个 chunk，记下 id；打开 ChromaDB 用 `chromadb.PersistentClient` 查同一个 id，对比 metadata 是否一致。
2. **观察检索**：在 Streamlit 问一个问题，展开"召回详情"，对比向量/BM25/融合/rerank 四列的命中变化。

### Level 2：小改

3. **加一个新的回答风格**：在 `chat.py` 的 `STYLE_PROMPTS` 里加 `"academic": "请用学术论文风格..."`，看 UI 是否自动出现选项。
4. **改 chunk_size 重新 ingest**，对比同一个问题的召回质量（先 256，再 1024）。
5. **加一个 `/documents/{file_hash}/preview` 路由**，返回该文件的全部 chunk。

### Level 3：扩展

6. **支持 `.html` 文件**：在 `_load_one()` 里加分支，用 `BeautifulSoup` 抽取正文。
7. **加一个 reranker 切换开关**：让 `RERANK_MODEL=none` 时跳过 rerank（已有降级逻辑，只需在 `_dashscope_rerank` 加判断）。
8. **加 cross-session 搜索**：在 `SessionStore` 里实现"全文搜索"，UI 端展示命中所在的会话+消息。

### Level 4：深入

9. **替换 BM25 为 SPLADE**（神经稀疏检索）：保持接口不变，仅替换 `_BM25Retriever` 内部实现。
10. **加 streaming 上的"打断"功能**：用户点"停止"后立刻断流并丢弃后续 token，但已生成的部分要存到会话里。
11. **加 "知识更新通知"**：当 `data/uploads/` 下文件被外部修改时（mtime 变化），自动重新 ingest 该文件。

---

## 9. 进阶扩展方向

如果想把这个项目演进成生产级系统：

| 方向 | 当前状态 | 演进方向 |
|---|---|---|
| 向量库 | 单机 ChromaDB | Milvus / Weaviate / Qdrant 集群 |
| 会话存储 | 文件 JSON | SQLite → Postgres |
| LLM | 单一 DashScope | 多 provider 路由（OpenAI/Claude/Qwen 自动 fallback） |
| 鉴权 | 无 | API Key + Session Cookie |
| 多用户 | 无 | 加 user_id 维度，retriever / sessions 都隔离 |
| 监控 | 日志 | OpenTelemetry + Prometheus |
| 向量去重 | 无 | 入库时检测高度相似 chunk，避免冗余 |
| 增量索引 | 全文件 hash | chunk-level diff（修改少量内容只重 embed 变更部分） |
| 反馈学习 | 无 | 用户给答案打分，定期重训 reranker |

---

## 10. 总结：这个项目想教你什么

把项目读完一遍，你应该带走以下几个"工程直觉"：

1. **抽象边界要清楚**：`config` 只管配置、`settings` 只管 LLM/Embedding 工厂、`doc_store` 只管文件级元数据。每个文件能用一句话说清职责。

2. **避免过度抽象**：不要为了用 LangChain 的高级 API 而放弃控制权。手写 RRF 比绕 `EnsembleRetriever` 简单得多。

3. **可观测性是 RAG 的氧气**：没有 `RetrievalTrace`，"为什么效果不好"就是黑盒。

4. **第三方 SDK 的全局副作用要集中管理**：`_ensure_dashscope_global_key` 救了三种坑（LLM/Embedding/Rerank 的 401）。

5. **对 LLM 的输出要"宽容"**：JSON 解析要三段式、流式要有 fallback、空输出要有兜底提示。

6. **数据冗余换查询效率是常见手法**：`nodes.json` 是 ChromaDB 的冗余备份，但让 BM25 启动从秒级降到毫秒级。

7. **单例 + 重置函数 是带状态服务的标配**：上传/删除后必须 reset，这是 RAG 系统最常见的"看不到新数据"问题的根因。

祝学习愉快。读完一遍后，强烈建议把 `streamlit_app.py` 跑起来，边操作边看代码，**让代码"动起来"远比读静态代码学得快**。
