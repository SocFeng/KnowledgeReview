# LlamaIndex × Qwen RAG

一个基于 **LlamaIndex + 阿里云百炼 (Qwen)** 的检索增强问答 (RAG) 项目。
开箱即用地集成了：

- 📚 **多格式文档**：PDF / Markdown / TXT / DOCX
- 🧠 **Qwen LLM**：通过 DashScope 调用 (`qwen-plus` / `qwen-max` 等)
- 🔢 **可切换 Embedding**：DashScope `text-embedding-v3` 或本地 HuggingFace BGE
- 🗄️ **ChromaDB**：本地持久化向量存储
- 🔍 **混合检索**：向量检索 + BM25（中文 jieba 分词）+ Reciprocal Rerank Fusion
- 🎯 **DashScope Rerank**：`gte-rerank-v2` 精排
- 💬 **多轮对话**：基于 `CondensePlusContextChatEngine`，自动浓缩历史
- 🔗 **引用溯源**：每条回答附带来源片段，UI 中 `[n]` 可点击跳转
- 📊 **召回可视化**：vector / BM25 / 融合 / rerank 四阶段命中分数对比
- ✨ **追问建议**：每轮回答后自动生成 3 个追问按钮，点击即发
- 🎛️ **回答风格切换**：简洁 / 详细 / 对比表格 / 步骤化
- 🗂️ **文档级管理**：按内容 hash 去重；删原始文件联动清理向量库
- 📌 **会话管理**：搜索 / 重命名 / 固定置顶 / 导入导出 (JSON & Markdown)
- 🌐 **FastAPI**：REST 接口，自带 `/docs` Swagger 页面
- 🖥️ **Streamlit Web UI**：上传 → 解析 → 问答全流程

---

## 1. 目录结构

```
llamaIndex_rag/
├── data/
│   ├── uploads/                # Streamlit 上传的原始文件
│   └── chat_history/           # 持久化会话 JSON（自动创建）
├── storage/                    # 索引持久化目录（首次 ingest 后生成）
│   ├── chroma/                 # Chroma 向量库
│   └── docstore/               # LlamaIndex docstore（BM25 复用）
├── src/
│   ├── config.py               # 配置（pydantic-settings 读取 .env）
│   ├── settings.py             # LlamaIndex 全局 LLM / Embedding
│   ├── ingest.py               # 数据摄入：读取 -> 切片 -> 注入元数据 -> 写库
│   ├── doc_store.py            # 文档级 CRUD：按 file_hash 查/删/统计
│   ├── retriever.py            # 混合检索 + DashScope rerank + doc_filter + trace
│   ├── chat.py                 # 多轮对话：流式 + 风格切换 + 追问生成 + 引用提取
│   ├── sessions.py             # 会话持久化：增删改查 + 导入导出 + Markdown 渲染
│   └── api.py                  # FastAPI 接口
├── scripts/
│   ├── ingest.py               # 入口：构建索引
│   ├── run_api.py              # 入口：启动 API
│   ├── ask_once.py             # 入口：命令行单轮问答
│   └── diagnose.py             # 入口：DashScope 连接 / API key 诊断
├── streamlit_app.py            # Streamlit Web UI 入口
├── .streamlit/config.toml      # Streamlit 配置（关闭文件 watcher 等）
├── requirements.txt
├── .env.example
└── README.md
```

---

## 2. 快速开始

### 2.1 创建并激活虚拟环境（推荐）

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 2.2 安装依赖

```bash
pip install -r requirements.txt
```

### 2.3 配置 API Key

复制 `.env.example` 为 `.env`，填入你的 DashScope API Key：

```bash
cp .env.example .env
# 然后编辑 .env，填上 DASHSCOPE_API_KEY=sk-xxxx
```

> 申请地址：<https://bailian.console.aliyun.com/>

### 2.4 启动 Streamlit Web UI（推荐入口）

```bash
streamlit run streamlit_app.py
```

浏览器打开后在「📤 文档管理」页上传文件 → 点「🚀 开始解析」就能开始问答。
完整 UI 功能见 [§3 Streamlit Web UI](#3-streamlit-web-ui)。

### 2.5 命令行方式构建索引（可选）

如果你不想用 UI，也可以把文档放到 `data/` 任意子目录，然后：

```bash
python -m scripts.ingest
```

成功后会在 `storage/chroma/` 与 `storage/docstore/` 看到持久化文件。

### 2.6 启动 FastAPI 服务（可选）

```bash
python -m scripts.run_api
```

打开浏览器访问 <http://localhost:8000/docs> 即可看到自动生成的 Swagger UI。

---

## 3. Streamlit Web UI

```bash
streamlit run streamlit_app.py
```

侧边栏 radio 切换两个页面（用 radio 而不是 tabs 是为了让 `st.chat_input`
能固定在屏幕底部）。

### 3.1 📤 文档管理页

- **多文件批量上传**（PDF / MD / TXT / DOCX），支持拖拽
- **内容级去重**：按文件 SHA-1 判断，重复文件不会重复落盘也不会重复入库
- **分阶段进度条**：保存文件 → 加载 → 切片 → 生成向量并写入
- **已上传文件列表**：每行显示
  - 🟢/⚪ 入库状态、文件大小、入库时间
  - 📦 在向量库中占多少节点 + 占比
  - 🗑️ **联动删除**：同时删除原始文件 + Chroma 中对应节点 + docstore 中对应节点
- **孤儿向量清理**：磁盘已删但 Chroma 还有的节点，会单独列出供 🧹 清理

### 3.2 💬 知识问答页

#### 对话功能
- 类 ChatGPT 流式打字机效果
- AI 状态指示：🔎 检索中 → 🤔 思考中 → ✍️ 回答中
- 每条回答附带 **「📎 引用 N 个片段」** 折叠面板（来源文件、相关度、原文预览）
- 答案中的 `[1]` `[2]` 引用编号是**可点击的链接**，点击自动滚到对应引用片段并高亮
- **追问建议**：每条回答下方自动生成 3 个相关追问按钮，点击直接发送
- **召回详情面板**：每条回答下可展开查看 vector / BM25 / 融合 / rerank 四阶段的命中分数与片段预览，方便调参

#### 侧边栏控制
- **🎯 检索范围**：多选框限定本次只在选中的文档里检索（留空 = 全部）
- **回答格式**：4 选 1 风格切换，切换后立即应用到当前会话
  - 🎯 简洁：1-3 句要点式
  - 📖 详细：分段展开背景、结论、注意事项
  - 📊 对比表格：尽量用 Markdown 表格组织
  - 🪜 步骤化：编号列表 + 注意事项

#### 会话管理
- **选择 / 新建 / 删除**：顶部 selectbox + 按钮
- **🔍 搜索**：按标题或消息内容关键字过滤会话列表
- **✏️ 重命名**：popover 输入新标题
- **📌 置顶**：固定的会话排在列表最前
- **⬇️ 导出**：当前会话可一键下载为 Markdown 或 JSON
- **📤 导入**：上传之前导出的 JSON 即可恢复会话（含历史 + 引用）
- **自动持久化**：每轮回答后自动落盘到 `data/chat_history/<sid>.json`，刷新 / 重启 / 切回都不丢

---

## 4. API 用法

启动后访问 <http://localhost:8000/docs> 看完整 Swagger UI。
路由按功能分组：`meta` / `chat` / `sessions` / `documents`。

### 4.1 健康检查

```bash
curl http://localhost:8000/health
# -> {"status":"ok","version":"0.2.0","llm_model":"qwen-plus",
#     "embedding_provider":"huggingface","vector_count":15}
```

### 4.2 文档管理

```bash
# 列出所有已入库文档（含每个文件的节点数）
curl http://localhost:8000/documents

# 上传 + 自动入库（多文件，按 hash 去重）
curl -X POST http://localhost:8000/documents/upload \
  -F "files=@./guide.pdf" \
  -F "files=@./faq.md"
# -> {"files":2,"nodes":17,"skipped":0,"skipped_files":[],"elapsed":3.42}

# 按 file_hash 删除（同时清 Chroma + docstore + 原始文件）
curl -X DELETE "http://localhost:8000/documents/<file_hash>?delete_file=true"
```

### 4.3 多轮对话

```bash
# 同步问答（带文档过滤 + 风格切换）
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "针织衣物如何养护？",
    "style": "steps",
    "doc_filter": ["洗涤养护.txt"]
  }'
```

返回示例：

```json
{
  "session_id": "2dff0f5c4896420d995fe94896b22aad",
  "answer": "1. 手洗优先...\n2. 平铺阴干...\n[1][3]",
  "citations": [
    {"index":1, "score":0.43, "file_name":"洗涤养护.txt", "text":"针织棉材质..."}
  ],
  "trace": {
    "vector_hits":  [...],
    "bm25_hits":    [...],
    "fused_hits":   [...],
    "rerank_hits":  [...],
    "rerank_used":  true,
    "doc_filter":   ["洗涤养护.txt"]
  }
}
```

不传 `session_id` 时会自动新建一个并随响应返回。
`style` 可选：`concise` / `detailed` / `table` / `steps`。

### 4.4 流式对话 (SSE)

```bash
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message":"什么是 RAG？","style":"detailed"}'
```

事件流（每行 `data: <json>\n\n`）：

- `{"event":"token", "text":"..."}` — 每次新 token
- `{"event":"done",  "session_id":"...", "answer":"...", "citations":[...], "trace":{...}}`
- `{"event":"error", "message":"..."}` — 中断

### 4.5 追问建议

```bash
curl -X POST http://localhost:8000/chat/followups \
  -H "Content-Type: application/json" \
  -d '{
    "question": "针织衣物如何养护？",
    "answer":   "手洗优先，水温≤25℃...",
    "n": 3
  }'
# -> {"suggestions": ["不同针织材质区别？", "起球怎么处理？", "如何收纳防变形？"]}
```

### 4.6 会话管理

```bash
# 列表（按标题/内容关键字搜索；置顶会话排前面）
curl "http://localhost:8000/sessions?q=养护"

# 获取详情（含所有消息 + 引用）
curl http://localhost:8000/sessions/<session_id>

# 重命名 + 置顶（PATCH，两字段都可选）
curl -X PATCH http://localhost:8000/sessions/<session_id> \
  -H "Content-Type: application/json" \
  -d '{"title":"衣物养护方案","pinned":true}'

# 删除（同时清后端 engine memory + 持久化文件）
curl -X DELETE http://localhost:8000/sessions/<session_id>

# 导出为 Markdown（带 attachment 头，可直接 -o 落盘）
curl -o session.md "http://localhost:8000/sessions/<session_id>/export?format=md"

# 导出为 JSON
curl -o session.json "http://localhost:8000/sessions/<session_id>/export?format=json"

# 导入 JSON 会话（始终分配新 session_id，不会覆盖现有）
curl -X POST http://localhost:8000/sessions/import \
  -H "Content-Type: application/json" \
  --data-binary @session.json
```

### 4.7 命令行单轮问答（无需启动服务）

```bash
python -m scripts.ask_once "什么是 RAG？"
```

---

## 5. 关键参数（在 `.env` 里调）

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DASHSCOPE_API_KEY` | （必填） | 阿里云百炼 API Key |
| `LLM_MODEL` | `qwen-plus` | 生成模型，可换成 `qwen-max` 等 |
| `EMBEDDING_PROVIDER` | `dashscope` | `dashscope` 或 `huggingface` |
| `EMBEDDING_MODEL` | `text-embedding-v3` | DashScope embedding 模型 |
| `RERANK_MODEL` | `gte-rerank-v2` | rerank 模型（注意 v1 已下线） |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 512 / 64 | 切片粒度 |
| `VECTOR_TOP_K` | 5 | 向量检索召回数 |
| `BM25_TOP_K` | 5 | BM25 召回数 |
| `RERANK_TOP_N` | 4 | rerank 后保留的最终上下文数 |

> 💡 **检索召回不足时**：retriever 内部会先按 `top_k * 3` 拿到更大的候选池，
> 再按 `doc_filter` 过滤、最后交给 rerank 截断。所以即便启用了"检索范围"
> 多选，召回质量也不会明显下降。

---

## 6. 切换到本地 HuggingFace Embedding

如果你想完全离线、或者希望用更适合自己领域的中文 embedding，可以把 embedding 切到本地 HF 模型（LLM 仍走 Qwen DashScope，rerank 也仍走 DashScope）。

### 6.1 准备模型

本地下载一个 BGE 模型（任选其一，按效果/体积权衡）：

| 模型 | 大小 | 推荐场景 |
| --- | --- | --- |
| `BAAI/bge-m3` | ~2.3 GB | 多语言 SOTA，最佳通用选择 |
| `BAAI/bge-large-zh-v1.5` | ~1.3 GB | 中文专项优化 |
| `BAAI/bge-base-zh-v1.5` | ~400 MB | 平衡选择 |
| `BAAI/bge-small-zh-v1.5` | ~100 MB | 资源紧张/快速验证 |

下载方式（任选一种）：

```powershell
# 方式 1：用 huggingface-cli（推荐，可断点续传）
huggingface-cli download BAAI/bge-large-zh-v1.5 --local-dir D:\models\bge-large-zh-v1.5

# 方式 2：国内镜像
$env:HF_ENDPOINT="https://hf-mirror.com"
huggingface-cli download BAAI/bge-large-zh-v1.5 --local-dir D:\models\bge-large-zh-v1.5

# 方式 3：git lfs clone
git lfs install
git clone https://huggingface.co/BAAI/bge-large-zh-v1.5 D:\models\bge-large-zh-v1.5
```

### 6.2 配置 `.env`

```ini
EMBEDDING_PROVIDER=huggingface
HF_EMBEDDING_MODEL=D:/models/bge-large-zh-v1.5      # 注意路径用斜杠 / 或转义反斜杠
HF_EMBEDDING_DEVICE=auto                            # auto / cpu / cuda / cuda:0 / mps
HF_EMBEDDING_BATCH_SIZE=16                          # CPU 建议 8~16，GPU 可调到 32~64
HF_EMBEDDING_QUERY_INSTRUCTION=为这个句子生成表示以用于检索相关文章：
```

> ⚠️ **切换 embedding 后必须重新 ingest**：不同 embedding 模型生成的向量空间互不兼容，旧的 Chroma 数据需要删除：
> ```powershell
> Remove-Item -Recurse -Force .\storage\
> python -m scripts.ingest
> ```

### 6.3 GPU 加速（可选）

默认安装的是 CPU 版 PyTorch。要使用 NVIDIA GPU，请先单独装 CUDA 版本（以 CUDA 12.1 为例）：

```powershell
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

之后 `HF_EMBEDDING_DEVICE=auto` 会自动检测并切到 `cuda`。

### 6.4 验证

启动 Streamlit / FastAPI 时，控制台会打印一行：

```
INFO | src.settings | 加载本地 HuggingFace embedding: D:\models\bge-large-zh-v1.5 (device=cpu)
```

看到这一行就说明已经切换成功。

---

## 7. 常见问题

**Q: 文档更新了怎么办？**
A: 在 Streamlit 「📤 文档管理」页直接重新上传即可。系统会按 SHA-1 自动判重——
内容相同的文件会被跳过、内容变化的文件会作为新文件入库。如果想清理旧版本节点，
点对应文件后面的 🗑️ 按钮即可联动清理 Chroma + docstore。命令行用户也可以直接
重新执行 `python -m scripts.ingest`。

**Q: 想完全重建索引？**
A: 删掉 `storage/` 目录后重新 ingest 即可：
```powershell
Remove-Item -Recurse -Force .\storage\
streamlit run streamlit_app.py    # 然后重新上传
```

**Q: 中文检索效果不好？**
A: BM25 已使用 jieba 中文分词；如果你的领域术语很特殊，可以在
`src/retriever.py` 的 `_chinese_tokenizer` 中预加载自定义词表
(`jieba.load_userdict(...)`)。也可以试试切换到本地 BGE-large-zh-v1.5
（见 §6）。问答页右下角的"召回详情"面板可以帮你对比四个阶段的命中差异，
直观看出问题出在 vector 还是 BM25 还是 rerank。

**Q: 历史会话存在哪？**
A: `data/chat_history/<session_id>.json`，每次回答后自动落盘。可以直接拷贝
迁移；也可以在 Web UI 上用「⬇️ 导出 JSON」+「📤 导入」做单文件流转。

**Q: 旧版本入库的数据没有 file_hash 元数据，新功能不生效怎么办？**
A: 这是因为 4.x 版本之前的 ingest 没注入文档级元数据。建议清空 `storage/`
和 `data/uploads/` 后重新上传一次，所有功能即可正常使用。

**Q: rerank 报 'NoneType' object has no attribute 'results'？**
A: 通常是 API key 没开 rerank 权限或 `RERANK_MODEL` 写错了。检查方法：
```bash
python -m scripts.diagnose
```
另外注意 `gte-rerank` v1 已下线，请使用 `gte-rerank-v2`。
即便 rerank 失败，系统也会自动降级为不 rerank 的混合检索结果，不会阻塞问答。

**Q: 流式输出 / 多轮中途突然返回 Empty Response？**
A: 这是 `llama-index-llms-dashscope` 的已知 bug：会给 assistant 消息塞空的
`tool_calls=[]`，被 DashScope API 拒绝。本项目在 `src/settings.py` 里做了
monkey-patch 修复，并在流式失败时自动降级为非流式重试。如果你看到日志里
有 `非流式重试成功`，说明兜底生效了。
