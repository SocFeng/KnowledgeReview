# 智能旅游规划 Agent · LangChain + LangGraph

> 用 **LangChain + LangGraph + 自定义工具** 实现的中文智能旅游规划助手。
> 用户可以通过类聊天对话告诉 Agent 出发地、目的地、出行天数、偏好等，
> Agent 会自动调用 **天气 / 地理编码 / 景点 / 文化 / 路线** 等工具，
> 整合成一份带 Markdown 排版的、可随时修改的行程方案。

```
┌──────────────────────────────────────────────────────────┐
│                Streamlit Chat UI（多会话）                │
└──────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────┐
│      LangGraph StateGraph (自定义 ReAct loop + Memory)    │
│                                                          │
│   ┌──────┐   tool_calls?   ┌──────┐                       │
│   │agent │ ─────yes──────► │tools │                       │
│   └──────┘ ◄────────────── └──────┘                       │
│      │                                                    │
│      ▼ no                                                 │
│    [END]                                                  │
└──────────────────────────────────────────────────────────┘
                          ↓ (bind_tools)
┌──────────────────────────────────────────────────────────┐
│  6 个自定义工具（@tool 装饰，免 Key 优先）                 │
│  geocode · weather · distance · attractions · culture · route │
└──────────────────────────────────────────────────────────┘
```

---

## 1. 功能特性

| 类别 | 能力 |
|---|---|
| 🤖 Agent 框架 | **LangGraph StateGraph 手写 ReAct loop**（不是 prebuilt 的 create_react_agent） |
| 🛠 工具 | 6 个全部**自己实现**：geocode / weather / distance / search_attractions / lookup_culture / plan_route |
| 🌐 数据源 | 优先免费无 key（OSM / Open-Meteo / Wikipedia），可选高德地图增强 |
| 💬 对话 | 类聊天 UI（Streamlit），支持**多会话切换、重命名、删除** |
| ⌨️ 真·token 流式 | 前端**打字机效果**：`stream_mode=["messages", "updates"]` 双通道，首 token ~2s 内抵达 |
| 💰 Token / 成本统计 | 按轮次 / 按会话累计 `input_tokens` / `output_tokens` / 人民币费用，侧边栏实时显示 |
| 🧠 记忆 | LangGraph **SqliteSaver checkpointer**，跨进程也能续聊 |
| 🔧 可观测 | 工具调用 trace 实时展开（参数 + 返回 JSON） |
| 🔁 上下文修改 | 用户随时改需求（"再加一天"、"去掉博物馆"），Agent 会基于历史增量调整 |
| 🇨🇳 中文 LLM | 阿里云百炼 DashScope Qwen 系列（`qwen-plus` / `qwen-max` / ...） |
| 🔌 LLM Provider 可切换 | 默认 **DashScope OpenAI 兼容端点 + `ChatOpenAI`**（流式 + tool_calls 稳定）；可切换回 `ChatTongyi` |

---

## 2. 目录结构

```
langChain_langGraph_agent/
├── README.md                       ← 本文
├── requirements.txt
├── .env.example                    ← 环境变量模板
├── .gitignore
├── streamlit_app.py                ← Web UI 入口
│
├── src/
│   ├── config.py                   ← pydantic-settings 配置中心
│   ├── llm.py                      ← LLM 工厂（ChatOpenAI 兼容模式 / ChatTongyi 双轨）
│   ├── prompts.py                  ← system prompt
│   ├── state.py                    ← LangGraph TravelState
│   ├── usage.py                    ← ⭐ Token 用量 + 成本估算 + 按 thread 持久化
│   ├── agent.py                    ← ⭐ StateGraph 编排 + checkpointer + 双 stream_mode
│   └── tools/                      ← 自定义工具
│       ├── _http.py                统一 HTTP 封装
│       ├── geocode.py              地址 → 经纬度
│       ├── weather.py              Open-Meteo 天气
│       ├── distance.py             haversine + 交通建议
│       ├── attractions.py          高德 POI / Wikipedia geosearch
│       ├── culture.py              Wikipedia 文化简介
│       └── route.py                高德 driving / 回退直线
│
└── scripts/
    ├── diagnose.py                 ⭐ 体检：配置 → LLM → 各工具
    └── chat_cli.py                 命令行版对话（无浏览器场景）
```

---

## 3. 快速开始

### 3.1 创建虚拟环境（Python 3.11）

```powershell
cd langChain_langGraph_agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3.2 安装依赖

```powershell
pip install -r requirements.txt
```

### 3.3 配置 API Key

```powershell
copy .env.example .env
# 用编辑器打开 .env，至少填入 DASHSCOPE_API_KEY
```

> DashScope Key 获取：[阿里云百炼控制台](https://bailian.console.aliyun.com/)
>
> 高德 Key（**可选**）获取：[高德开放平台](https://console.amap.com/dev/key/app) → 选 "Web 服务"。
> 不填高德 key 也能跑，路线 / 景点工具会 fallback 到免费数据源 + 模型常识。

### 3.4 自检（强烈推荐）

```powershell
python -m scripts.diagnose
```

会逐项验证：配置 → LLM → 6 个工具是否都能拿到数据。哪一项失败都会给出具体错误。

### 3.5 启动 Web UI

```powershell
streamlit run streamlit_app.py
```

浏览器打开 `http://localhost:8501`，左侧创建会话，右侧开聊。

### 3.6 命令行版（可选）

```powershell
python -m scripts.chat_cli
```

---

## 4. 使用示例

打开 Web UI 后，可以直接试：

```
我想 5 月 1 号到 5 月 3 号从北京去西安玩，我喜欢历史和小吃，
预算 2000 元/人，请帮我规划行程。
```

Agent 会：
1. 调 `compute_distance` 算北京-西安距离 → 给出"建议高铁"
2. 调 `get_weather_forecast` 查西安 5/1~5/3 天气
3. 调 `search_attractions` 拉西安景点
4. 调 `lookup_culture` 拿西安文化简介
5. 整合成 3 天的 Markdown 行程

然后你可以追问，比如：

```
第二天太满了，把回民街挪到第一天晚上吧
```

Agent 会基于上下文**修改**之前的行程，而不是重新规划。

---

## 5. 工具一览

| 工具名 | 作用 | 参数 | 主要数据源 |
|---|---|---|---|
| `geocode_address` | 地名 → 经纬度 | `address`, `city?` | 高德优先 → OSM Nominatim |
| `get_weather_forecast` | 未来 1~7 天天气 | `location`, `days` | Open-Meteo（免费无 key） |
| `compute_distance` | 直线距离 + 交通方式建议 | `origin`, `destination` | 纯计算（haversine） |
| `search_attractions` | 城市景点 / POI 搜索 | `city`, `keyword?`, `max_results?` | 高德优先 → Wikipedia geosearch |
| `lookup_culture` | 历史 / 文化简介 | `place_or_topic` | 中文 Wikipedia → 英文 Wikipedia |
| `plan_route` | 真实道路路线规划 | `origin`, `destination`, `mode`, `city?` | 高德 driving/walking/transit → 回退 haversine |

每个工具内部都做了 **失败兜底**：永远不会把异常抛给 Agent，而是返回 `{"error": "..."}`，
让 LLM 自己看到错误后可以选择换路或如实告诉用户。

---

## 6. Agent 工作机制

### 6.1 图结构

```python
g = StateGraph(TravelState)
g.add_node("agent", _agent_node)   # LLM 决策节点
g.add_node("tools", ToolNode(...)) # 工具执行节点
g.add_edge(START, "agent")
g.add_conditional_edges(
    "agent",
    _should_continue,              # 看 last AIMessage 有没有 tool_calls
    {"tools": "tools", END: END},
)
g.add_edge("tools", "agent")       # 工具结果回到 agent，继续推理
```

这就是经典的 **ReAct loop**：Reason → Act → Observe → Reason → ...

### 6.2 记忆（Memory）

- **持久化**：`SqliteSaver(sqlite3.connect("./data/checkpoints.sqlite"))`
- **作用域**：以 `thread_id` 为单位，每个会话对应一个 thread
- **自动衔接**：每次 `invoke` / `stream` 时 LangGraph 会自动把这个 thread 的全部历史 messages 拼到 prompt 前面，无需我们手动做 ConversationBufferMemory 之类的事

### 6.3 上下文修改

因为 messages 是被 checkpointer 自动累积的，所以"用户改需求"在 Agent 看来就是
**对话历史里多了一句话**——LLM 自然会基于上下文做增量改动，无需任何特殊代码。

### 6.4 真·token 流式 + 用量统计

`src/agent.py` 中 `stream_agent_tokens()` 用 LangGraph 的**多 `stream_mode`** 能力同时订阅两种粒度：

```python
for mode, data in agent.stream(..., stream_mode=["messages", "updates"]):
    if mode == "messages":
        # AIMessageChunk：LLM 的 token 增量 → 前端打字机
    elif mode == "updates":
        # 节点级变更：工具调用、工具结果
```

外部迭代器统一 yield 出五种事件：`token` / `tool_call` / `tool_result` / `usage` / `done`，前端只认协议不关心底层。

**Token 用量聚合**：一轮对话在 ReAct 里可能触发多次 LLM 调用，`_extract_usage()` 会同时兼容两种上报渠道：

- `usage_metadata`（LangChain 0.3+ 标准，`ChatOpenAI` 默认走这里）
- `response_metadata.token_usage`（`ChatTongyi` 实际使用的渠道）

去重按 `request_id` —— 避免 messages 和 updates 两个通道把同一次响应的 usage 累加两遍。一轮结束时写入 `data/usage/{thread_id}.json` 并按 `PRICING` 表（`src/usage.py`）估算人民币费用。

### 6.5 LLM Provider 双轨

默认走 **DashScope OpenAI 兼容端点 + `ChatOpenAI`**（`LLM_PROVIDER=openai_compat`），原因：

- `langchain_community.ChatTongyi` 在 `streaming=True` + `tool_calls` 场景下有个
  [已知 bug](https://github.com/langchain-ai/langchain/issues)：
  `subtract_client_response` 做增量 tool_call 对齐时会抛 `IndexError: list index out of range`。
- `ChatOpenAI` 对 stream + tools 的实现非常成熟，搭配 DashScope 的 OpenAI 兼容接口，
  就能同时拿到"真 token 流式 + tool calling 稳定 + usage_metadata 标准"。

如需切回 ChatTongyi（比如做对照测试），把 `.env` 里 `LLM_PROVIDER=tongyi` 即可。**但不要再开 `streaming=True` + tool_calls**，否则就会踩那个 bug。

### 6.6 教学版：手写 tool node

`src/agent.py` 末尾保留了一份 `_manual_tool_node` 实现，与 `ToolNode` 等价但完全手写，
方便理解"工具调用是如何被分发执行的"。生产路径用 `ToolNode`（自带并发 + 错误兜底）。

---

## 7. 常见问题

**Q: 没有高德 key 能用吗？**
A: 能。地理编码、景点、路线都有免费数据源 fallback，质量稍差但完整可用。

**Q: 为什么用 DashScope 而不是 OpenAI？**
A: 这套项目是为国内环境设计的，DashScope 国内可直连且 Qwen 中文效果好。
如果想换 OpenAI / Claude，只改 `src/llm.py` 的工厂函数即可，其余代码无变化。

**Q: LLM_PROVIDER=openai_compat 和 tongyi 的区别？**
A: 两者都走 DashScope 后端 + 同一个 Key，只是 LangChain 侧用哪个 wrapper。
默认的 `openai_compat` 走 `ChatOpenAI` + DashScope 的 OpenAI 兼容端点
(`https://dashscope.aliyuncs.com/compatible-mode/v1`)，**流式 + tool_calls 稳定**，
能配合 `stream_usage=True` 在流式最后一 chunk 拿到 usage_metadata。
`tongyi` 走 `ChatTongyi` + dashscope SDK，**在 streaming=True + tool_calls 下有上游 bug**
（`IndexError` in `subtract_client_response`），保留仅作对照。

**Q: Token 用量在哪里看？**
A: Streamlit 侧边栏的 "📊 Token & 费用" 面板实时显示**当前会话**的累计 ↑输入 / ↓输出 / 人民币费用，
展开 "📈 逐轮明细" 能看最近 20 轮的明细表。原始数据在 `data/usage/{thread_id}.json`，方便做跨会话分析。

**Q: 定价表不准怎么办？**
A: 直接改 `src/usage.py` 的 `PRICING` 字典。它按模型名前缀匹配（`qwen-max` / `qwen-plus` / ...），
没匹配到的会 fallback 到 `_default`。DashScope 官方调价时更新这张表即可。

**Q: 多个会话的数据存在哪？**
A: `data/checkpoints.sqlite`（LangGraph checkpoint）+ `data/sessions.json`（前端用的标题等元信息）。
删除会话会同时清掉这两边的数据。

**Q: Agent 死循环怎么办？**
A: `recursion_limit = MAX_TOOL_ITERATIONS * 2 + 4`，达到上限会抛 `GraphRecursionError`。
默认 10 步通常足够，复杂场景可在 `.env` 调大。

**Q: 想加新工具？**
A:
1. 在 `src/tools/` 下新建文件，写一个 `@tool` 装饰的函数；
2. 在 `src/tools/__init__.py` 的 `ALL_TOOLS` 里把它加上；
3. 重启即可，无需改 Agent 代码。

---

## 8. 与本仓库另外两个子项目的关系

| 子项目 | 类型 | 关键技术 |
|---|---|---|
| `llamaIndex_rag/` | RAG 知识库 | LlamaIndex + Chroma + DashScope |
| `langchain_rag/` | RAG 知识库 | LangChain + Chroma + DashScope |
| **`langChain_langGraph_agent/`**（本项目） | **Agent + 工具调用** | **LangChain + LangGraph + 自定义 Tools** |

三者覆盖了"问知识库" → "调外部世界"两条主流的 LLM 应用范式，配合食用更香。

---

## 9. License

仅作学习用途，欢迎 fork 改造为你自己的领域 Agent（医疗咨询、求职规划、做菜助理…… 都可以）。
