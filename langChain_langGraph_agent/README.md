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
| 🧠 记忆 | LangGraph **SqliteSaver checkpointer**，跨进程也能续聊 |
| 🔧 可观测 | 工具调用 trace 实时展开（参数 + 返回 JSON） |
| 🔁 上下文修改 | 用户随时改需求（"再加一天"、"去掉博物馆"），Agent 会基于历史增量调整 |
| 🇨🇳 中文 LLM | 阿里云百炼 DashScope Qwen 系列（`qwen-plus` / `qwen-max` / ...） |

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
│   ├── llm.py                      ← LLM 工厂（ChatTongyi）
│   ├── prompts.py                  ← system prompt
│   ├── state.py                    ← LangGraph TravelState
│   ├── agent.py                    ← ⭐ StateGraph 编排 + checkpointer
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

### 6.4 教学版：手写 tool node

`src/agent.py` 末尾保留了一份 `_manual_tool_node` 实现，与 `ToolNode` 等价但完全手写，
方便理解"工具调用是如何被分发执行的"。生产路径用 `ToolNode`（自带并发 + 错误兜底）。

---

## 7. 常见问题

**Q: 没有高德 key 能用吗？**
A: 能。地理编码、景点、路线都有免费数据源 fallback，质量稍差但完整可用。

**Q: 为什么用 DashScope 而不是 OpenAI？**
A: 这套项目是为国内环境设计的，DashScope 国内可直连且 Qwen 中文效果好。
如果想换 OpenAI / Claude，只改 `src/llm.py` 的工厂函数即可，其余代码无变化。

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
