# LangChain + LangGraph Agent 学习指南

> 这是一份**学习路径文档**，目标读者是想把这个智能旅游规划 Agent 项目读懂、改造、二次开发的工程师。
>
> 它**不是** README 的复述，而是按"为什么这样写"的顺序，把整个项目拆开讲一遍，并把每一处涉及的 LLM/Agent 知识点都标注出来。
>
> 阅读建议：
> 1. 先把**学习目标（§1）**和**项目地图（§2）**过一遍，建立全局
> 2. 按**推荐阅读顺序**逐模块走读（§4）
> 3. 每读完一个模块，去做一次配套的**实战练习**（§9）
> 4. 全部读完后，回头看**§10 与 RAG 项目的对比**

---

## 1. 学习目标

读完本文档你应该能回答以下问题：

1. **Agent 和 Chain 有什么本质区别？** 为什么旅游规划这种任务必须用 Agent，不能用 Chain？
2. **ReAct（Reason + Act）** 范式的核心是什么？图里"agent → tools → agent"为什么必须形成循环？
3. **LangGraph 的 StateGraph、节点、条件边、checkpointer** 各解决什么问题？为什么不直接用 LangChain 的 `AgentExecutor`？
4. **`@tool` 装饰器** 是怎么把一个 Python 函数变成 LLM 可见的 schema 的？LLM 又是怎么决定"调哪个 + 传什么参数"的？
5. **`bind_tools`** 内部做了什么？ChatTongyi（Qwen）和 ChatOpenAI 在 tool calling 协议上是不是兼容的？
6. **持久化记忆（SqliteSaver）** 是怎么按 `thread_id` 做到"每个会话独立、跨进程不丢"的？
7. **`add_messages` reducer** 解决了什么并发问题？为什么 State 字段要写成 `Annotated[list, add_messages]`？
8. **工具失败（404 / 超时 / SSL 抖动）** 时，为什么我们一律返回 `{"error": ...}` 而不是抛异常？这条原则对 LLM 推理有什么影响？
9. **多源 fallback**（高德 → Open-Meteo → OSM）的设计权衡是什么？怎么避免"北京"被解析到重庆某村镇？
10. **Streaming 的 `stream_mode="updates"`** 和 `stream_mode="values"` 区别在哪？为什么前端用 updates 更省带宽？

如果上面任何一题答不出来，就值得在对应章节多停留一会。

---

## 2. 项目地图

```
langChain_langGraph_agent/
├── .env                    # 实际配置（DashScope Key、模型名、可选高德 Key）
├── .env.example            # 模板
├── requirements.txt        # langchain 0.3 / langgraph 0.2 / dashscope / streamlit ...
├── README.md               # 给"使用者"看的：怎么跑起来
├── LEARNING.md             # 本文，给"学习者"看的：为什么这样设计
│
├── data/
│   ├── checkpoints.sqlite  # ⭐ LangGraph checkpointer：跨进程持久化的对话状态
│   └── sessions.json       # 前端用的会话标题/创建时间元信息
│
├── src/                    # 核心代码
│   ├── config.py           # 1) 配置中心（pydantic-settings 读 .env）
│   ├── llm.py              # 2) LLM 工厂（ChatTongyi 单例缓存）
│   ├── prompts.py          # 3) system prompt（含工作流强约束）
│   ├── state.py            # 4) ⭐ LangGraph State 定义（add_messages reducer）
│   ├── agent.py            # 5) ⭐⭐ StateGraph 编排：手写 ReAct loop + checkpointer
│   └── tools/              # 6) ⭐ 6 个自定义工具
│       ├── _http.py        共用 HTTP（指数退避重试 + 浏览器 UA）
│       ├── geocode.py      地址 → 经纬度（高德 → Open-Meteo → OSM 三档 fallback）
│       ├── weather.py      Open-Meteo 未来 1~7 天天气
│       ├── distance.py     haversine 距离 + 智能交通方式建议（纯本地）
│       ├── attractions.py  高德 POI / Wikipedia geosearch
│       ├── culture.py      Wikipedia 中文/英文文化简介
│       └── route.py        高德路线规划 / 回退直线估算
│
├── scripts/                # 命令行入口
│   ├── diagnose.py         ⭐ 一键体检：配置 → LLM → 6 个工具
│   └── chat_cli.py         命令行版对话（无浏览器场景）
│
└── streamlit_app.py        Web UI（多会话 + 工具 trace 实时展开）
```

**推荐阅读顺序**（也是各文件的依赖顺序）：

```
config.py ─→ llm.py
                │
                ↓
   tools/*.py（6 个独立工具，互不依赖）
                │
                ↓
       prompts.py + state.py
                │
                ↓
              agent.py（核心编排）
                │
                ↓
   streamlit_app.py / scripts/chat_cli.py
```

---

## 3. 第一性原理：从 Chain 到 Agent

### 3.1 Chain（链）的本质

**Chain** 是**确定性**的步骤序列：

```
A → B → C → D
```

- 每一步都是事先编排好的（开发者写死的）
- 每一步的输入是上一步的输出
- 不需要 LLM "决策"，LLM 只是其中一两个步骤的"翻译机"

适合：知识库问答（RAG）、文本分类、摘要、SQL 生成 ……
**这些任务有一个共性：流程可以被穷举**。

### 3.2 Agent 的本质

**Agent** 是**LLM 自己决定下一步做什么**的循环：

```
LLM 看着上下文 → 决定调哪个工具（或不调） → 拿到结果 → 再决定 …… → 觉得够了就给最终答案
```

它把"流程编排权"交给了 LLM。开发者只负责：
1. **给工具**（让 LLM 有"手脚"可以触达外部世界）
2. **给目标**（system prompt 里告诉它要做什么）
3. **给约束**（最多循环几次、每次必须用中文 ……）

适合：旅游规划、客服、代码 review、深度搜索 ……
**这些任务的共性：不可能事先穷举所有流程**——用户问"北京玩 3 天"和"成都玩 5 天"需要的工具调用次数、顺序、组合都不一样。

### 3.3 旅游规划为什么必须用 Agent？

来看一句真实用户输入：

> "我想周末从北京去天津玩 1 天，告诉我天气怎么样、距离多远，并推荐 3 个景点"

如果用 Chain，你得这样写死：

```python
def chain(query):
    extract_cities(query)               # 提取出发地、目的地
    weather_info = get_weather(...)     # 查天气
    distance_info = get_distance(...)   # 算距离
    attractions = search_attractions(...) # 找景点
    return llm.invoke(format_prompt(...))
```

但下一句用户改成"我只想知道距离"，整个 chain 就崩了——你得为每种"组合"写一条 chain。

Agent 直接交给 LLM 决定：
- 看到"天气" → 自己调 `get_weather_forecast`
- 看到"距离" → 自己调 `compute_distance`
- 看到"景点" → 自己调 `search_attractions`
- **没看到的就不调**

这就是 Agent 的真正价值：**把组合爆炸的复杂度从开发者转移给 LLM 的推理能力**。

> 🧠 **知识点**：Agent vs Chain 的本质区别 = "流程是事先编排还是 LLM 实时决策"。

### 3.4 ReAct 框架（本项目的理论基础）

ReAct 来自论文 [ReAct: Synergizing Reasoning and Acting in Language Models (2022)](https://arxiv.org/abs/2210.03629)。

它的核心循环是：

```
Thought（思考）→ Action（行动）→ Observation（观察）→ Thought → ...
```

翻译到 LLM 的真实输出：

```
[Thought] 用户想去天津，我先查天气
[Action]  调用 get_weather_forecast(location="天津", days=1)
[Observation] {"forecast": [{...}]}        ← 工具返回
[Thought] 还需要距离，再调 compute_distance
[Action]  调用 compute_distance(origin="北京", destination="天津")
[Observation] {"distance_km": 137, ...}
[Thought] 信息够了，可以给出最终答案
[最终回答] "北京到天津约 137 km，明天天津阴天 ……"
```

现代的 OpenAI / Anthropic / Qwen 都把这个循环用 **tool calling** 协议封装好了——LLM 不再用 Thought/Action 这种文本格式，而是直接吐出**结构化的 JSON 字段** `tool_calls=[{name, args}]`，框架负责执行并把结果以 `ToolMessage` 形式喂回去。

> 🧠 **知识点**：ReAct = Reason + Act 循环；现代实现是通过 LLM 的 **structured tool calling** 协议而不是文本解析。

### 3.5 LangGraph 是什么？

LangGraph 是 LangChain 团队为 **"循环、分叉、有状态"** 的 LLM 应用做的 DAG/状态机框架。

它的世界观：

| 概念 | 在本项目的对应物 |
|---|---|
| **State**（共享状态） | `TravelState`（messages 列表） |
| **Node**（节点） | `agent` 节点（LLM 决策） + `tools` 节点（工具执行） |
| **Edge**（边） | `START → agent`、`tools → agent` |
| **Conditional Edge**（条件边） | `agent → ?`：有 tool_calls 走 `tools`，没有走 `END` |
| **Checkpointer**（检查点） | `SqliteSaver`：每个节点跑完后把 state 存到 sqlite |

为什么不用更高层的 `AgentExecutor`（LangChain 老接口）？
- AgentExecutor 是黑盒，看不到中间步骤
- 不支持多分支、并行、回滚
- 状态管理简陋（无 checkpointer）

LangGraph 把"图"显式暴露出来，让我们能精细地控制：节点之间走哪条边、什么条件下回头、状态怎么合并。

> 🧠 **知识点**：LangGraph = 给 LLM 应用用的**状态机/有向图**框架；区别于 LangChain Chain 的 DAG 是**它允许循环（cycle）**，这是实现 ReAct 必需的。

---

## 4. 模块走读

按依赖顺序，逐文件讲。

### 4.1 `src/config.py` —— 配置中心

**这一节涉及的知识点：**
- pydantic-settings 自动从 `.env` 读取并做类型校验
- 派生属性（`@property`）封装"路径要保证存在"这种横切逻辑
- 启动前 fail-fast：`assert_llm_ready()` 在拿不到 API Key 时直接抛错

```python
class Settings(BaseSettings):
    DASHSCOPE_API_KEY: str = Field(default="")
    LLM_MODEL: str = Field(default="qwen-plus")
    AMAP_API_KEY: str = Field(default="")
    MAX_TOOL_ITERATIONS: int = Field(default=10)
    ...
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)
```

为什么单独抽配置中心？
- **避免散落**：不要在 6 个工具文件里 `os.getenv("AMAP_API_KEY")` 散写
- **类型安全**：拼错字段名 IDE 立刻报错，而 dict access 要等运行时
- **测试友好**：单测可以临时覆盖某个字段

**关键设计：`MAX_TOOL_ITERATIONS`**

LangGraph 默认有 `recursion_limit`（默认 25），但旅游规划只需 5~8 次工具调用就够。这里设 10 次（agent.py 里乘 2 是因为每次 ReAct 实际会走 agent → tools 两个节点 = 2 步）。

> 🧠 **知识点**：递归上限是 Agent 必须设置的"安全带"，否则 LLM 在 tool 调用错误时可能反复重试形成死循环烧 token。

---

### 4.2 `src/llm.py` —— LLM 工厂

**这一节涉及的知识点：**
- LangChain 的 `ChatTongyi`（Qwen 的 LangChain Wrapper）
- DashScope SDK 的认证：通过环境变量 `DASHSCOPE_API_KEY`
- `lru_cache` 让多次调用不重复构造对象（轻量单例）

```python
@lru_cache(maxsize=4)
def get_llm(streaming: bool = False, temperature: float | None = None):
    return ChatTongyi(
        model=settings.LLM_MODEL,
        dashscope_api_key=settings.DASHSCOPE_API_KEY,
        temperature=settings.LLM_TEMPERATURE,
        streaming=streaming,
    )
```

**为什么 ChatTongyi 能用 OpenAI 风格的 tool calling？**

LangChain 的 `BaseChatModel.bind_tools()` 是**统一抽象**——所有 ChatModel 都按 OpenAI 的 tool 协议把工具转成 LLM 可见的 schema，发给后端时再由各家 wrapper 翻译成自己平台的 API 格式。

DashScope 自己的 Generation API 已经原生支持 OpenAI 风格的 tools 字段（Qwen 系列 2.5+ 都支持），所以 LangChain 几乎是直传。

> 🧠 **知识点**：LangChain ChatModel 抽象的核心价值就是 **tool calling 协议统一**——你换 OpenAI / Claude / Qwen，业务代码不用改一行。

---

### 4.3 `src/tools/` —— 6 个自定义工具

这是**整个项目最值得反复读**的部分。Agent 的能力 = LLM 的推理能力 × 工具的覆盖面。

#### 4.3.1 `@tool` 装饰器：函数 → LLM 可见的 schema

看 `src/tools/weather.py`：

```python
@tool("get_weather_forecast", return_direct=False)
def get_weather_forecast(
    location: str,
    days: int = 3,
) -> dict[str, Any]:
    """查询某个地点未来 N 天的天气预报（数据源：Open-Meteo，免费）。

    Args:
        location: 地名或地址，例如 "成都"、"杭州西湖"。
        days: 预报天数，1~7。默认 3 天。

    Returns:
        含 location / forecast 的字典：...
    """
    ...
```

`@tool` 装饰器会自动从这段函数生成一个 JSON Schema：

```json
{
  "name": "get_weather_forecast",
  "description": "查询某个地点未来 N 天的天气预报（数据源：Open-Meteo，免费）。",
  "parameters": {
    "type": "object",
    "properties": {
      "location": {"type": "string", "description": "地名或地址，例如 \"成都\"..."},
      "days": {"type": "integer", "description": "预报天数，1~7。默认 3 天。"}
    },
    "required": ["location"]
  }
}
```

LLM 拿到这段 schema 后，会**自动**学会"哪些参数是必填的、什么时候该用这个工具、参数取什么值"。所以：

> ⚠️ **极其重要**：函数的 **docstring + 参数 type hint** 不是给开发者看的，是给 **LLM** 看的。一定要写清楚！
> - 第一句要点明"做什么、用什么数据源"
> - 每个参数都要给 1~2 个具体例子
> - Returns 部分写返回结构（即使 LLM 不一定全用，但失败时它能识别错误）

> 🧠 **知识点**：Tool schema 是 LLM 决策的**唯一**输入。Agent 表现差 80% 是 schema 写得不清楚导致的。

#### 4.3.2 工具内部互相调用

`weather.py` 里：

```python
from .geocode import geocode_address as _geocode_tool

def get_weather_forecast(location, days):
    geo = _geocode_tool.invoke({"address": location})  # 直接调，不经过 LLM
    ...
```

注意我们没有让 LLM 串联"先 geocode 再 weather"——而是 weather 工具内部自己做。这是**工具粒度**的设计取舍：

| 选择 | 优点 | 缺点 |
|---|---|---|
| 让 LLM 自己串联 | 灵活，能复用 | 多一次往返 + token 消耗 |
| 工具内部串联 | 快、节省 token | 工具变复杂 |

**经验法则**：如果 A 几乎总是为 B 服务（90%+ 概率），就把 A 内嵌到 B；否则暴露给 LLM 自己决定。

> 🧠 **知识点**：工具粒度设计 ≠ 越细越好。**适当组合**能显著提升 Agent 的速度和稳定性。

#### 4.3.3 失败哲学：永不抛异常

看 `_http.py`：

```python
def http_get(...) -> dict[str, Any]:
    for attempt in range(retries + 1):
        try:
            ...
            return {"ok": True, "data": ...}
        except ...:
            last_error = ...
    return {"ok": False, "error": last_error}
```

**Tool 永远不要把异常抛回给 LangGraph**，原因有三：

1. **LangGraph 默认 `handle_tool_errors=True`** 会把异常转成 ToolMessage 兜底，但消息体是堆栈，LLM 看不懂
2. **可控的 `{"error": "..."}` 让 LLM 能"看懂错误"**：比如返回 `{"error": "超时"}`，LLM 会换路径或如实告诉用户
3. **方便单测**：异常很难测，dict 一行 assert 就完事

> 🧠 **知识点**：Agent 工程的"软鲁棒性"原则——**失败不抛异常，而是返回结构化错误**。

#### 4.3.4 多源 fallback：以 geocode 为例

`geocode.py` 的设计：

```
高德地图 (有 key 才用，中文最准)
   ↓ 失败
Open-Meteo geocoding (免费无 key、对 UA 限制宽)
   ↓ 失败
OSM Nominatim (最后兜底，部分网络下会 403)
```

为什么不直接只用一个？
- 高德要 key，但很多读者懒得申请
- Open-Meteo 不识别"北京天安门"这种长地名，但识别"北京"
- OSM 在中国网络下偶尔被限速

**关键陷阱**：`_smart_keywords` 函数

```python
def _smart_keywords(address: str) -> list[str]:
    candidates = []
    # 1) 优先去掉景点后缀的 stem，让"北京天安门" → "北京"
    for suffix in _LANDMARK_SUFFIXES:
        if address.endswith(suffix):
            candidates.append(address[:-len(suffix)])
            break
    # 2) 原文（如"成都"短行政区名直接命中）
    candidates.append(address)
    # 3) 行政区后缀（"北京市" → "北京"）
    ...
```

为什么要写这个？因为 Open-Meteo 的 geocoding 是按"行政区/地名"匹配的，对**景点级地名**支持不好。我们要把"北京天安门"先剥成"北京"再查，否则 API 会返回空。

**第二个陷阱**：`_score` 排序

```python
def _score(it: dict) -> tuple[int, int, int]:
    cc_match = 1 if it.get("country_code") == "CN" else 0
    rank = _RANK.get(it.get("feature_code", ""), 0)  # PPLC=4, PPLA=3
    pop = int(it.get("population") or 0)
    return (cc_match, rank, pop)
```

直接用 Open-Meteo 返回的第一条结果，"北京"会被解析到重庆某个叫"北京镇"的村子（feature_code=PPL，普通居民点）。我们用三层排序：
1. 先按 country=中国
2. 再按 feature_code 等级（首都 > 省会 > 地级市 > 村镇）
3. 最后按 population

> 🧠 **知识点**：多源 fallback 的核心难点不是"调多少个 API"，而是**结果质量怎么排序选优**。

#### 4.3.5 工具并行执行

ToolNode 内部对一次 AIMessage 里的多个 tool_calls 是**并发执行**的：

```python
# 来自 langgraph.prebuilt.ToolNode 内部
with ThreadPoolExecutor() as ex:
    futures = [ex.submit(self._run_one, tc) for tc in tool_calls]
```

意味着：如果 LLM 一次决定调 weather + distance + attractions 三个工具，它们是**同时**发出 HTTP 请求，而不是串行。这是为什么我们要**鼓励** LLM 一次决定多个工具调用（在 system prompt 里写"工具调用尽量并行"）。

> 🧠 **知识点**：Tool 并行 = Agent 的性能优化关键，可以节省 N-1 次往返时间。

---

### 4.4 `src/prompts.py` —— System Prompt 工程

很多人以为 Agent 的 system prompt 就是"你是一个助手"——大错特错。Agent 的 prompt 实际承担**工作流约束**的职责：

```python
SYSTEM_PROMPT = """你是一名资深中文旅游规划师...

# 你拥有的工具
1. `geocode_address(address, city?)` —— 把地名/地址转成经纬度
...

# 工作流程（必须遵循）
1. **理解需求**：先确认用户的出发地、目的地、出行天数、人数、预算...
2. **收集事实**：对涉及到的城市，至少调一次 get_weather_forecast / search_attractions ...
3. **整合方案**：把工具返回的事实整合成天-by-天的行程，每一天给出...
4. **持续修正**：用户随时会改需求，你要基于已有 messages 增量调整，不要重头再来。

# 重要约束
- 不要编造工具返回的数据
- 工具调用尽量并行（同一轮可以 a、b、c 三个工具一起发）
- 用户没问的城市，不要调工具
- 拿不准时主动反问
"""
```

每一段都对应解决一个具体问题：

| Prompt 段落 | 解决什么 |
|---|---|
| "你拥有的工具" 列出 schema | 加深 LLM 对工具的"记忆"（虽然 bind_tools 已经传过 schema） |
| "工作流程" | 防止 LLM 不调工具直接编 |
| "工具调用尽量并行" | 性能优化 |
| "用户没问的城市，不要调工具" | 节省 token |
| "拿不准时主动反问" | 防止 LLM 自行假设用户需求 |

> 🧠 **知识点**：Agent prompt 的核心 = **工作流约束 + 失败兜底语义**，不是"扮演什么角色"。

---

### 4.5 `src/state.py` —— LangGraph State

只有 13 行代码，但概念非常重要：

```python
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class TravelState(TypedDict):
    messages: Annotated[list, add_messages]
```

#### 4.5.1 `TypedDict` vs `Pydantic Model`

LangGraph 支持两种 State 定义：
- `TypedDict`：轻量、零开销，适合纯字段
- `BaseModel`：有 validation，适合复杂结构

本项目只有 messages 一个字段，用 TypedDict 足够。

#### 4.5.2 `Annotated[list, add_messages]` 的魔法

这一句话是 LangGraph 的"灵魂语法"。它告诉 LangGraph：

> 当多个节点同时往 `messages` 里写入时，**不要覆盖**，而是用 `add_messages` 函数**合并**。

`add_messages` 的行为：
- 输入是 `list + list` → 输出是 `list1 + list2`（追加）
- 自动去重（按 message id）
- 自动处理 ToolMessage 的 tool_call_id 关联

如果不写 `Annotated`，默认行为是覆盖，那么 tools 节点写完 messages，agent 节点的历史就丢了。

> 🧠 **知识点**：LangGraph 的 State 字段 = **数据 + reducer**。Reducer 决定"多个节点写同一字段时怎么合并"，是并发安全的关键。

#### 4.5.3 为什么不需要 `user_profile / itinerary` 字段？

很多教程会让你在 State 里塞 `current_itinerary`、`preferences`、`user_profile`……但本项目**故意只放 messages**。

原因：**只要信息出现在对话历史里，LLM 自然就能基于上下文使用它**。再加额外字段反而要写很多"提取-存储-注入"的胶水代码，且容易和 messages 不一致。

只有在以下情况才值得加结构化 state：
- 信息**会被多个节点机械地读写**（比如有个独立的"行程评估"节点要打分）
- 信息**LLM 不能可靠地从历史里复原**（比如非常长的对话，老消息会被 LLM 忽略）

> 🧠 **知识点**：State 设计的最简原则 = "messages 能装下的，就不要单独建字段"。

---

### 4.6 `src/agent.py` —— ⭐⭐ 核心编排

这是整个项目最重要的文件。慢慢读。

#### 4.6.1 图结构

```python
g = StateGraph(TravelState)
g.add_node("agent", _agent_node)        # LLM 决策
g.add_node("tools", _tool_node)         # 工具执行

g.add_edge(START, "agent")              # 入口必经 agent
g.add_conditional_edges(
    "agent",
    _should_continue,                    # 路由函数
    {"tools": "tools", END: END},        # 路由结果映射
)
g.add_edge("tools", "agent")            # 工具结果回到 agent
```

画成图：

```
┌────────┐
│ START  │
└───┬────┘
    ▼
┌────────┐   _should_continue == "tools"   ┌────────┐
│ agent  │ ──────────────────────────────► │ tools  │
└───┬────┘                                  └───┬────┘
    │ _should_continue == END                  │
    ▼                                          │
┌────────┐                                     │
│  END   │                                     │
└────────┘                                     │
    ▲                                          │
    │←──── 工具结果回到 agent ←─────────────────┘
```

这个图的**核心特点是有 cycle**（agent ↔ tools 形成循环）。这是**普通 LangChain Chain 做不到的**，也是 LangGraph 存在的根本理由。

#### 4.6.2 `_agent_node`：LLM 决策节点

```python
def _agent_node(state: TravelState) -> dict[str, Any]:
    llm = get_llm()
    llm_with_tools = llm.bind_tools(ALL_TOOLS)  # ⭐ 关键

    msgs = state["messages"]
    if not msgs or not isinstance(msgs[0], SystemMessage):
        msgs = [SystemMessage(content=SYSTEM_PROMPT), *msgs]

    response = llm_with_tools.invoke(msgs)
    return {"messages": [response]}
```

要点：

1. **`bind_tools(ALL_TOOLS)`**：把 6 个工具的 JSON schema 注入到 LLM 调用中。LLM 看到 schema 后，可能返回：
   - `AIMessage(content="...", tool_calls=[])` —— 不调工具，直接给答案
   - `AIMessage(content="", tool_calls=[{name: "get_weather_forecast", args: {...}}])` —— 调工具
2. **System prompt 只在第一次塞进去**：checkpointer 累积的 messages 不会自带 system，所以每次 agent_node 跑都要补上。
3. **返回格式 `{"messages": [response]}`**：因为 State 的 messages 字段配了 `add_messages` reducer，这里返回的 list 会被**追加**而非覆盖。

> 🧠 **知识点**：节点函数的返回值是 **state 的 partial update**，由 reducer 合并。

#### 4.6.3 `_tool_node = ToolNode(ALL_TOOLS)`

LangGraph 官方 `ToolNode` 内部：

```python
def __call__(self, state):
    last_msg = state["messages"][-1]
    if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
        return {}
    results = []
    for tool_call in last_msg.tool_calls:
        tool = self.tools_by_name[tool_call["name"]]
        result = tool.invoke(tool_call["args"])  # 真正执行
        results.append(ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"],
            name=tool_call["name"],
        ))
    return {"messages": results}
```

我们也可以**完全手写**，看 `agent.py` 末尾的 `_manual_tool_node`——和 ToolNode 等价，但展示了"工具调用是如何被分发执行的"全过程。学习时强烈建议读一遍。

#### 4.6.4 `_should_continue`：条件边路由函数

```python
def _should_continue(state):
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return END
```

逻辑就一句：**最新一条 AIMessage 还有 tool_calls** → 继续调工具；否则结束。

`add_conditional_edges` 的路由表 `{"tools": "tools", END: END}` 把这个返回值映射到具体节点。

#### 4.6.5 Checkpointer：跨进程持久化

```python
@contextmanager
def open_agent(persistent=True):
    if persistent:
        conn = sqlite3.connect(str(settings.checkpoint_db_path), check_same_thread=False)
        try:
            yield _build_graph(SqliteSaver(conn))
        finally:
            conn.close()
    else:
        yield _build_graph(MemorySaver())
```

**MemorySaver**：进程内存，重启就丢，单测用。
**SqliteSaver**：本地 sqlite，跨进程持久化，生产用。

每次 `agent.invoke(...)` 时，LangGraph 会做：

1. 用 `config["configurable"]["thread_id"]` 当 key，从 sqlite 加载历史 state
2. 把这次的输入合并进去
3. 跑节点，每个节点跑完都 dump 一次新 state 到 sqlite
4. 返回最终 state

所以两次调用之间不需要我们自己维护"对话历史"——LangGraph 全帮你管了。

**调用时怎么指定 thread_id**：

```python
config = {"configurable": {"thread_id": "user-42-conv-1"}}
agent.invoke({"messages": [HumanMessage("...")]}, config=config)
```

> 🧠 **知识点**：LangGraph checkpointer = **类 Git 的状态快照**。每个节点结束都自动 commit。Thread_id 类似 Git branch。

#### 4.6.6 `recursion_limit` 安全带

```python
config = {
    "configurable": {"thread_id": thread_id},
    "recursion_limit": settings.MAX_TOOL_ITERATIONS * 2 + 4
}
```

`recursion_limit` 是 LangGraph 防死循环的硬限制——节点跑了这么多步还没到 END 就抛 `GraphRecursionError`。乘 2 是因为 ReAct 循环每次走 agent + tools 两个节点。

#### 4.6.7 流式接口 `stream_agent`

```python
def stream_agent(user_message, thread_id, persistent=True):
    config = {...}
    with open_agent(persistent) as agent:
        for event in agent.stream(
            {"messages": [HumanMessage(content=user_message)]},
            config=config,
            stream_mode="updates",
        ):
            yield event
```

`stream_mode` 三种值：

| mode | 每个 event 内容 | 适用场景 |
|---|---|---|
| `"values"` | 整个 state 的最新值 | 调试，能看到全量 |
| `"updates"` | 每个节点新增的字段 | **前端**，省带宽 |
| `"messages"` | 逐 token 的 AIMessage stream | 真正的"打字机"流式输出 |

本项目前端用 `"updates"`：每次只拿到节点新增的 message（一个 AIMessage 或几个 ToolMessage），渲染快、不用 diff。

> 🧠 **知识点**：LangGraph 的 streaming 三种粒度，按需选；不是流式 token 就是"打字机"，updates 也是流式。

---

### 4.7 `streamlit_app.py` —— 前端 UI

要点（不是教 Streamlit 用法，是讲为什么这么组织）：

1. **会话标题元信息单独存 JSON**：`data/sessions.json` 只记 title / created_at；真实对话历史在 `checkpoints.sqlite`。两边解耦——删除会话时要同时清 JSON 元信息和 sqlite checkpoint。

2. **`get_history(thread_id)` 直接读 LangGraph 的 state**：

   ```python
   def get_history(thread_id):
       with open_agent(persistent=True) as agent:
           snap = agent.get_state(config)
           return snap.values.get("messages", [])
   ```
   
   注意我们没有自己维护一份"对话历史"——直接从 checkpointer 读，这样和真实状态永远一致。

3. **Trace 实时渲染**：消费 `stream_agent` 的 event 流时，分别处理 AIMessage（看 tool_calls）和 ToolMessage（看返回 JSON），分别画到 trace expander 和主回答 box。

4. **`st.rerun()` 在一轮完成后**：把"实时占位"的 placeholder 清掉，让历史对话以静态形式重排。这是 Streamlit 流式 UI 的常见模式。

---

### 4.8 `scripts/diagnose.py` —— 体检脚本

这种脚本在生产 Agent 项目里**强烈建议每个都有**。因为：

1. Agent 涉及外部依赖太多：LLM API、N 个 tool API
2. 任何一个失败都会让 Agent 退化或崩溃
3. 如果不分层测试，用户看到的只是"为什么 Agent 答非所问"

我们的 `diagnose.py` 按依赖顺序逐项验证：

```
配置 → LLM → 6 个工具
```

任何一项失败立即给出可操作的提示（例如"未配置 AMAP_API_KEY"）。

---

## 5. Tool Calling 协议深入

这是一个值得单独成章的话题，因为它是 Agent 工作的**通信协议**。

### 5.1 一次完整的 tool call 报文

LLM 看到 user message + tool schema 后，决定调用工具，返回的 AIMessage 大概长这样（伪 JSON）：

```json
{
  "role": "assistant",
  "content": "",
  "tool_calls": [
    {
      "id": "call_abc123",
      "name": "get_weather_forecast",
      "args": {"location": "成都", "days": 3}
    }
  ]
}
```

注意：
- **`content` 通常是空字符串**：LLM 决定调工具时不输出文本
- **`id` 是关键**：后续的 ToolMessage 必须用同一 id 关联，否则 LLM 不知道哪个结果对应哪个调用
- **`args` 是已经解析好的 dict**：DashScope/OpenAI 把 LLM 的 JSON 输出解析成结构化字段了

### 5.2 ToolNode 的执行报文

```json
{
  "role": "tool",
  "tool_call_id": "call_abc123",
  "name": "get_weather_forecast",
  "content": "{\"location\": \"成都\", \"forecast\": [...]}"
}
```

`content` **必须是字符串**——这就是为什么我们在工具里返回 dict，但 ToolNode 内部会 `str(result)`。

### 5.3 LLM 怎么"看懂"工具结果

LLM 拿到 ToolMessage 后，把它的 content（一段 JSON 字符串）当成普通文本读。所以**返回结构清晰、字段名有自描述性**对 LLM 理解至关重要：

| 好的字段名 | 不好的字段名 |
|---|---|
| `forecast: [...]` | `data: [...]` |
| `transport_suggestion.primary` | `t.p` |
| `error: "请求超时"` | `error: 1001` |

> 🧠 **知识点**：Tool 返回 = 给 LLM 的"输入文本"，命名清晰、结构扁平、错误友好这三条比性能更重要。

---

## 6. 上下文与记忆机制深入

### 6.1 Token 累积 vs 记忆

每一轮对话，LangGraph 都会把这个 thread 的**全部历史 messages** 拼到 prompt 前面。所以：

- 第 1 轮：1 user message
- 第 2 轮：1 user + 1 ai + 1 user
- 第 5 轮：…… 累积 9 条
- 第 10 轮：累积 19 条 + 中间还有 ToolMessage（每次工具调用 1 个）

很快就到 context limit。

### 6.2 实际生产中怎么解决？

本项目没做（学习版），但生产应该考虑：

1. **截断**：只保留最近 N 条 + system prompt
2. **摘要**：超过阈值时，用一次 LLM 调用把老对话浓缩成摘要 message（langchain_rag 的 condense 思路）
3. **持久化用户画像**：把"用户偏好亲子游"这类长期事实存到独立字段（不靠 messages 携带）

LangGraph 提供了 `pre_model_hook` / `post_model_hook` 可以接管这种处理。

### 6.3 修改上下文

用户说"再加一天"，Agent 怎么知道"再加"指的是哪个行程？答案就在 messages 历史里。

LLM 上下文窗口里能看到：
- 它之前给出的 3 天行程方案
- 用户的"再加一天"

它会自己推断"再加 = 在原 3 天基础上加 1 天 = 4 天行程"。

**关键**：我们没有写任何"识别用户改需求"的代码——上下文做了所有事情。

---

## 7. 失败处理与可观测性

### 7.1 三层防御

| 层 | 失败处理方式 |
|---|---|
| HTTP 层（`_http.py`） | 指数退避重试 N 次；最终把异常吃掉，返回 `{"ok": False, "error": ...}` |
| 工具层（`weather.py` 等） | 不抛异常，返回 `{"error": "..."}` 让 LLM 看到 |
| Agent 层（`ToolNode`） | `handle_tool_errors=True`：万一工具真崩了，转成 ToolMessage 兜底 |

### 7.2 可观测性

调试 Agent 最痛苦的是"它为什么这么决定"。本项目用三个手段：

1. **`stream_mode="updates"`**：把每个节点的输出都拿出来看
2. **Streamlit trace expander**：实时显示每次调了哪个工具、传了什么参数、返回了什么
3. **`scripts/chat_cli.py --no-trace`**：CLI 调试时可关 trace 简洁输出

更专业可以接 LangSmith（LangChain 官方的 trace 平台），免费版够用。

> 🧠 **知识点**：Agent 项目的 **可观测性 = 一等公民**。没有 trace，问题永远定位不到。

---

## 8. 关键概念总结表

| 概念 | 出现位置 | 一句话总结 |
|---|---|---|
| **Chain vs Agent** | §3.1, §3.2 | Chain 是写死流程，Agent 是 LLM 决策流程 |
| **ReAct** | §3.4 | Reason → Act → Observe → Reason 循环 |
| **Tool Calling 协议** | §5 | LLM 用结构化 JSON 决定调哪个工具、传什么参数 |
| **`@tool` 装饰器** | §4.3.1 | 把函数 + docstring + type hint 自动转成 LLM 可见的 schema |
| **`bind_tools()`** | §4.6.2 | 把工具 schema 注入到 LLM 调用的 tools 字段 |
| **LangGraph StateGraph** | §4.6.1 | 显式有向图，**支持 cycle**，是 ReAct 必需的 |
| **State + Reducer** | §4.5 | `Annotated[list, add_messages]` 让多节点写同字段时合并而非覆盖 |
| **Conditional Edge** | §4.6.4 | 节点的下一步由路由函数动态决定 |
| **Checkpointer** | §4.6.5 | 自动在每个节点后 dump state 到 sqlite，支持跨进程续聊 |
| **thread_id** | §4.6.5 | 类似 Git branch，区分不同会话 |
| **recursion_limit** | §4.6.6 | 防死循环的硬上限 |
| **Streaming modes** | §4.6.7 | `values` / `updates` / `messages` 三种粒度 |
| **System prompt 工程** | §4.4 | Agent 的 prompt = 工作流约束 + 失败兜底语义 |
| **Tool 失败哲学** | §4.3.3 | 永不抛异常，返回 `{"error": ...}` |
| **多源 fallback** | §4.3.4 | 关键不是"调多少 API"，而是结果质量怎么排序选优 |
| **工具粒度** | §4.3.2 | 适当组合 > 越细越好，省 token 又快 |
| **工具并行** | §4.3.5 | ToolNode 内部自动并发，prompt 里要鼓励 LLM 一次调多个 |

---

## 9. 实战练习（按难度递增）

### Level 1：读懂

1. 把 `src/agent.py` 里的 `_should_continue` 函数注释掉，改成**永远返回 "tools"**，启动后会发生什么？为什么？
2. 把 `src/state.py` 里 `messages: Annotated[list, add_messages]` 改成 `messages: list`，再问一轮问题，前端历史会怎么样？
3. 把 `src/prompts.py` 里"工作流程"那一段全删掉，看 LLM 会不会还主动调工具？

### Level 2：改造

4. 在 `src/tools/` 下新增一个 `currency.py`：调用 `https://open.er-api.com/v6/latest/CNY` 查汇率。要求：
   - `@tool` 注册
   - 在 `__init__.py` 加进 `ALL_TOOLS`
   - 不改 `agent.py` 的任何代码
   - 问 Agent："去日本玩 3000 元能换多少日元？" 看它会不会自动调

5. 把 `system prompt` 改成"你是一个英文旅游规划师"，看它的中文输出会怎么变化。再加一句"始终用 emoji"，验证 prompt 工程的边际效果。

6. 改 `streamlit_app.py`，加一个 "导出当前会话为 Markdown" 按钮（提示：用 `get_history(thread_id)` + 简单格式化）。

### Level 3：深入

7. 把 SqliteSaver 换成 [Postgres checkpointer](https://github.com/langchain-ai/langgraph/tree/main/libs/checkpoint-postgres)，让多个 Streamlit 实例可以共享状态。

8. 给 `src/agent.py` 加一个 "summarize" 节点：当 messages 数 > 20 时，自动调一次 LLM 把老对话压缩成 SystemMessage 摘要（替代老 messages）。

9. 接入 [LangSmith](https://smith.langchain.com/)：在 `src/llm.py` 的 ChatTongyi 实例化时加 `tags=["travel-agent"]`，把 trace 发到云端。然后看完整的 ReAct 决策图。

### Level 4：架构

10. **多 Agent 协作**：把"行程规划"拆成两个 sub-agent：
    - `planner`（规划行程）
    - `validator`（评分 + 反馈）
    
    用 LangGraph 的 [Multi-Agent Patterns](https://langchain-ai.github.io/langgraph/concepts/multi_agent/) 把它们串起来：planner 给方案，validator 评分，分数低就反馈给 planner 重做（最多 3 轮）。

11. **结合 RAG**：把本项目的"文化背景"工具改成基于 `langchain_rag` 项目的知识库——本地维护一份《中国地理人文》知识库，工具直接走 retriever 而不是 Wikipedia。这就是 **Agent + RAG** 的真实生产形态。

---

## 10. 与本仓库 RAG 项目的对比

读完本文档后，强烈建议你回头对比 `langchain_rag/`：

| 维度 | RAG 项目 | Agent 项目（本文） |
|---|---|---|
| 任务类型 | "问知识库"（封闭世界） | "调外部世界"（开放世界） |
| 流程结构 | DAG（线性管线） | 状态机（循环） |
| LLM 角色 | 翻译机（资料 → 答案） | 决策者（决定下一步） |
| 框架抽象 | LangChain Runnable / Retriever | LangGraph StateGraph + Tool |
| 主要数据结构 | Document + Embedding 向量 | Message（含 tool_calls） |
| 性能瓶颈 | Embedding + 向量检索 | 工具调用网络往返 |
| 评测方式 | Recall@K / NDCG | Tool 调用准确率 / 任务完成率 |
| Memory | 浓缩 + 上下文 | thread checkpoint |

两者**互补，不是替代**：现实生产里，几乎所有复杂应用都是 **Agent 调 RAG 当作其中一个工具**。本仓库后续可以二次开发把这两个项目接起来。

---

## 11. 推荐继续阅读

读完本项目后，按这个顺序往深里走：

1. **LangGraph 官方 conceptual guide**：<https://langchain-ai.github.io/langgraph/concepts/>
2. **ReAct 论文**：<https://arxiv.org/abs/2210.03629>
3. **LangGraph Multi-Agent**：<https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/>
4. **Anthropic 的 "Building effective agents"**：<https://www.anthropic.com/research/building-effective-agents>（讲 workflow vs agent 的取舍，必读）
5. **OpenAI 的 function calling 文档**：<https://platform.openai.com/docs/guides/function-calling>（虽然我们用 Qwen，但协议是一样的）
6. **LangSmith trace** 实践：开一个免费账号，把本项目接进去，肉眼看 LLM 每一次决策的完整 prompt + 输出

---

## 12. 一句话总结

> **Agent = LLM × 工具 × 控制循环**
>
> - **LLM** 决定下一步做什么（推理）
> - **工具**让它能触达外部世界（行动）
> - **控制循环**确保它不会失控、能记住上下文（工程）
>
> LangGraph 帮你把"控制循环"用清晰的图结构画出来，
> `@tool` 帮你把"工具"用最少代码注册进来，
> 你只需要专注于：**给 LLM 写一份漂亮的 system prompt + 一组好用的工具**。

学完这个项目，你已经掌握了构建任意领域 Agent 的全部基础设施。
下一步：**换个领域试试**——医疗咨询助手、面试模拟官、做菜助理、代码评审 bot……
工具变了、prompt 变了，但 LangGraph + ReAct 的骨架完全可以复用。

旅途愉快 🧳
