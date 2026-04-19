"""LangGraph Agent 编排：自定义 ReAct loop + checkpointer 记忆。

为什么不用 `langgraph.prebuilt.create_react_agent`？
- 用户要求"自己实现工具的调用"——这里我们手写 agent 节点 + tool 节点 + 路由函数，
  让 ReAct loop 完全可见、可调试。

图结构（伪图）：
    [START] -> agent -> (有 tool_call?) --yes--> tools -> agent
                          |
                          no
                          v
                        [END]

记忆：
- 用 SqliteSaver checkpointer：每个 thread_id 自动 append messages，
  下一轮调用时整段历史会被自动 prepend，无需我们手动维护对话。
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from .config import settings
from .llm import get_llm
from .prompts import SYSTEM_PROMPT
from .state import TravelState
from .tools import ALL_TOOLS

# 工具按 name -> tool 建索引（备用，目前 ToolNode 已经接管了执行）
TOOL_NAME_MAP = {t.name: t for t in ALL_TOOLS}


# ---------- 节点 1：agent ----------
def _agent_node(state: TravelState) -> dict[str, Any]:
    """让 LLM 决定下一步：要么继续调工具，要么直接给最终答案。"""
    llm = get_llm()
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    msgs: list[BaseMessage] = state["messages"]

    # 把 system prompt 放到最前；checkpointer 不会重复存它
    if not msgs or not isinstance(msgs[0], SystemMessage):
        msgs = [SystemMessage(content=SYSTEM_PROMPT), *msgs]

    response = llm_with_tools.invoke(msgs)
    return {"messages": [response]}


# ---------- 节点 2：tools ----------
# 直接复用 LangGraph 官方 ToolNode：内部就是
#   for tool_call in last_message.tool_calls:
#       result = tool.invoke(tool_call.args)
#       yield ToolMessage(result, tool_call_id=tool_call.id)
# 我们也可以完全手写一份（见文末注释），但 ToolNode 自带并发 + 错误兜底，已经够用。
_tool_node = ToolNode(ALL_TOOLS, handle_tool_errors=True)


# ---------- 路由函数 ----------
def _should_continue(state: TravelState) -> str:
    """检查最新一条 AIMessage 是否还在 call tool。"""
    last = state["messages"][-1]
    # ChatTongyi 返回的 AIMessage 含 tool_calls 字段
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return END


# ---------- 构图 ----------
def _build_graph(checkpointer):
    g = StateGraph(TravelState)
    g.add_node("agent", _agent_node)
    g.add_node("tools", _tool_node)

    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", _should_continue, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile(checkpointer=checkpointer)


# ---------- 对外 API ----------
@contextmanager
def open_agent(persistent: bool = True):
    """上下文管理器形式打开 Agent，确保 sqlite 连接被关闭。

    用法：
        with open_agent() as agent:
            agent.invoke(...)
    """
    if persistent:
        db_path = settings.checkpoint_db_path
        # check_same_thread=False 让 streamlit 多线程也能用同一个连接
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        try:
            checkpointer = SqliteSaver(conn)
            yield _build_graph(checkpointer)
        finally:
            conn.close()
    else:
        # 内存版：方便单测 / 一次性脚本
        yield _build_graph(MemorySaver())


def stream_agent(
    user_message: str,
    thread_id: str,
    *,
    persistent: bool = True,
) -> Iterable[dict[str, Any]]:
    """以流式方式跑一次对话。

    yield 出 LangGraph 的原生 stream event，前端可以基于此渲染 trace。
    """
    from langchain_core.messages import HumanMessage

    config = {"configurable": {"thread_id": thread_id},
              "recursion_limit": settings.MAX_TOOL_ITERATIONS * 2 + 4}
    with open_agent(persistent) as agent:
        for event in agent.stream(
            {"messages": [HumanMessage(content=user_message)]},
            config=config,
            stream_mode="updates",
        ):
            yield event


def invoke_agent(
    user_message: str,
    thread_id: str,
    *,
    persistent: bool = True,
) -> dict[str, Any]:
    """一次性跑完，返回完整 state。"""
    from langchain_core.messages import HumanMessage

    config = {"configurable": {"thread_id": thread_id},
              "recursion_limit": settings.MAX_TOOL_ITERATIONS * 2 + 4}
    with open_agent(persistent) as agent:
        return agent.invoke(
            {"messages": [HumanMessage(content=user_message)]},
            config=config,
        )


def get_history(thread_id: str) -> list[BaseMessage]:
    """取出某个 thread 的全部历史 messages（不含 system prompt）。"""
    config = {"configurable": {"thread_id": thread_id}}
    with open_agent(persistent=True) as agent:
        snap = agent.get_state(config)
        msgs: list[BaseMessage] = snap.values.get("messages", []) if snap else []
        return [m for m in msgs if not isinstance(m, SystemMessage)]


def reset_thread(thread_id: str) -> None:
    """清空某个 thread 的历史。直接 delete 这个 thread 的所有 checkpoint。"""
    db_path = settings.checkpoint_db_path
    if not Path(db_path).exists():
        return
    conn = sqlite3.connect(str(db_path))
    try:
        # SqliteSaver 默认表名：checkpoints / writes
        for table in ("checkpoints", "writes"):
            try:
                conn.execute(f"DELETE FROM {table} WHERE thread_id = ?", (thread_id,))
            except sqlite3.OperationalError:
                pass
        conn.commit()
    finally:
        conn.close()


__all__ = [
    "open_agent",
    "stream_agent",
    "invoke_agent",
    "get_history",
    "reset_thread",
    "TOOL_NAME_MAP",
]


# ============================================================
# 附：完全手写 tool node 的实现，仅供学习对照（默认不启用）
# ============================================================
def _manual_tool_node(state: TravelState) -> dict[str, Any]:
    """与 ToolNode 等价的手写版本——展示"自己实现工具调用"的最小骨架。"""
    last = state["messages"][-1]
    if not isinstance(last, AIMessage) or not last.tool_calls:
        return {}
    out: list[ToolMessage] = []
    for call in last.tool_calls:
        name = call["name"]
        args = call.get("args") or {}
        tool = TOOL_NAME_MAP.get(name)
        if tool is None:
            out.append(ToolMessage(content=f"未知工具：{name}", tool_call_id=call["id"], name=name))
            continue
        try:
            result = tool.invoke(args)
        except Exception as e:  # noqa: BLE001
            result = {"error": f"{type(e).__name__}: {e}"}
        out.append(ToolMessage(content=str(result), tool_call_id=call["id"], name=name))
    return {"messages": out}
