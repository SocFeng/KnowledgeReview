"""LangGraph 的 State 定义。

为什么单独抽一个文件：
- State 是节点之间共享的数据结构，agent.py 里既会读它也会写它，
  独立成文件方便后续扩展（比如要加 user_profile / preferences）。
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class TravelState(TypedDict):
    """对话状态。

    - `messages` 是核心字段，使用 LangGraph 的 add_messages reducer，
      新追加的 message 会被自动 append（包含 ToolMessage / AIMessage 等）。
    - 不需要把 user_profile / itinerary 单独存：只要它出现在 messages 里，
      LLM 自然就能基于上下文修改。
    """

    messages: Annotated[list, add_messages]


__all__ = ["TravelState"]
