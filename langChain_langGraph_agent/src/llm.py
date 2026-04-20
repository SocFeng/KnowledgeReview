"""LLM 工厂：统一创建 Chat 模型。

为什么支持两种 Provider
-----------------------
1. **openai_compat（默认，推荐）**：走 DashScope 的 OpenAI 兼容端点
   (`https://dashscope.aliyuncs.com/compatible-mode/v1`) + ``ChatOpenAI``。
   流式 + tool_calls 组合下完全稳定，能做真·token 打字机。

2. **tongyi（备选）**：``langchain_community.ChatTongyi``，走 dashscope SDK。
   注意：在 ``streaming=True`` + ``tool_calls`` 场景下有已知 bug：
   ``subtract_client_response`` 会因 tool_calls 增量对齐失败抛
   ``IndexError: list index out of range``。
   如果非要用，请把 ``streaming=False``，但会失去 token 级打字机效果。

后续想换 Claude / OpenAI 原生 / 本地 vLLM 只要在此文件加一个分支。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from .config import assert_llm_ready, settings


def _build_openai_compat(streaming: bool, temperature: float) -> Any:
    """走 DashScope OpenAI 兼容端点，复用 ChatOpenAI 更成熟的 stream+tools 实现。"""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.LLM_MODEL,
        api_key=settings.DASHSCOPE_API_KEY,
        base_url=settings.LLM_BASE_URL,
        temperature=temperature,
        streaming=streaming,
        # 让流式模式的最后一个 chunk 带 usage_metadata（langchain-openai 默认关）
        # 这样我们才能统计 token / 计费
        stream_usage=True,
    )


def _build_tongyi(streaming: bool, temperature: float) -> Any:
    """ChatTongyi 走 dashscope SDK；保留作为 fallback。"""
    from langchain_community.chat_models.tongyi import ChatTongyi

    return ChatTongyi(
        model=settings.LLM_MODEL,
        dashscope_api_key=settings.DASHSCOPE_API_KEY,
        temperature=temperature,
        streaming=streaming,
    )


@lru_cache(maxsize=4)
def get_llm(streaming: bool = True, temperature: float | None = None):
    """返回一个 Chat 模型实例，跨调用缓存。

    默认 ``streaming=True``：这样 LangGraph 以 ``stream_mode="messages"`` 迭代时，
    会把 LLM token 分片（AIMessageChunk）一边到达一边吐给前端，实现"打字机"效果。
    """
    assert_llm_ready()
    temp = settings.LLM_TEMPERATURE if temperature is None else temperature

    provider = (settings.LLM_PROVIDER or "openai_compat").lower()
    if provider in ("openai_compat", "openai", "compat"):
        return _build_openai_compat(streaming, temp)
    if provider in ("tongyi", "dashscope"):
        return _build_tongyi(streaming, temp)
    raise ValueError(
        f"未知 LLM_PROVIDER={settings.LLM_PROVIDER!r}；可选：openai_compat / tongyi"
    )


__all__ = ["get_llm"]
