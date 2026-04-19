"""LLM 工厂：统一创建 ChatTongyi（Qwen via DashScope）。

为什么单独抽工厂：后续若想换 OpenAI / Claude，只需改这里一处。
"""

from __future__ import annotations

from functools import lru_cache

from langchain_community.chat_models.tongyi import ChatTongyi

from .config import assert_llm_ready, settings


@lru_cache(maxsize=4)
def get_llm(streaming: bool = False, temperature: float | None = None):
    """返回一个 ChatTongyi 实例，跨调用缓存。"""
    assert_llm_ready()
    return ChatTongyi(
        model=settings.LLM_MODEL,
        dashscope_api_key=settings.DASHSCOPE_API_KEY,
        temperature=settings.LLM_TEMPERATURE if temperature is None else temperature,
        streaming=streaming,
        # ChatTongyi 默认会兼容 OpenAI tool calling 格式
    )


__all__ = ["get_llm"]
