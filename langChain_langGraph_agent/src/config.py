"""集中读取 .env，对外提供一个全局 settings 对象。

模式参考 langchain_rag/src/config.py：使用 pydantic-settings，
任何模块都可以 `from src.config import settings` 拿到强类型配置。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# 在导入 settings 前先 load 一次 .env，确保子进程也能拿到
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH, override=False)


class Settings(BaseSettings):
    """全局配置。字段名与 .env 中大写键一一对应。"""

    # --- LLM ---
    DASHSCOPE_API_KEY: str = Field(default="", description="阿里云百炼 API Key")
    LLM_MODEL: str = Field(default="qwen-plus")
    LLM_TEMPERATURE: float = Field(default=0.5)
    # LLM Provider 选择：
    #   "openai_compat"（默认，推荐）— 走 DashScope OpenAI 兼容端点，流式 + tool_calls 稳定
    #   "tongyi"                   — 走 langchain-community ChatTongyi（流式 + tool_calls 有已知 bug）
    LLM_PROVIDER: str = Field(default="openai_compat")
    LLM_BASE_URL: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        description="OpenAI 兼容端点；默认指向 DashScope",
    )

    # --- 第三方工具 ---
    AMAP_API_KEY: str = Field(default="", description="高德地图 Web 服务 Key（可选）")
    TAVILY_API_KEY: str = Field(default="", description="Tavily 联网搜索 Key（可选）")

    # --- Agent 行为 ---
    MAX_TOOL_ITERATIONS: int = Field(default=10)
    SHOW_TOOL_TRACE: bool = Field(default=True)

    # --- 路径 ---
    DATA_DIR: str = Field(default="./data")
    CHECKPOINT_DB: str = Field(default="./data/checkpoints.sqlite")

    model_config = SettingsConfigDict(
        env_file=str(_ENV_PATH),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ---------- 派生属性 ----------
    @property
    def data_dir(self) -> Path:
        p = Path(self.DATA_DIR)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def checkpoint_db_path(self) -> Path:
        p = Path(self.CHECKPOINT_DB)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def has_amap(self) -> bool:
        return bool(self.AMAP_API_KEY)

    @property
    def has_tavily(self) -> bool:
        return bool(self.TAVILY_API_KEY)


settings = Settings()


def assert_llm_ready() -> None:
    """启动前自检：没有 LLM key 直接抛错，避免一头雾水。"""
    if not settings.DASHSCOPE_API_KEY:
        raise RuntimeError(
            "未检测到 DASHSCOPE_API_KEY。\n"
            "请复制 .env.example 为 .env 并填入你的 DashScope Key：\n"
            "  https://bailian.console.aliyun.com/"
        )
    # dashscope SDK 走环境变量，主动同步一次
    os.environ["DASHSCOPE_API_KEY"] = settings.DASHSCOPE_API_KEY


__all__ = ["settings", "Settings", "assert_llm_ready"]
