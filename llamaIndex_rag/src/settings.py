"""配置 LlamaIndex 全局 Settings：LLM、Embedding、切片器。

LlamaIndex v0.12 推荐使用 `Settings` 单例代替旧版 ServiceContext。

Embedding 支持两种 provider：
- ``dashscope``：在线 DashScope text-embedding 系列
- ``huggingface``：本地 HuggingFace 模型（如 BAAI/bge-large-zh-v1.5）

通过 ``.env`` 中的 ``EMBEDDING_PROVIDER`` 切换；切换之后必须重新 ingest 索引，
因为不同 embedding 模型生成的向量空间互不兼容。
"""

from __future__ import annotations

import logging
from pathlib import Path

from llama_index.core import Settings
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,
    DashScopeTextEmbeddingModels,
    DashScopeTextEmbeddingType,
)
from llama_index.llms.dashscope import DashScope

from .config import Settings as AppSettings, get_settings

logger = logging.getLogger(__name__)

_initialized = False


def init_settings() -> None:
    """初始化全局 LLM / Embedding / NodeParser，幂等。"""
    global _initialized
    if _initialized:
        return

    cfg = get_settings()

    # 必须在创建 LLM 之前打补丁
    _patch_dashscope_strip_empty_tool_calls()

    Settings.llm = DashScope(
        model_name=cfg.llm_model,
        api_key=cfg.dashscope_api_key,
        max_tokens=2048,
    )

    Settings.embed_model = _build_embedding(cfg)

    Settings.node_parser = SentenceSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )

    _initialized = True


# ---------- DashScope SDK 补丁 ----------
def _patch_dashscope_strip_empty_tool_calls() -> None:
    """规避 llama-index-llms-dashscope 的 messages 序列化 bug。

    现象：llama-index 在把 ChatMessage 转成 DashScope 请求 dict 时，会给
    assistant 角色的消息加上 ``"tool_calls": []`` 字段。但 DashScope 服务端
    在多轮对话时拒绝空 tool_calls，会返回::

        InvalidParameter: Empty tool_calls is not supported in message.

    导致流式调用第二轮起就静默返回空。这里在 SDK 入口剥掉空 tool_calls 字段。

    Issue 跟踪（如官方修复后可移除此补丁）:
    - https://github.com/run-llama/llama_index/issues  搜索 dashscope tool_calls
    """
    try:
        from dashscope.aigc.generation import Generation
    except ImportError:
        logger.warning("未找到 dashscope SDK，跳过 tool_calls 补丁")
        return

    if getattr(Generation, "_strip_empty_tool_calls_patched", False):
        return

    _orig_call = Generation.call

    def _patched_call(*args, **kwargs):
        msgs = kwargs.get("messages")
        if isinstance(msgs, list):
            for m in msgs:
                if isinstance(m, dict) and "tool_calls" in m and not m["tool_calls"]:
                    del m["tool_calls"]
        return _orig_call(*args, **kwargs)

    Generation.call = staticmethod(_patched_call)
    Generation._strip_empty_tool_calls_patched = True
    logger.info("已应用 DashScope 补丁：自动剥离空 tool_calls 字段")


# ---------- Embedding 工厂 ----------
def _build_embedding(cfg: AppSettings) -> BaseEmbedding:
    provider = (cfg.embedding_provider or "dashscope").strip().lower()
    if provider == "dashscope":
        logger.info("使用 DashScope embedding: %s", cfg.embedding_model)
        return DashScopeEmbedding(
            model_name=_resolve_dashscope_embed_model(cfg.embedding_model),
            text_type=DashScopeTextEmbeddingType.TEXT_TYPE_DOCUMENT,
            api_key=cfg.dashscope_api_key,
        )
    if provider in ("huggingface", "hf", "local"):
        return _build_hf_embedding(cfg)
    raise ValueError(
        f"不支持的 EMBEDDING_PROVIDER={provider!r}，"
        "可选值为 'dashscope' 或 'huggingface'。"
    )


def _resolve_dashscope_embed_model(name: str) -> str:
    """允许用户用短名 (text-embedding-v3) 或枚举原始值。"""
    mapping = {
        "text-embedding-v1": DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V1,
        "text-embedding-v2": DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,
        "text-embedding-v3": DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V3,
    }
    return mapping.get(name, name)


def _build_hf_embedding(cfg: AppSettings) -> BaseEmbedding:
    """加载本地 HuggingFace embedding 模型。"""
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    except ImportError as exc:
        raise ImportError(
            "未安装 llama-index-embeddings-huggingface。请执行：\n"
            "    pip install llama-index-embeddings-huggingface sentence-transformers"
        ) from exc

    model_name_or_path = cfg.hf_embedding_model
    device = _resolve_device(cfg.hf_embedding_device)

    # 如果给的是本地路径，先校验存在
    p = Path(model_name_or_path)
    if any(sep in model_name_or_path for sep in ("/", "\\")) and p.exists():
        logger.info(
            "加载本地 HuggingFace embedding: %s (device=%s)", p.resolve(), device
        )
        model_arg = str(p.resolve())
    else:
        # 视为 HF Hub model_id，首次会自动下载到 ~/.cache/huggingface/
        logger.info(
            "加载 HuggingFace embedding (model_id=%s, device=%s)",
            model_name_or_path, device,
        )
        model_arg = model_name_or_path

    kwargs = {
        "model_name": model_arg,
        "device": device,
        "embed_batch_size": cfg.hf_embedding_batch_size,
        "trust_remote_code": True,
    }
    if cfg.hf_embedding_query_instruction:
        kwargs["query_instruction"] = cfg.hf_embedding_query_instruction

    return HuggingFaceEmbedding(**kwargs)


def _resolve_device(device: str) -> str:
    device = (device or "auto").strip().lower()
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        # Apple Silicon
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"
