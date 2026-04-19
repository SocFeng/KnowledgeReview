"""LLM / Embedding / 文本切片器工厂。

LangChain 没有像 LlamaIndex 那样的全局 ``Settings`` 单例，所以这里把
"懒加载 + 单例 + 环境补丁"集中到本模块；其它模块通过 ``get_llm()`` /
``get_embedding()`` / ``get_text_splitter()`` 获取实例即可。

Embedding 支持两种 provider：
- ``dashscope``：在线 DashScope text-embedding 系列
- ``huggingface``：本地 HuggingFace 模型（如 BAAI/bge-large-zh-v1.5）

通过 ``.env`` 中的 ``EMBEDDING_PROVIDER`` 切换；切换之后必须重新 ingest 索引，
因为不同 embedding 模型生成的向量空间互不兼容。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings as AppSettings, get_settings

logger = logging.getLogger(__name__)

_initialized = False
_llm: Optional[BaseChatModel] = None
_embedding: Optional[Embeddings] = None
_text_splitter: Optional[RecursiveCharacterTextSplitter] = None


def init_settings() -> None:
    """初始化全局 LLM / Embedding / TextSplitter，幂等。"""
    global _initialized, _llm, _embedding, _text_splitter
    if _initialized:
        return

    cfg = get_settings()

    # 关键：langchain-community 的 DashScopeEmbeddings / ChatTongyi 在某些
    # 版本下不会把构造器里传入的 api_key 写回 dashscope SDK 的全局配置，
    # 必须显式设一次，否则底层 HTTP 请求会缺 Authorization 头，返回 401。
    _ensure_dashscope_global_key(cfg.dashscope_api_key)

    _llm = _build_llm(cfg)
    _embedding = _build_embedding(cfg)
    _text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        # 中文友好的分隔顺序：段落 -> 中文标点 -> 英文标点 -> 字符
        separators=[
            "\n\n", "\n",
            "。", "！", "？", "；", "，",
            ".", "!", "?", ";", ",",
            " ", "",
        ],
    )

    _initialized = True


# ---------- DashScope 全局 key ----------
def _ensure_dashscope_global_key(api_key: str) -> None:
    """把 api_key 写到 dashscope SDK 的全局变量 + 环境变量。

    覆盖三处常见取值点：
    1. ``dashscope.api_key``：SDK 内部 TextEmbedding / Generation / TextReRank
       的默认 key 来源。
    2. ``DASHSCOPE_API_KEY`` 环境变量：langchain-community / 一些 wrapper
       会通过环境变量回退取 key。
    3. 同时保留：构造器里仍然显式传 api_key=...（已在各 builder 里做过）。
    """
    if not api_key:
        return
    try:
        import os
        import dashscope

        dashscope.api_key = api_key
        os.environ.setdefault("DASHSCOPE_API_KEY", api_key)
        # 即便已经被设过，也再覆盖一次，避免别处占了空值
        os.environ["DASHSCOPE_API_KEY"] = api_key
        logger.info("已写入 DashScope 全局 API Key（前缀 %s***）", api_key[:6])
    except ImportError:
        logger.warning("未安装 dashscope SDK，跳过全局 key 设置")


# ---------- LLM 工厂 ----------
def _build_llm(cfg: AppSettings) -> BaseChatModel:
    """优先用 langchain-community 的 ChatTongyi 包装 DashScope 上的 Qwen。"""
    try:
        from langchain_community.chat_models.tongyi import ChatTongyi
    except ImportError as exc:
        raise ImportError(
            "未安装 langchain-community（包含 ChatTongyi）。请执行：\n"
            "    pip install langchain-community dashscope"
        ) from exc

    logger.info("加载 Qwen LLM via DashScope: %s", cfg.llm_model)
    # ChatTongyi 接受 streaming=True 后 .stream() 会真流式
    return ChatTongyi(
        model=cfg.llm_model,
        api_key=cfg.dashscope_api_key,
        streaming=True,
        # 给单轮回答留出更长上限
        model_kwargs={"max_tokens": 2048},
    )


# ---------- Embedding 工厂 ----------
def _build_embedding(cfg: AppSettings) -> Embeddings:
    provider = (cfg.embedding_provider or "dashscope").strip().lower()
    if provider == "dashscope":
        return _build_dashscope_embedding(cfg)
    if provider in ("huggingface", "hf", "local"):
        return _build_hf_embedding(cfg)
    raise ValueError(
        f"不支持的 EMBEDDING_PROVIDER={provider!r}，"
        "可选值为 'dashscope' 或 'huggingface'。"
    )


def _build_dashscope_embedding(cfg: AppSettings) -> Embeddings:
    """优先用我们自己的 SDK 直连实现，避免 langchain-community wrapper 在
    部分版本下不正确传递 api_key 导致 401 的坑。"""
    logger.info("使用 DashScope embedding: %s", cfg.embedding_model)
    return _DashScopeSDKEmbeddings(
        model=cfg.embedding_model,
        api_key=cfg.dashscope_api_key,
    )


class _DashScopeSDKEmbeddings(Embeddings):
    """直接走 ``dashscope.TextEmbedding.call`` 的 Embeddings 实现。

    - 完全按 dashscope SDK 行为，避开 langchain-community wrapper 的 401 坑
    - 区分 query / document 两种 text_type（DashScope 对此敏感）
    - 自动按 SDK 上限分批（v3 单次最多 25 条）
    """

    # text-embedding-v3 单次请求最多 25 条；v1/v2 是 25。保守设 16。
    _BATCH_LIMIT = 16

    def __init__(self, model: str, api_key: str) -> None:
        try:
            import dashscope  # noqa: F401
            from dashscope import TextEmbedding  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "未安装 dashscope SDK，请 `pip install dashscope`"
            ) from exc
        self._model = model
        self._api_key = api_key

    def _call(self, texts: list[str], text_type: str) -> list[list[float]]:
        from dashscope import TextEmbedding

        out: list[list[float]] = []
        for start in range(0, len(texts), self._BATCH_LIMIT):
            batch = texts[start: start + self._BATCH_LIMIT]
            resp = TextEmbedding.call(
                model=self._model,
                input=batch,
                api_key=self._api_key,
                text_type=text_type,
            )
            if resp.status_code != 200 or resp.output is None:
                raise RuntimeError(
                    f"DashScope embedding 调用失败：status={resp.status_code} "
                    f"code={getattr(resp, 'code', None)} "
                    f"message={getattr(resp, 'message', None)}"
                )
            embeddings = resp.output.get("embeddings") or []
            # 按 text_index 排序确保和 batch 对齐
            embeddings.sort(key=lambda x: x.get("text_index", 0))
            out.extend([e["embedding"] for e in embeddings])
        return out

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._call(texts, text_type="document")

    def embed_query(self, text: str) -> list[float]:
        return self._call([text], text_type="query")[0]


def _build_hf_embedding(cfg: AppSettings) -> Embeddings:
    """加载本地 HuggingFace embedding 模型。"""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:
        raise ImportError(
            "未安装 langchain-huggingface。请执行：\n"
            "    pip install langchain-huggingface sentence-transformers"
        ) from exc

    model_name_or_path = cfg.hf_embedding_model
    device = _resolve_device(cfg.hf_embedding_device)

    p = Path(model_name_or_path)
    if any(sep in model_name_or_path for sep in ("/", "\\")) and p.exists():
        logger.info(
            "加载本地 HuggingFace embedding: %s (device=%s)", p.resolve(), device
        )
        model_arg = str(p.resolve())
    else:
        logger.info(
            "加载 HuggingFace embedding (model_id=%s, device=%s)",
            model_name_or_path, device,
        )
        model_arg = model_name_or_path

    encode_kwargs = {"normalize_embeddings": True, "batch_size": cfg.hf_embedding_batch_size}
    model_kwargs = {"device": device, "trust_remote_code": True}
    embed = HuggingFaceEmbeddings(
        model_name=model_arg,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    # bge / e5 系列对查询有特殊前缀；通过 wrapper 注入
    if cfg.hf_embedding_query_instruction:
        return _QueryInstructionEmbeddings(embed, cfg.hf_embedding_query_instruction)
    return embed


class _QueryInstructionEmbeddings(Embeddings):
    """给 query 加固定前缀（bge / e5 等模型推荐写法）。"""

    def __init__(self, base: Embeddings, instruction: str) -> None:
        self._base = base
        self._instruction = instruction

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._base.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._base.embed_query(self._instruction + text)


def _resolve_device(device: str) -> str:
    device = (device or "auto").strip().lower()
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


# ---------- 公共访问器 ----------
def get_llm() -> BaseChatModel:
    init_settings()
    assert _llm is not None
    return _llm


def get_embedding() -> Embeddings:
    init_settings()
    assert _embedding is not None
    return _embedding


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    init_settings()
    assert _text_splitter is not None
    return _text_splitter
