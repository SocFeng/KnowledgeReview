"""检索层：向量检索 + BM25 混合，再用 DashScope gte-rerank 精排。

设计要点：
- **向量检索**直接基于 Chroma 中已存储的 embeddings 工作，
  不依赖 LlamaIndex 的 ``index_store.json`` 注册表（``build_index_with_progress``
  路径不会写这个文件）。
- **BM25** 需要拿到原始节点文本，所以独立从持久化的 docstore 加载，
  与 ``index.docstore`` 解耦（后者在 ``from_vector_store`` 后是空的）。
- 如果没有持久化的 docstore，自动降级为"仅向量检索"，不再硬性报错。
- 暴露 ``last_trace`` 字段，记录最近一次检索的三阶段命中（向量/BM25/rerank
  前后），便于 UI 做"召回可视化"。
"""

from __future__ import annotations

import contextvars
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import chromadb
import jieba
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.retrievers import BaseRetriever, QueryFusionRetriever
from llama_index.core.schema import BaseNode, NodeWithScore, QueryBundle
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.postprocessor.dashscope_rerank import DashScopeRerank
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore

from .config import get_settings
from .settings import init_settings

logger = logging.getLogger(__name__)

# 用 contextvar 记录"本次请求只在哪些文件名内检索"，避免改动现有 API
# （retriever.retrieve(query) 的签名不变，调用方在外层 set）
_doc_filter_var: contextvars.ContextVar[Optional[Set[str]]] = (
    contextvars.ContextVar("doc_filter", default=None)
)


@dataclass
class RetrievalTrace:
    """单次检索的诊断信息，UI 用来做召回可视化。"""
    vector_hits: List[Dict] = field(default_factory=list)
    bm25_hits: List[Dict] = field(default_factory=list)
    fused_hits: List[Dict] = field(default_factory=list)
    rerank_hits: List[Dict] = field(default_factory=list)
    rerank_used: bool = False
    doc_filter: Optional[List[str]] = None


def set_doc_filter(file_names: Optional[List[str]]) -> contextvars.Token:
    """限定接下来一次检索只在这些 ``original_name`` / ``file_name`` 中召回。

    返回 token，调用方记得在结束后 ``reset_doc_filter(token)`` 还原。
    传入 ``None`` 或 ``[]`` 表示不过滤。
    """
    return _doc_filter_var.set(set(file_names) if file_names else None)


def reset_doc_filter(token: contextvars.Token) -> None:
    _doc_filter_var.reset(token)


def _chinese_tokenizer(text: str) -> List[str]:
    """jieba 分词，BM25 在中文场景下需要自定义分词器。"""
    return [tok for tok in jieba.lcut(text) if tok.strip()]


def _load_chroma_vector_store() -> ChromaVectorStore:
    cfg = get_settings()
    client = chromadb.PersistentClient(path=str(cfg.chroma_dir))
    collection = client.get_or_create_collection(name=cfg.chroma_collection)
    if collection.count() == 0:
        raise FileNotFoundError(
            "向量库为空，请先在 Streamlit 页面上传文档，"
            "或执行 `python -m scripts.ingest` 构建索引。"
        )
    return ChromaVectorStore(chroma_collection=collection)


def _load_persisted_nodes() -> List[BaseNode]:
    """从 docstore.json 恢复节点列表，供 BM25 使用。失败返回空列表。"""
    cfg = get_settings()
    docstore_file = cfg.docstore_dir / "docstore.json"
    if not docstore_file.exists():
        logger.warning("docstore.json 未找到：%s", docstore_file)
        return []
    try:
        docstore = SimpleDocumentStore.from_persist_dir(persist_dir=str(cfg.docstore_dir))
        nodes = list(docstore.docs.values())
        logger.info("从 docstore 加载 %d 个节点", len(nodes))
        return nodes
    except Exception as exc:  # noqa: BLE001
        logger.warning("加载 docstore 失败：%s", exc)
        return []


def _node_file_name(node) -> str:
    """统一拿"原始文件名"——优先 ``original_name``，回退 ``file_name``。"""
    meta = getattr(node, "metadata", {}) or {}
    return meta.get("original_name") or meta.get("file_name") or ""


def _hits_from_nodes(nodes: List[NodeWithScore]) -> List[Dict]:
    """把 ``NodeWithScore`` 简化成 UI 友好的小 dict（用于 trace 面板）。"""
    out: List[Dict] = []
    for n in nodes:
        text = n.node.get_content() or ""
        out.append({
            "file_name": _node_file_name(n.node),
            "score": n.score,
            "preview": text if len(text) <= 180 else text[:180] + "…",
            "node_id": getattr(n.node, "node_id", None) or getattr(n.node, "id_", ""),
        })
    return out


class HybridRetriever(BaseRetriever):
    """对外暴露的检索器：内部组合向量 + BM25（可选），并在最后做 rerank。

    通过 ``last_trace`` 暴露最近一次检索的命中详情；通过 ``set_doc_filter()``
    在外层限定本次只在指定文件名内检索。
    """

    def __init__(self) -> None:
        super().__init__()
        init_settings()
        cfg = get_settings()

        # ---- 向量检索 ----
        vector_store = _load_chroma_vector_store()
        self._index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        # 故意要更多候选（top_k * 3），后面再按 doc_filter 过滤、再交给 rerank
        # 截断；否则启用过滤时容易召回不足。
        self._raw_vector_top_k = cfg.vector_top_k
        self._raw_bm25_top_k = cfg.bm25_top_k
        self._vector_retriever = self._index.as_retriever(
            similarity_top_k=max(cfg.vector_top_k * 3, 15),
        )

        # ---- BM25 检索（可选）----
        nodes = _load_persisted_nodes()
        self._bm25_retriever: Optional[BM25Retriever] = None
        if nodes:
            self._bm25_retriever = BM25Retriever.from_defaults(
                nodes=nodes,
                similarity_top_k=max(cfg.bm25_top_k * 3, 15),
                tokenizer=_chinese_tokenizer,
            )
            logger.info("混合检索：向量 + BM25 (nodes=%d)", len(nodes))
        else:
            logger.warning("docstore 为空或加载失败，已降级为仅向量检索")

        # ---- 融合策略 ----
        if self._bm25_retriever is not None:
            self._main_retriever: BaseRetriever = QueryFusionRetriever(
                retrievers=[self._vector_retriever, self._bm25_retriever],
                similarity_top_k=cfg.vector_top_k + cfg.bm25_top_k,
                num_queries=1,           # 不做 query 改写，保留原始问题
                mode="reciprocal_rerank",
                use_async=False,
                verbose=False,
            )
        else:
            self._main_retriever = self._vector_retriever

        # ---- Rerank ----
        self._rerank_top_n = cfg.rerank_top_n
        try:
            self._reranker: Optional[DashScopeRerank] = DashScopeRerank(
                model=cfg.rerank_model,
                top_n=cfg.rerank_top_n,
                api_key=cfg.dashscope_api_key,
            )
            logger.info("已启用 DashScope rerank：model=%s", cfg.rerank_model)
        except Exception as exc:  # noqa: BLE001
            logger.warning("rerank 初始化失败，将跳过 rerank：%s", exc)
            self._reranker = None

        # 给 UI 用的诊断信息，每次 _retrieve 都会刷新
        self.last_trace: RetrievalTrace = RetrievalTrace()

    def _filter_by_doc(
        self,
        nodes: List[NodeWithScore],
        allowed: Optional[Set[str]],
    ) -> List[NodeWithScore]:
        if not allowed:
            return nodes
        return [n for n in nodes if _node_file_name(n.node) in allowed]

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        allowed = _doc_filter_var.get()
        trace = RetrievalTrace(doc_filter=sorted(allowed) if allowed else None)

        # ---- 三路检索 + 单独记录 trace ----
        try:
            vec_hits = self._vector_retriever.retrieve(query_bundle)
        except Exception as exc:  # noqa: BLE001
            logger.warning("vector 检索失败：%s", exc)
            vec_hits = []
        trace.vector_hits = _hits_from_nodes(vec_hits[: self._raw_vector_top_k * 2])

        bm25_hits: List[NodeWithScore] = []
        if self._bm25_retriever is not None:
            try:
                bm25_hits = self._bm25_retriever.retrieve(query_bundle)
            except Exception as exc:  # noqa: BLE001
                logger.warning("BM25 检索失败：%s", exc)
        trace.bm25_hits = _hits_from_nodes(bm25_hits[: self._raw_bm25_top_k * 2])

        # ---- 融合（QueryFusionRetriever 自己做 reciprocal_rerank）----
        try:
            fused = self._main_retriever.retrieve(query_bundle)
        except Exception as exc:  # noqa: BLE001
            logger.warning("融合检索失败，回退到向量：%s", exc)
            fused = vec_hits

        fused = self._filter_by_doc(fused, allowed)
        trace.fused_hits = _hits_from_nodes(fused)

        if not fused:
            self.last_trace = trace
            return []

        # ---- Rerank ----
        if self._reranker is None:
            final = fused[: self._rerank_top_n]
            trace.rerank_hits = _hits_from_nodes(final)
            trace.rerank_used = False
            self.last_trace = trace
            return final

        try:
            reranked = self._reranker.postprocess_nodes(
                fused, query_bundle=query_bundle
            )
            logger.debug("candidates=%d -> rerank=%d", len(fused), len(reranked))
            trace.rerank_hits = _hits_from_nodes(reranked)
            trace.rerank_used = True
            self.last_trace = trace
            return reranked
        except Exception as exc:  # noqa: BLE001
            # DashScope rerank 调用失败（API key 无 rerank 权限 / 模型名错误 /
            # 限流 / 网络等）时，库内部会返回 None，再尝试 .output.results
            # 触发 AttributeError。这里捕获，降级为不 rerank 的原始 top_n。
            logger.warning(
                "DashScope rerank 调用失败，已降级为仅检索结果（无 rerank）: %s", exc
            )
            final = fused[: self._rerank_top_n]
            trace.rerank_hits = _hits_from_nodes(final)
            trace.rerank_used = False
            self.last_trace = trace
            return final


_retriever: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    """单例：避免每次请求都重新加载 docstore / 构建 BM25。"""
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever
