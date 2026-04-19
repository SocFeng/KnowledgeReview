"""检索层：向量检索 + BM25 混合，再用 DashScope gte-rerank 精排。

设计要点：
- **向量检索**直接基于 Chroma 中已存储的 embeddings 工作，
  只用原生 chromadb 接口（不依赖 LangChain Chroma 包装的版本细节）。
- **BM25** 复用 ingest 阶段写出的 ``storage/docstore/nodes.json`` 节点快照，
  用 ``rank_bm25`` 配 ``jieba`` 分词。如果快照为空就降级为"仅向量检索"。
- **融合**用经典的 Reciprocal Rank Fusion（RRF），不使用 LangChain 的
  EnsembleRetriever，避免它在某些版本下与自定义检索器接口对不上。
- **Rerank** 走 dashscope SDK 的 ``TextReRank.call``，失败自动降级到无 rerank。
- 暴露 ``last_trace`` 字段，记录最近一次检索的四阶段命中（向量/BM25/融合/rerank
  前后），便于 UI 做"召回可视化"。
- 通过 contextvar 提供 ``set_doc_filter()``，让外层在不改 retrieve 签名的
  前提下临时限定本次只在指定文件名内检索。
"""

from __future__ import annotations

import contextvars
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import chromadb
import jieba
from langchain_core.documents import Document

from .config import get_settings
from .doc_store import get_doc_store
from .settings import get_embedding, init_settings

logger = logging.getLogger(__name__)

_doc_filter_var: contextvars.ContextVar[Optional[Set[str]]] = (
    contextvars.ContextVar("doc_filter", default=None)
)


@dataclass
class ScoredDoc:
    """检索结果：把 ``Document`` 和分数 / id 绑在一起。"""
    doc: Document
    score: Optional[float]
    node_id: str = ""


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


def _node_file_name(doc: Document) -> str:
    meta = doc.metadata or {}
    return meta.get("original_name") or meta.get("file_name") or ""


def _hits_summary(items: List[ScoredDoc]) -> List[Dict]:
    out: List[Dict] = []
    for it in items:
        text = it.doc.page_content or ""
        out.append({
            "file_name": _node_file_name(it.doc),
            "score": it.score,
            "preview": text if len(text) <= 180 else text[:180] + "…",
            "node_id": it.node_id,
        })
    return out


# ---------- 子检索器 ----------
class _ChromaVectorRetriever:
    """直接走 chromadb 原生 ``query``，避开 LangChain Chroma 版本差异。"""

    def __init__(self, top_k: int) -> None:
        cfg = get_settings()
        client = chromadb.PersistentClient(path=str(cfg.chroma_dir))
        self._collection = client.get_or_create_collection(name=cfg.chroma_collection)
        if self._collection.count() == 0:
            raise FileNotFoundError(
                "向量库为空，请先在 Streamlit 页面上传文档，"
                "或执行 `python -m scripts.ingest` 构建索引。"
            )
        self._embed = get_embedding()
        self._top_k = top_k

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[ScoredDoc]:
        k = top_k or self._top_k
        try:
            qvec = self._embed.embed_query(query)
        except Exception as exc:  # noqa: BLE001
            logger.warning("query embedding 失败：%s", exc)
            return []
        try:
            res = self._collection.query(
                query_embeddings=[qvec],
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Chroma query 失败：%s", exc)
            return []

        docs_lists = res.get("documents") or [[]]
        metas_lists = res.get("metadatas") or [[]]
        ids_lists = res.get("ids") or [[]]
        dists_lists = res.get("distances") or [[]]

        docs = docs_lists[0] if docs_lists else []
        metas = metas_lists[0] if metas_lists else []
        ids = ids_lists[0] if ids_lists else []
        dists = dists_lists[0] if dists_lists else []

        out: List[ScoredDoc] = []
        for i, text in enumerate(docs):
            meta = metas[i] if i < len(metas) else {}
            d = dists[i] if i < len(dists) else None
            # Chroma 默认返回的是 L2 / cosine distance；UI 上希望"越大越相似"，
            # 这里把 cosine distance 转成相似度：相似度 = 1 - distance
            score = (1.0 - d) if isinstance(d, (int, float)) else None
            out.append(ScoredDoc(
                doc=Document(page_content=text or "", metadata=meta or {}),
                score=score,
                node_id=ids[i] if i < len(ids) else "",
            ))
        return out


class _BM25Retriever:
    """基于 rank_bm25 的中文 BM25。所有节点都是从 nodes.json 复活的。"""

    def __init__(self, top_k: int) -> None:
        from rank_bm25 import BM25Okapi  # 局部 import 避免没装时报错

        snapshot = get_doc_store().load_snapshot()
        if not snapshot:
            self._bm25 = None
            self._docs: List[ScoredDoc] = []
            self._top_k = top_k
            logger.warning("nodes.json 为空，BM25 不可用")
            return

        self._docs = [
            ScoredDoc(
                doc=Document(
                    page_content=n.get("text", ""),
                    metadata=n.get("metadata") or {},
                ),
                score=None,
                node_id=n.get("id", ""),
            )
            for n in snapshot
        ]
        tokenized = [_chinese_tokenizer(d.doc.page_content) for d in self._docs]
        self._bm25 = BM25Okapi(tokenized)
        self._top_k = top_k
        logger.info("BM25 初始化完成（nodes=%d）", len(self._docs))

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[ScoredDoc]:
        if self._bm25 is None:
            return []
        k = top_k or self._top_k
        scores = self._bm25.get_scores(_chinese_tokenizer(query))
        # 取分数最高的 k 条
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        out: List[ScoredDoc] = []
        for i in order:
            base = self._docs[i]
            out.append(ScoredDoc(
                doc=base.doc,
                score=float(scores[i]),
                node_id=base.node_id,
            ))
        return out


# ---------- 融合 + Rerank ----------
def _rrf_fuse(
    rankings: List[List[ScoredDoc]],
    k: int = 60,
) -> List[ScoredDoc]:
    """Reciprocal Rank Fusion：score = Σ 1/(k + rank_i)。

    去重以 node_id 优先；node_id 缺失时退化到 page_content。
    """
    score_map: Dict[str, float] = {}
    keep: Dict[str, ScoredDoc] = {}
    for ranked in rankings:
        for rank, item in enumerate(ranked, start=1):
            key = item.node_id or item.doc.page_content[:64]
            score_map[key] = score_map.get(key, 0.0) + 1.0 / (k + rank)
            if key not in keep:
                keep[key] = item
    fused = sorted(keep.values(), key=lambda x: -score_map[
        x.node_id or x.doc.page_content[:64]
    ])
    # 把 RRF 分数写回 score 字段，便于 UI 展示
    for x in fused:
        key = x.node_id or x.doc.page_content[:64]
        x.score = score_map[key]
    return fused


def _dashscope_rerank(
    query: str,
    candidates: List[ScoredDoc],
    top_n: int,
    model: str,
    api_key: str,
) -> Optional[List[ScoredDoc]]:
    """直接调 dashscope SDK 做 rerank。失败返回 None。"""
    if not candidates:
        return []
    try:
        import dashscope
        from dashscope import TextReRank
    except ImportError:
        logger.warning("未安装 dashscope SDK，跳过 rerank")
        return None

    try:
        dashscope.api_key = api_key
        resp = TextReRank.call(
            model=model,
            query=query,
            documents=[c.doc.page_content for c in candidates],
            top_n=min(top_n, len(candidates)),
            return_documents=False,
        )
        if resp.status_code != 200 or resp.output is None:
            logger.warning(
                "rerank 调用失败：status=%s code=%s msg=%s",
                resp.status_code, getattr(resp, "code", None),
                getattr(resp, "message", None),
            )
            return None
        results = resp.output.results or []
        out: List[ScoredDoc] = []
        for r in results:
            idx = r.index
            if 0 <= idx < len(candidates):
                base = candidates[idx]
                out.append(ScoredDoc(
                    doc=base.doc,
                    score=float(r.relevance_score),
                    node_id=base.node_id,
                ))
        return out
    except Exception as exc:  # noqa: BLE001
        logger.warning("rerank 调用过程异常：%s", exc)
        return None


# ---------- 主检索器 ----------
class HybridRetriever:
    """对外暴露的检索器：内部组合向量 + BM25（可选），最后做 rerank。

    通过 ``last_trace`` 暴露最近一次检索的命中详情；通过 ``set_doc_filter()``
    在外层限定本次只在指定文件名内检索。
    """

    def __init__(self) -> None:
        init_settings()
        cfg = get_settings()
        self._cfg = cfg

        # 故意要更多候选（top_k * 3），后面再按 doc_filter 过滤、再交给 rerank
        # 截断；否则启用过滤时容易召回不足。
        self._raw_vector_top_k = cfg.vector_top_k
        self._raw_bm25_top_k = cfg.bm25_top_k
        vec_pool = max(cfg.vector_top_k * 3, 15)
        bm25_pool = max(cfg.bm25_top_k * 3, 15)

        self._vector = _ChromaVectorRetriever(top_k=vec_pool)

        bm25 = _BM25Retriever(top_k=bm25_pool)
        self._bm25: Optional[_BM25Retriever] = bm25 if bm25._bm25 is not None else None
        if self._bm25 is None:
            logger.warning("已降级为仅向量检索（nodes.json 为空）")
        else:
            logger.info("混合检索：向量 + BM25")

        self._rerank_top_n = cfg.rerank_top_n
        self.last_trace: RetrievalTrace = RetrievalTrace()

    # ----- 工具 -----
    def _filter_by_doc(
        self,
        items: List[ScoredDoc],
        allowed: Optional[Set[str]],
    ) -> List[ScoredDoc]:
        if not allowed:
            return items
        return [x for x in items if _node_file_name(x.doc) in allowed]

    # ----- 主入口 -----
    def retrieve(self, query: str) -> List[Document]:
        """对外接口：返回最终用于 LLM 上下文的 ``Document`` 列表。"""
        scored = self.retrieve_scored(query)
        return [s.doc for s in scored]

    def retrieve_scored(self, query: str) -> List[ScoredDoc]:
        allowed = _doc_filter_var.get()
        trace = RetrievalTrace(doc_filter=sorted(allowed) if allowed else None)

        # ---- 向量 ----
        try:
            vec_hits = self._vector.retrieve(query)
        except FileNotFoundError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("vector 检索失败：%s", exc)
            vec_hits = []
        trace.vector_hits = _hits_summary(vec_hits[: self._raw_vector_top_k * 2])

        # ---- BM25 ----
        bm25_hits: List[ScoredDoc] = []
        if self._bm25 is not None:
            try:
                bm25_hits = self._bm25.retrieve(query)
            except Exception as exc:  # noqa: BLE001
                logger.warning("BM25 检索失败：%s", exc)
        trace.bm25_hits = _hits_summary(bm25_hits[: self._raw_bm25_top_k * 2])

        # ---- 融合 ----
        if self._bm25 is not None and bm25_hits:
            fused = _rrf_fuse([vec_hits, bm25_hits])
        else:
            fused = list(vec_hits)
        # 按 doc_filter 过滤；再按 vector_top_k + bm25_top_k 截断
        fused = self._filter_by_doc(fused, allowed)
        fused = fused[: max(self._cfg.vector_top_k + self._cfg.bm25_top_k, 4)]
        trace.fused_hits = _hits_summary(fused)

        if not fused:
            self.last_trace = trace
            return []

        # ---- Rerank ----
        reranked = _dashscope_rerank(
            query=query,
            candidates=fused,
            top_n=self._rerank_top_n,
            model=self._cfg.rerank_model,
            api_key=self._cfg.dashscope_api_key,
        )
        if reranked is None:
            final = fused[: self._rerank_top_n]
            trace.rerank_hits = _hits_summary(final)
            trace.rerank_used = False
            self.last_trace = trace
            return final

        trace.rerank_hits = _hits_summary(reranked)
        trace.rerank_used = True
        self.last_trace = trace
        return reranked


# ---------- 单例 ----------
_retriever: Optional[HybridRetriever] = None


def get_retriever() -> HybridRetriever:
    """单例：避免每次请求都重新加载 BM25 / 重建 chroma 客户端。"""
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


def reset_retriever() -> None:
    """ingest 完成后调用：让下一次 get_retriever 重新加载 BM25 节点。"""
    global _retriever
    _retriever = None
