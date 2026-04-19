"""数据摄入：扫描 data/ 下的文档 -> 切片 -> 写入 Chroma + Docstore。

为了让 BM25Retriever 能在重启后复用 nodes，这里把 docstore 也持久化到磁盘。
节点元数据会附带 ``file_hash`` / ``original_name`` / ``uploaded_at`` 等字段，
便于上传去重 + 联动删除（参见 ``doc_store.py``）。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import chromadb
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.schema import Document
from llama_index.vector_stores.chroma import ChromaVectorStore

from .config import get_settings
from .doc_store import (
    DocStat,
    file_sha1,
    get_doc_store,
    make_node_metadata,
    reset_doc_store,
)
from .settings import init_settings

ProgressCallback = Callable[[str, int, int], None]

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".md", ".txt", ".docx"}


def _load_documents(data_dir: Path) -> list[Document]:
    if not data_dir.exists() or not any(data_dir.iterdir()):
        raise FileNotFoundError(
            f"数据目录为空：{data_dir}，请把待索引的文档放进去后再试。"
        )

    reader = SimpleDirectoryReader(
        input_dir=str(data_dir),
        recursive=True,
        required_exts=list(SUPPORTED_SUFFIXES),
        filename_as_id=True,
    )
    docs = reader.load_data()
    logger.info("加载文档 %d 篇", len(docs))
    return docs


def _build_chroma_store() -> ChromaVectorStore:
    cfg = get_settings()
    client = chromadb.PersistentClient(path=str(cfg.chroma_dir))
    collection = client.get_or_create_collection(name=cfg.chroma_collection)
    return ChromaVectorStore(chroma_collection=collection)


def build_index(documents: Iterable[Document] | None = None) -> VectorStoreIndex:
    """构建（或重建）索引并持久化。"""
    init_settings()
    cfg = get_settings()

    if documents is None:
        documents = _load_documents(cfg.data_dir)

    vector_store = _build_chroma_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        list(documents),
        storage_context=storage_context,
        show_progress=True,
    )

    # 持久化 docstore / index_store（BM25 复用 nodes 需要）
    storage_context.persist(persist_dir=str(cfg.docstore_dir))
    logger.info("索引构建完成，已持久化到 %s", cfg.storage_dir)
    return index


def precheck_files(
    file_paths: Iterable[Path],
    original_names: Optional[Dict[str, str]] = None,
) -> Tuple[List[Tuple[Path, str, str]], List[Tuple[Path, DocStat]]]:
    """上传前去重检查。

    返回 ``(to_ingest, duplicates)``：
        - ``to_ingest``: ``[(path, file_hash, original_name), ...]``
        - ``duplicates``: ``[(path, existing_doc_stat), ...]``

    判定逻辑：按 ``file_hash`` 去 Chroma 查；命中则视为重复（跳过入库），
    UI 层可决定是直接跳过、删原文件还是替换。
    """
    doc_store = get_doc_store()
    to_ingest: List[Tuple[Path, str, str]] = []
    duplicates: List[Tuple[Path, DocStat]] = []
    names = original_names or {}

    for p in file_paths:
        try:
            h = file_sha1(p)
        except Exception as exc:  # noqa: BLE001
            logger.warning("hash 计算失败，跳过 %s：%s", p, exc)
            continue
        existing = doc_store.find_by_hash(h)
        original = names.get(str(p)) or p.name
        if existing:
            duplicates.append((p, existing))
        else:
            to_ingest.append((p, h, original))
    return to_ingest, duplicates


def build_index_with_progress(
    file_paths: Iterable[Path] | None = None,
    progress_cb: ProgressCallback | None = None,
    embed_batch_size: int = 10,
    original_names: Optional[Dict[str, str]] = None,
    skip_existing: bool = True,
) -> dict:
    """带进度回调的索引构建，便于 UI 展示。

    阶段：
        1. 加载文档
        2. 切片 (按文档计数)
        3. 生成向量并写入 Chroma (按节点计数，分批)

    参数:
        file_paths: 指定要解析的文件列表；为空时回退到扫描整个 data 目录。
        progress_cb: 形如 ``cb(stage, current, total)`` 的回调，用于驱动进度条。
        embed_batch_size: 每次调用 embedding 接口的节点数，越小进度越细。
        original_names: ``{str(saved_path): original_filename}``，让 metadata
            里的 ``original_name`` 不带 timestamp 前缀。
        skip_existing: True 时按 ``file_hash`` 去重；已入库的文件直接跳过。

    返回:
        ``{"files": int, "nodes": int, "elapsed": float, "skipped": int,
           "skipped_files": List[str]}``
    """
    init_settings()
    cfg = get_settings()

    cb: ProgressCallback = progress_cb or (lambda *_: None)
    started = time.time()

    # ---- 1. 加载文档 + 去重 ----
    cb("加载文档", 0, 1)
    if file_paths is not None:
        all_paths = [Path(p) for p in file_paths]
        if not all_paths:
            raise ValueError("file_paths 为空")
    else:
        if not cfg.data_dir.exists() or not any(cfg.data_dir.iterdir()):
            raise FileNotFoundError(
                f"数据目录为空：{cfg.data_dir}，请先放入文档。"
            )
        # 全量索引模式：扫描 data 目录下所有支持的文件
        all_paths = [
            p for p in cfg.data_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
        ]

    skipped: List[str] = []
    if skip_existing:
        to_ingest, duplicates = precheck_files(all_paths, original_names)
        for path, existing in duplicates:
            skipped.append(path.name)
            logger.info(
                "去重跳过：%s（已有节点 %d 个，hash=%s）",
                path.name, existing.node_count, existing.file_hash[:10] + "…",
            )
    else:
        to_ingest = []
        for p in all_paths:
            try:
                to_ingest.append(
                    (p, file_sha1(p), (original_names or {}).get(str(p), p.name))
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("hash 计算失败，跳过 %s：%s", p, exc)

    if not to_ingest:
        elapsed = time.time() - started
        logger.info("无需入库：全部 %d 个文件均已存在", len(skipped))
        return {
            "files": 0, "nodes": 0, "elapsed": elapsed,
            "skipped": len(skipped), "skipped_files": skipped,
        }

    paths_to_load = [str(p) for p, _, _ in to_ingest]
    hash_by_path = {str(p): h for p, h, _ in to_ingest}
    name_by_path = {str(p): n for p, _, n in to_ingest}

    reader = SimpleDirectoryReader(input_files=paths_to_load, filename_as_id=True)
    documents = reader.load_data()
    cb("加载文档", 1, 1)
    logger.info("加载文档 %d 篇（跳过 %d 个重复）", len(documents), len(skipped))

    if not documents:
        return {
            "files": 0, "nodes": 0, "elapsed": time.time() - started,
            "skipped": len(skipped), "skipped_files": skipped,
        }

    # ---- 2. 切片 + 注入元数据 ----
    parser = Settings.node_parser
    total_docs = len(documents)
    cb("切片", 0, total_docs)
    nodes = []
    for i, doc in enumerate(documents, start=1):
        new_nodes = parser.get_nodes_from_documents([doc])
        # 找到这个文档对应的源文件路径，注入文档级元数据
        src_path = (
            doc.metadata.get("file_path")
            or (doc.metadata.get("file_name") and str(Path(cfg.data_dir) / doc.metadata["file_name"]))
        )
        meta_extra: Dict[str, object] = {}
        if src_path and src_path in hash_by_path:
            sp = Path(src_path)
            meta_extra = make_node_metadata(
                sp,
                file_hash=hash_by_path[src_path],
                original_name=name_by_path.get(src_path),
            )
        for n in new_nodes:
            n.metadata.update(meta_extra)
            # 把 file_hash / original_name 等也加入 LLM 可见的 metadata key 列表，
            # 防止默认隐藏导致 retriever rerank 拿不到
            for k in ("file_hash", "original_name", "uploaded_at", "file_size"):
                if k in n.excluded_llm_metadata_keys:
                    continue
                n.excluded_llm_metadata_keys.append(k)
                n.excluded_embed_metadata_keys.append(k)
        nodes.extend(new_nodes)
        cb("切片", i, total_docs)
    logger.info("切片得到节点 %d 个", len(nodes))

    # ---- 3. Embedding + 写入 Chroma + 持久化 docstore ----
    vector_store = _build_chroma_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    embed_model = Settings.embed_model

    total_nodes = len(nodes)
    cb("生成向量并写入", 0, total_nodes)
    for start in range(0, total_nodes, embed_batch_size):
        batch = nodes[start : start + embed_batch_size]
        texts = [n.get_content(metadata_mode="all") for n in batch]
        embeddings = embed_model.get_text_embedding_batch(texts, show_progress=False)
        for node, emb in zip(batch, embeddings):
            node.embedding = emb
        vector_store.add(batch)
        storage_context.docstore.add_documents(batch)
        cb("生成向量并写入", min(start + embed_batch_size, total_nodes), total_nodes)

    storage_context.persist(persist_dir=str(cfg.docstore_dir))
    # Chroma 的 collection.count() 是连接内缓存的，删/加之后清单例确保下次拿到最新
    reset_doc_store()
    elapsed = time.time() - started
    logger.info(
        "索引完成：files=%d nodes=%d skipped=%d 耗时=%.2fs",
        total_docs, total_nodes, len(skipped), elapsed,
    )

    return {
        "files": total_docs,
        "nodes": total_nodes,
        "elapsed": elapsed,
        "skipped": len(skipped),
        "skipped_files": skipped,
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    build_index()


if __name__ == "__main__":
    main()
