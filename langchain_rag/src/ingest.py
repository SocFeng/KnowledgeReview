"""数据摄入：扫描 data/ 下的文档 -> 切片 -> 写入 Chroma + nodes.json。

要点：
- 走原生 chromadb 接口，避免 LangChain Chroma 包装在不同版本下细节差异。
- 每个 chunk 的 metadata 会注入 ``file_hash`` / ``original_name`` /
  ``uploaded_at`` 等字段，供上传去重和联动删除（参见 ``doc_store.py``）。
- 同步把节点 ``{id, text, metadata}`` 三元组写到 ``storage/docstore/nodes.json``，
  供 BM25 重启时复用（``retriever.py``）。
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import chromadb

from .config import get_settings
from .doc_store import (
    DocStat,
    file_sha1,
    get_doc_store,
    make_node_metadata,
    reset_doc_store,
)
from .settings import get_embedding, get_text_splitter, init_settings

ProgressCallback = Callable[[str, int, int], None]

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".md", ".txt", ".docx"}


# ---------- 文档加载 ----------
def _load_one(path: Path) -> str:
    """根据后缀选择加载器，返回纯文本（已合并所有页/段）。"""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(str(path))
        docs = loader.load()
        return "\n\n".join(d.page_content for d in docs if d.page_content)
    if suffix == ".docx":
        try:
            from langchain_community.document_loaders import Docx2txtLoader
            loader = Docx2txtLoader(str(path))
            docs = loader.load()
            return "\n\n".join(d.page_content for d in docs if d.page_content)
        except Exception:
            from docx import Document
            d = Document(str(path))
            return "\n".join(p.text for p in d.paragraphs if p.text)
    if suffix == ".md":
        # Markdown 直接读纯文本，让 splitter 按段落切；保留 markdown 结构对 BM25/向量
        # 都更友好，比 unstructured 转换的中间产物可控。
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")
    raise ValueError(f"不支持的文件类型: {suffix}")


def _build_chroma_collection():
    cfg = get_settings()
    client = chromadb.PersistentClient(path=str(cfg.chroma_dir))
    return client.get_or_create_collection(name=cfg.chroma_collection)


# ---------- 上传去重检查 ----------
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


# ---------- 主流程 ----------
def build_index_with_progress(
    file_paths: Iterable[Path] | None = None,
    progress_cb: ProgressCallback | None = None,
    embed_batch_size: int = 10,
    original_names: Optional[Dict[str, str]] = None,
    skip_existing: bool = True,
) -> dict:
    """带进度回调的索引构建，便于 UI 展示。

    阶段：
        1. 加载文档（含 hash 去重）
        2. 切片 + 注入元数据（按文档计数）
        3. 生成向量并写入 Chroma + nodes.json（按节点计数，分批）

    参数:
        file_paths: 指定要解析的文件列表；为空时回退到扫描整个 data 目录。
        progress_cb: 形如 ``cb(stage, current, total)`` 的回调，用于驱动进度条。
        embed_batch_size: 每次调用 embedding 接口的 chunk 数，越小进度越细。
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

    # ---- 1. 加载文件清单 + 去重 ----
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

    # ---- 2. 加载 + 切片 + 注入元数据 ----
    splitter = get_text_splitter()
    total_files = len(to_ingest)
    cb("切片", 0, total_files)

    chunks: List[Dict] = []  # 每个元素：{"id", "text", "metadata"}
    for i, (path, file_hash, original) in enumerate(to_ingest, start=1):
        try:
            text = _load_one(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("文件解析失败，跳过 %s：%s", path, exc)
            cb("切片", i, total_files)
            continue
        if not text or not text.strip():
            logger.warning("文件解析为空，跳过 %s", path)
            cb("切片", i, total_files)
            continue

        base_meta = make_node_metadata(
            path, file_hash=file_hash, original_name=original
        )
        for piece in splitter.split_text(text):
            piece = (piece or "").strip()
            if not piece:
                continue
            chunks.append({
                "id": uuid.uuid4().hex,
                "text": piece,
                "metadata": dict(base_meta),
            })
        cb("切片", i, total_files)

    if not chunks:
        return {
            "files": 0, "nodes": 0, "elapsed": time.time() - started,
            "skipped": len(skipped), "skipped_files": skipped,
        }
    logger.info("切片得到节点 %d 个（来自 %d 个文件）", len(chunks), total_files)

    # ---- 3. Embedding + 写入 Chroma + 追加 nodes.json ----
    embed_model = get_embedding()
    collection = _build_chroma_collection()
    snapshot_buffer: List[Dict] = []

    total_nodes = len(chunks)
    cb("生成向量并写入", 0, total_nodes)
    for start in range(0, total_nodes, embed_batch_size):
        batch = chunks[start: start + embed_batch_size]
        texts = [c["text"] for c in batch]
        try:
            embeddings = embed_model.embed_documents(texts)
        except Exception as exc:  # noqa: BLE001
            logger.exception("embedding 调用失败：%s", exc)
            raise

        collection.add(
            ids=[c["id"] for c in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[c["metadata"] for c in batch],
        )
        snapshot_buffer.extend(batch)
        cb("生成向量并写入", min(start + embed_batch_size, total_nodes), total_nodes)

    # 追加节点快照（供 BM25 复用）
    get_doc_store().append_snapshot(snapshot_buffer)
    reset_doc_store()  # 让下一次 get_doc_store 重新连 Chroma 拿到新 count

    elapsed = time.time() - started
    logger.info(
        "索引完成：files=%d nodes=%d skipped=%d 耗时=%.2fs",
        total_files, total_nodes, len(skipped), elapsed,
    )

    return {
        "files": total_files,
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
    build_index_with_progress()


if __name__ == "__main__":
    main()
