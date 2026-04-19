"""文档级元数据管理：在 Chroma + docstore 之上提供"按文件查/删/统计"。

为什么需要这一层：
- ``ingest.py`` 写入向量时，每个 node 的 metadata 会被注入 ``file_hash``、
  ``original_name``、``uploaded_at``。本模块就是用这些字段做"按文件
  维度"的批量操作。
- 上传去重：算 ``file_hash`` -> 看 Chroma 里是否已存在同 hash 的节点。
- 联动删除：删原始文件时，把 Chroma + docstore.json 中所有 ``file_hash``
  匹配的节点一并清掉。
- 文档列表：返回每个文件占多少个向量节点，便于 UI 展示。
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import chromadb

from .config import get_settings

logger = logging.getLogger(__name__)


# ---------- 工具 ----------
def file_sha1(path: Path, block_size: int = 1 << 20) -> str:
    """计算文件 SHA-1（足够区分同名文件是否变化，又比 SHA-256 快）。"""
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            chunk = f.read(block_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def bytes_sha1(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


# ---------- 数据结构 ----------
@dataclass
class DocStat:
    """一个文档在向量库中的统计。"""
    file_hash: str
    original_name: str            # 用户上传时的原始文件名
    saved_path: Optional[str]     # 落盘到 data/uploads/ 的实际路径（可能为空）
    node_count: int
    uploaded_at: Optional[float]
    size_bytes: int


# ---------- 主类 ----------
class DocStore:
    """对 Chroma collection 做"按文件 hash"的批量查询/删除。

    设计上不依赖 LlamaIndex 的高级 API，直接走 chromadb 原生接口，
    免得受 LlamaIndex 版本变更影响。
    """

    def __init__(self) -> None:
        cfg = get_settings()
        self._cfg = cfg
        self._client = chromadb.PersistentClient(path=str(cfg.chroma_dir))
        self._collection = self._client.get_or_create_collection(
            name=cfg.chroma_collection
        )

    # ----- 查询 -----
    def total_nodes(self) -> int:
        try:
            return self._collection.count()
        except Exception:  # noqa: BLE001
            return 0

    def list_documents(self) -> List[DocStat]:
        """聚合 metadata 给出"每个 file_hash 一个 DocStat"。"""
        try:
            data = self._collection.get(include=["metadatas"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("读取 Chroma metadata 失败: %s", exc)
            return []

        metas = data.get("metadatas") or []
        groups: Dict[str, DocStat] = {}
        for m in metas:
            if not m:
                continue
            h = m.get("file_hash") or m.get("file_name") or "(unknown)"
            if h not in groups:
                groups[h] = DocStat(
                    file_hash=h,
                    original_name=m.get("original_name")
                    or m.get("file_name")
                    or "(未知)",
                    saved_path=m.get("file_path"),
                    node_count=0,
                    uploaded_at=m.get("uploaded_at"),
                    size_bytes=int(m.get("file_size", 0) or 0),
                )
            groups[h].node_count += 1
        return sorted(
            groups.values(),
            key=lambda s: (-(s.uploaded_at or 0), s.original_name),
        )

    def find_by_hash(self, file_hash: str) -> Optional[DocStat]:
        for stat in self.list_documents():
            if stat.file_hash == file_hash:
                return stat
        return None

    def has_hash(self, file_hash: str) -> bool:
        return self.find_by_hash(file_hash) is not None

    # ----- 删除 -----
    def delete_by_hash(self, file_hash: str) -> int:
        """按 ``file_hash`` 删 Chroma + docstore 中的所有匹配节点。

        返回实际删除的节点条数。
        """
        if not file_hash:
            return 0

        # ---- Chroma 端 ----
        chroma_deleted = 0
        try:
            got = self._collection.get(
                where={"file_hash": file_hash},
                include=[],  # 只要 ids
            )
            ids = got.get("ids") or []
            if ids:
                self._collection.delete(ids=ids)
                chroma_deleted = len(ids)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Chroma 按 hash 删除失败: %s", exc)

        # ---- docstore.json 端 ----
        docstore_deleted = self._purge_docstore(file_hash)

        logger.info(
            "已删除 file_hash=%s：chroma=%d docstore=%d",
            file_hash[:10] + "…", chroma_deleted, docstore_deleted,
        )
        return chroma_deleted

    def _purge_docstore(self, file_hash: str) -> int:
        """直接编辑 ``docstore.json``，把 metadata.file_hash 匹配的节点删掉。

        这里手工改 JSON 而不是用 SimpleDocumentStore，是为了避免引入更
        多 LlamaIndex 内部依赖；docstore.json 结构相对稳定。
        """
        docstore_file = self._cfg.docstore_dir / "docstore.json"
        if not docstore_file.exists():
            return 0
        try:
            data = json.loads(docstore_file.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("读取 docstore.json 失败: %s", exc)
            return 0

        # docstore.json 内部结构：{"docstore/data": {<id>: {"__data__": {...}}}}
        bucket = (
            data.get("docstore/data")
            or data.get("docstore", {}).get("data")
            or {}
        )
        to_delete: List[str] = []
        for nid, payload in bucket.items():
            inner = payload.get("__data__", payload) if isinstance(payload, dict) else {}
            meta = inner.get("metadata") or {}
            if meta.get("file_hash") == file_hash:
                to_delete.append(nid)

        for nid in to_delete:
            bucket.pop(nid, None)

        # 同步删 ref_doc / metadata 这些"附属表"
        for k in ("docstore/ref_doc_info", "docstore/metadata"):
            sub = data.get(k)
            if isinstance(sub, dict):
                for nid in list(sub.keys()):
                    if nid in to_delete:
                        sub.pop(nid, None)

        try:
            tmp = docstore_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            tmp.replace(docstore_file)
        except Exception as exc:  # noqa: BLE001
            logger.warning("写回 docstore.json 失败: %s", exc)
            return 0

        return len(to_delete)


# ---------- 元数据注入 ----------
def make_node_metadata(
    file_path: Path,
    file_hash: str,
    original_name: Optional[str] = None,
) -> Dict[str, object]:
    """ingest 时统一构造要写到 node.metadata 的字段。"""
    try:
        size = file_path.stat().st_size
    except Exception:  # noqa: BLE001
        size = 0
    return {
        "file_hash": file_hash,
        "file_name": file_path.name,
        "original_name": original_name or file_path.name,
        "file_path": str(file_path),
        "file_size": size,
        "uploaded_at": time.time(),
    }


# ---------- 单例 ----------
_doc_store: Optional[DocStore] = None


def get_doc_store() -> DocStore:
    global _doc_store
    if _doc_store is None:
        _doc_store = DocStore()
    return _doc_store


def reset_doc_store() -> None:
    """删除文件后调用，让下一次 get_doc_store 重新连 Chroma（拿到最新 count）。"""
    global _doc_store
    _doc_store = None
