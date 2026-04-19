"""文档级元数据管理：在 Chroma + nodes.json 之上提供"按文件查/删/统计"。

为什么需要这一层：
- ``ingest.py`` 写入向量时，每个 chunk 的 metadata 会被注入 ``file_hash``、
  ``original_name``、``uploaded_at``。本模块就是用这些字段做"按文件
  维度"的批量操作。
- 上传去重：算 ``file_hash`` -> 看 Chroma 里是否已存在同 hash 的节点。
- 联动删除：删原始文件时，把 Chroma + ``nodes.json`` 中所有 ``file_hash``
  匹配的节点一并清掉。
- 文档列表：返回每个文件占多少个向量节点，便于 UI 展示。

LangChain 版与 LlamaIndex 版的区别在于：BM25 复用所需的"节点快照"
直接以 ``[{id, text, metadata}, ...]`` 的形式存到一个 JSON 文件里，
不依赖 LlamaIndex 的 SimpleDocumentStore。结构更简单、跨版本更稳。
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

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


# ---------- nodes.json 维护 ----------
class NodeSnapshotStore:
    """BM25 复用所需的节点快照存储（线程安全）。

    文件结构（JSON 数组）::

        [
          {"id": "...", "text": "...", "metadata": {...}},
          ...
        ]
    """

    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def load(self) -> List[Dict[str, Any]]:
        if not self.file_path.exists():
            return []
        try:
            with self.file_path.open(encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            return []
        except Exception as exc:  # noqa: BLE001
            logger.warning("加载 nodes.json 失败: %s", exc)
            return []

    def save(self, nodes: List[Dict[str, Any]]) -> None:
        tmp = self.file_path.with_suffix(".tmp")
        with self._lock:
            tmp.write_text(
                json.dumps(nodes, ensure_ascii=False), encoding="utf-8"
            )
            tmp.replace(self.file_path)

    def append(self, new_nodes: List[Dict[str, Any]]) -> int:
        with self._lock:
            existing = self.load()
            existing.extend(new_nodes)
            tmp = self.file_path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(existing, ensure_ascii=False), encoding="utf-8"
            )
            tmp.replace(self.file_path)
            return len(existing)

    def remove_by_hash(self, file_hash: str) -> int:
        with self._lock:
            existing = self.load()
            kept: List[Dict[str, Any]] = []
            removed = 0
            for n in existing:
                meta = n.get("metadata") or {}
                if meta.get("file_hash") == file_hash:
                    removed += 1
                else:
                    kept.append(n)
            tmp = self.file_path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(kept, ensure_ascii=False), encoding="utf-8"
            )
            tmp.replace(self.file_path)
            return removed


# ---------- 主类 ----------
class DocStore:
    """对 Chroma collection 做"按文件 hash"的批量查询/删除。

    设计上不依赖 LangChain 的高级 API，直接走 chromadb 原生接口，
    免得受版本变更影响。
    """

    def __init__(self) -> None:
        cfg = get_settings()
        self._cfg = cfg
        self._client = chromadb.PersistentClient(path=str(cfg.chroma_dir))
        self._collection = self._client.get_or_create_collection(
            name=cfg.chroma_collection
        )
        self._snapshot = NodeSnapshotStore(cfg.docstore_file)

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
        """按 ``file_hash`` 删 Chroma + nodes.json 中的所有匹配节点。

        返回 Chroma 端实际删除的节点条数（与 nodes.json 端一般一致）。
        """
        if not file_hash:
            return 0

        chroma_deleted = 0
        try:
            got = self._collection.get(
                where={"file_hash": file_hash},
                include=[],
            )
            ids = got.get("ids") or []
            if ids:
                self._collection.delete(ids=ids)
                chroma_deleted = len(ids)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Chroma 按 hash 删除失败: %s", exc)

        snapshot_deleted = self._snapshot.remove_by_hash(file_hash)

        logger.info(
            "已删除 file_hash=%s：chroma=%d snapshot=%d",
            file_hash[:10] + "…", chroma_deleted, snapshot_deleted,
        )
        return chroma_deleted

    # ----- 节点快照（供 BM25 复用）-----
    def append_snapshot(self, nodes: List[Dict[str, Any]]) -> None:
        self._snapshot.append(nodes)

    def load_snapshot(self) -> List[Dict[str, Any]]:
        return self._snapshot.load()


# ---------- 元数据注入 ----------
def make_node_metadata(
    file_path: Path,
    file_hash: str,
    original_name: Optional[str] = None,
) -> Dict[str, Any]:
    """ingest 时统一构造要写到 chunk.metadata 的字段。"""
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
