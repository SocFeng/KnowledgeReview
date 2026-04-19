"""对话历史持久化：每个会话存为 ``<chat_history_dir>/<sid>.json``。

文件结构::

    {
      "session_id": "xxx",
      "title": "羊毛衣物如何养护",
      "created_at": 1729345678.123,
      "updated_at": 1729345700.456,
      "messages": [
        {"role": "user", "content": "...", "citations": []},
        {"role": "assistant", "content": "...", "citations": [
            {"index": 1, "score": 0.83, "file_name": "xx.txt", "text": "..."}
        ]}
      ]
    }

设计要点：
- 单独的一层，不依赖 LlamaIndex；UI / 后端都能直接调用。
- 用 ``asdict`` + JSON 持久化，加载时再转回 dataclass，避免和 ``Citation``
  类型纠缠太紧。
- 写入采用 ``tmp -> replace`` 的方式，避免半写入造成损坏。
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------- 数据结构 ----------
@dataclass
class StoredCitation:
    index: int
    score: Optional[float]
    file_name: Optional[str]
    text: str


@dataclass
class StoredMessage:
    role: str  # "user" | "assistant"
    content: str
    citations: List[StoredCitation] = field(default_factory=list)
    # 标识 assistant 消息是否被中断（用户在生成中点了 stop）。
    # 用户消息恒为 False。仅用于 UI 渲染"已中断"徽章。
    partial: bool = False


@dataclass
class StoredSession:
    session_id: str
    title: str
    created_at: float
    updated_at: float
    messages: List[StoredMessage] = field(default_factory=list)
    pinned: bool = False
    style: str = "concise"


@dataclass
class SessionMeta:
    """轻量元数据，用于侧边栏列表展示。"""
    session_id: str
    title: str
    created_at: float
    updated_at: float
    message_count: int
    pinned: bool = False


# ---------- 存储层 ----------
class SessionStore:
    """会话持久化存储。线程安全（粗粒度锁）。"""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    # ----- 路径辅助 -----
    def _path(self, sid: str) -> Path:
        return self.root / f"{sid}.json"

    # ----- 列出 / 读取 -----
    def list(self, query: str = "") -> List[SessionMeta]:
        """列出所有会话。

        ``query`` 非空时按关键字过滤：匹配 title 或任一消息正文（大小写不敏感）。
        排序：固定会话优先，其次按最近活跃时间倒序。
        """
        q = (query or "").strip().lower()
        items: List[SessionMeta] = []
        for p in self.root.glob("*.json"):
            try:
                with p.open(encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as exc:  # noqa: BLE001
                logger.warning("会话文件损坏，已跳过：%s (%s)", p, exc)
                continue

            title = data.get("title", "未命名会话")
            messages = data.get("messages", [])
            if q:
                hay = title.lower() + "\n" + "\n".join(
                    str(m.get("content", "")) for m in messages
                ).lower()
                if q not in hay:
                    continue
            items.append(
                SessionMeta(
                    session_id=data["session_id"],
                    title=title,
                    created_at=data.get("created_at", 0.0),
                    updated_at=data.get("updated_at", 0.0),
                    message_count=len(messages),
                    pinned=bool(data.get("pinned", False)),
                )
            )
        # 固定优先，再按最近活跃倒序
        items.sort(key=lambda x: (not x.pinned, -x.updated_at))
        return items

    def load(self, sid: str) -> Optional[StoredSession]:
        p = self._path(sid)
        if not p.exists():
            return None
        try:
            with p.open(encoding="utf-8") as f:
                data = json.load(f)
            messages = [
                StoredMessage(
                    role=m["role"],
                    content=m["content"],
                    citations=[StoredCitation(**c) for c in m.get("citations", [])],
                    partial=bool(m.get("partial", False)),
                )
                for m in data.get("messages", [])
            ]
            return StoredSession(
                session_id=data["session_id"],
                title=data.get("title", "未命名会话"),
                created_at=data.get("created_at", 0.0),
                updated_at=data.get("updated_at", 0.0),
                messages=messages,
                pinned=bool(data.get("pinned", False)),
                style=data.get("style", "concise"),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("加载会话失败：%s (%s)", sid, exc)
            return None

    # ----- 写入 / 删除 -----
    def save(self, session: StoredSession) -> None:
        session.updated_at = time.time()
        target = self._path(session.session_id)
        tmp = target.with_suffix(".tmp")
        payload = json.dumps(asdict(session), ensure_ascii=False, indent=2)
        with self._lock:
            tmp.write_text(payload, encoding="utf-8")
            tmp.replace(target)
        logger.debug("已保存会话 %s（%d 条消息）",
                     session.session_id, len(session.messages))

    def delete(self, sid: str) -> bool:
        p = self._path(sid)
        if not p.exists():
            return False
        with self._lock:
            p.unlink()
        logger.info("已删除会话：%s", sid)
        return True

    def rename(self, sid: str, new_title: str) -> bool:
        s = self.load(sid)
        if not s:
            return False
        s.title = (new_title or "").strip() or s.title
        self.save(s)
        return True

    def set_pinned(self, sid: str, pinned: bool) -> bool:
        s = self.load(sid)
        if not s:
            return False
        s.pinned = bool(pinned)
        self.save(s)
        return True

    def import_session(self, payload: Dict[str, Any], overwrite_id: bool = True) -> str:
        """从 JSON 字典导入一个会话，返回新的 session_id。

        ``overwrite_id=True`` 时无论原 ``session_id`` 是什么，都新生成一个
        UUID，避免覆盖现有会话。
        """
        sid = self.new_session_id() if overwrite_id else payload.get("session_id") or self.new_session_id()
        messages = [
            StoredMessage(
                role=m.get("role", "user"),
                content=m.get("content", ""),
                citations=[
                    StoredCitation(
                        index=c.get("index", 0),
                        score=c.get("score"),
                        file_name=c.get("file_name"),
                        text=c.get("text", ""),
                    )
                    for c in (m.get("citations") or [])
                ],
                partial=bool(m.get("partial", False)),
            )
            for m in payload.get("messages", [])
        ]
        now = time.time()
        session = StoredSession(
            session_id=sid,
            title=payload.get("title") or "导入的会话",
            created_at=payload.get("created_at") or now,
            updated_at=now,
            messages=messages,
            pinned=bool(payload.get("pinned", False)),
            style=payload.get("style", "concise"),
        )
        self.save(session)
        logger.info("导入会话 %s（%d 条消息）", sid, len(messages))
        return sid

    # ----- 工具 -----
    @staticmethod
    def new_session_id() -> str:
        return uuid.uuid4().hex

    @staticmethod
    def auto_title(first_user_message: str, max_len: int = 20) -> str:
        msg = (first_user_message or "").strip().replace("\n", " ")
        if not msg:
            return "新会话"
        if len(msg) > max_len:
            return msg[:max_len] + "…"
        return msg


# ---------- 单例 ----------
_store: Optional[SessionStore] = None


def get_session_store() -> SessionStore:
    global _store
    if _store is None:
        from .config import get_settings
        cfg = get_settings()
        # 直接基于 data_dir 拼路径，不依赖 Settings 上可能尚未刷新的
        # `chat_history_dir` property（避免 streamlit 热重载留旧版 Settings
        # 类实例时找不到属性的坑）。
        history_dir = cfg.data_dir / "chat_history"
        _store = SessionStore(history_dir)
    return _store


# ---------- UI 层友好转换 ----------
def messages_to_dict(messages: List[StoredMessage]) -> List[Dict[str, Any]]:
    """``List[StoredMessage]`` -> Streamlit 用的 dict 列表。"""
    return [
        {
            "role": m.role,
            "content": m.content,
            "citations": [asdict(c) for c in m.citations],
            "partial": bool(getattr(m, "partial", False)),
        }
        for m in messages
    ]


def session_to_markdown(session: StoredSession) -> str:
    """把会话渲染成 Markdown，供下载。带引用编号 + 来源文件。"""
    lines: List[str] = []
    lines.append(f"# {session.title}")
    lines.append("")
    lines.append(
        f"- 会话 ID：`{session.session_id}`"
    )
    lines.append(
        f"- 创建时间：{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(session.created_at))}"
    )
    lines.append(
        f"- 更新时间：{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(session.updated_at))}"
    )
    lines.append(f"- 消息条数：{len(session.messages)}")
    lines.append("")
    lines.append("---")
    lines.append("")

    for i, m in enumerate(session.messages, start=1):
        role = "🧑 用户" if m.role == "user" else "🤖 助手"
        lines.append(f"## {i}. {role}")
        lines.append("")
        lines.append(m.content or "(空)")
        lines.append("")
        if m.citations:
            lines.append("**引用：**")
            lines.append("")
            for c in m.citations:
                score = f"{c.score:.4f}" if c.score is not None else "-"
                lines.append(
                    f"- **[{c.index}]** `{c.file_name or '未知'}` · 相关度 `{score}`"
                )
                preview = (c.text or "").strip().replace("\n", " ")
                if len(preview) > 200:
                    preview = preview[:200] + "…"
                lines.append(f"  > {preview}")
            lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


def session_to_json_bytes(session: StoredSession) -> bytes:
    """把会话序列化成 UTF-8 JSON bytes，用于 ``st.download_button``。"""
    return json.dumps(asdict(session), ensure_ascii=False, indent=2).encode("utf-8")


def dict_to_messages(messages: List[Dict[str, Any]]) -> List[StoredMessage]:
    """反过来：UI 端的 dict 列表 -> ``List[StoredMessage]``。

    UI 端的 ``citations`` 元素既可能是 ``Citation`` dataclass 实例，也可能是
    持久化后又读回来的 dict，这里两种都兼容。
    """
    out: List[StoredMessage] = []
    for m in messages:
        cits_raw = m.get("citations", []) or []
        cits: List[StoredCitation] = []
        for c in cits_raw:
            if isinstance(c, dict):
                cits.append(
                    StoredCitation(
                        index=c.get("index", 0),
                        score=c.get("score"),
                        file_name=c.get("file_name"),
                        text=c.get("text", ""),
                    )
                )
            else:
                # Citation dataclass 实例
                cits.append(
                    StoredCitation(
                        index=getattr(c, "index", 0),
                        score=getattr(c, "score", None),
                        file_name=getattr(c, "file_name", None),
                        text=getattr(c, "text", ""),
                    )
                )
        out.append(StoredMessage(
            role=m["role"],
            content=m["content"],
            citations=cits,
            partial=bool(m.get("partial", False)),
        ))
    return out
