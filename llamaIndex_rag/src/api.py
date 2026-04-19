"""FastAPI HTTP 接口层。

完整对齐 ``ChatService`` / ``DocStore`` / ``SessionStore`` 的全部能力，
让外部程序无需进入 Python 进程也能用上：
- 多轮对话（同步 + SSE 流式）
- 检索范围过滤、回答风格切换、追问生成、召回 trace
- 文档管理（列表 / 上传入库 / 删除）
- 会话管理（CRUD / 重命名 / 置顶 / 导出 Markdown / JSON / 导入）
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from .chat import Citation, get_chat_service
from .config import get_settings
from .doc_store import bytes_sha1, get_doc_store, reset_doc_store
from .ingest import SUPPORTED_SUFFIXES, build_index_with_progress
from .sessions import (
    StoredSession,
    dict_to_messages,
    get_session_store,
    messages_to_dict,
    session_to_json_bytes,
    session_to_markdown,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Schemas
# ============================================================================
class ChatRequest(BaseModel):
    message: str = Field(..., description="用户的本轮提问")
    session_id: Optional[str] = Field(
        None, description="会话 ID；不传则自动新建一个会话"
    )
    doc_filter: Optional[List[str]] = Field(
        None,
        description="只在这些文件名（original_name）中检索；空/不传 = 全部",
    )
    style: str = Field(
        "concise",
        description="回答格式：concise / detailed / table / steps",
    )


class CitationModel(BaseModel):
    index: int
    score: Optional[float] = None
    file_name: Optional[str] = None
    text: str

    @classmethod
    def from_dataclass(cls, c: Citation) -> "CitationModel":
        return cls(index=c.index, score=c.score, file_name=c.file_name, text=c.text)


class TraceHit(BaseModel):
    file_name: Optional[str] = None
    score: Optional[float] = None
    preview: str = ""
    node_id: str = ""


class TraceModel(BaseModel):
    vector_hits: List[TraceHit] = Field(default_factory=list)
    bm25_hits: List[TraceHit] = Field(default_factory=list)
    fused_hits: List[TraceHit] = Field(default_factory=list)
    rerank_hits: List[TraceHit] = Field(default_factory=list)
    rerank_used: bool = False
    doc_filter: Optional[List[str]] = None


class ChatResponseModel(BaseModel):
    session_id: str
    answer: str
    citations: List[CitationModel]
    trace: Optional[TraceModel] = None


class FollowupsRequest(BaseModel):
    question: str
    answer: str
    n: int = 3


class FollowupsResponseModel(BaseModel):
    suggestions: List[str]


class SessionResponse(BaseModel):
    session_id: str


class SessionMetaModel(BaseModel):
    session_id: str
    title: str
    created_at: float
    updated_at: float
    message_count: int
    pinned: bool = False


class StoredMessageModel(BaseModel):
    role: str
    content: str
    citations: List[CitationModel] = Field(default_factory=list)


class StoredSessionModel(BaseModel):
    session_id: str
    title: str
    created_at: float
    updated_at: float
    pinned: bool = False
    style: str = "concise"
    messages: List[StoredMessageModel] = Field(default_factory=list)

    @classmethod
    def from_stored(cls, s: StoredSession) -> "StoredSessionModel":
        return cls(
            session_id=s.session_id,
            title=s.title,
            created_at=s.created_at,
            updated_at=s.updated_at,
            pinned=s.pinned,
            style=s.style,
            messages=[
                StoredMessageModel(
                    role=m.role,
                    content=m.content,
                    citations=[
                        CitationModel(
                            index=c.index,
                            score=c.score,
                            file_name=c.file_name,
                            text=c.text,
                        )
                        for c in m.citations
                    ],
                )
                for m in s.messages
            ],
        )


class SessionPatch(BaseModel):
    title: Optional[str] = Field(None, description="新标题；不传则不修改")
    pinned: Optional[bool] = Field(None, description="是否置顶；不传则不修改")


class DocumentModel(BaseModel):
    file_hash: str
    original_name: str
    saved_path: Optional[str] = None
    node_count: int
    uploaded_at: Optional[float] = None
    size_bytes: int = 0


class IngestResponseModel(BaseModel):
    files: int = Field(..., description="本次实际入库的文件数")
    nodes: int = Field(..., description="新增节点数")
    skipped: int = Field(0, description="按内容 hash 去重跳过的文件数")
    skipped_files: List[str] = Field(default_factory=list)
    elapsed: float = Field(..., description="耗时秒")


# ============================================================================
# App & lifespan
# ============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("正在初始化 RAG 服务……")
    try:
        get_chat_service()
        logger.info("RAG 服务就绪。")
    except Exception as exc:  # noqa: BLE001
        # 没有任何文档入库时 retriever 会抛 FileNotFoundError，
        # 这种情况下不应阻止 API 启动——文档接口还能用来上传。
        logger.warning("ChatService 预热失败（向量库可能为空）：%s", exc)
    yield


app = FastAPI(
    title="LlamaIndex + Qwen RAG API",
    version="0.2.0",
    description=(
        "基于 LlamaIndex、DashScope (Qwen) 与 ChromaDB 的检索增强问答服务。"
        "完整暴露文档管理 / 多轮对话（含流式）/ 会话管理 / 召回 trace 等能力。"
    ),
    lifespan=lifespan,
)


# ============================================================================
# Health
# ============================================================================
@app.get("/health", tags=["meta"])
def health() -> dict:
    cfg = get_settings()
    try:
        total = get_doc_store().total_nodes()
    except Exception:  # noqa: BLE001
        total = 0
    return {
        "status": "ok",
        "version": app.version,
        "llm_model": cfg.llm_model,
        "embedding_provider": cfg.embedding_provider,
        "vector_count": total,
    }


# ============================================================================
# Chat
# ============================================================================
def _trace_to_model(trace) -> Optional[TraceModel]:
    if trace is None:
        return None
    return TraceModel(
        vector_hits=[TraceHit(**h) for h in trace.vector_hits],
        bm25_hits=[TraceHit(**h) for h in trace.bm25_hits],
        fused_hits=[TraceHit(**h) for h in trace.fused_hits],
        rerank_hits=[TraceHit(**h) for h in trace.rerank_hits],
        rerank_used=trace.rerank_used,
        doc_filter=trace.doc_filter,
    )


@app.post("/chat", response_model=ChatResponseModel, tags=["chat"])
def chat(req: ChatRequest) -> ChatResponseModel:
    """同步多轮对话。返回完整答案 + 引用 + 召回 trace。"""
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message 不能为空")

    try:
        result = get_chat_service().chat(
            req.message,
            session_id=req.session_id,
            doc_filter=req.doc_filter,
            style=req.style,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    return ChatResponseModel(
        session_id=result.session_id,
        answer=result.answer,
        citations=[CitationModel.from_dataclass(c) for c in result.citations],
        trace=_trace_to_model(result.trace),
    )


@app.post("/chat/stream", tags=["chat"])
def chat_stream(req: ChatRequest):
    """SSE 流式对话。

    协议（每行 ``data: <json>\\n\\n``）：

    - ``{"event": "token", "text": "..."}``  每次新 token
    - ``{"event": "done",  "session_id": "...", "answer": "...",``
      ``"citations": [...], "trace": {...}}``  收尾
    - ``{"event": "error", "message": "..."}``  出错
    """
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message 不能为空")

    def _format_event(payload: dict) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    def _generator():
        try:
            handle = get_chat_service().stream_chat(
                req.message,
                session_id=req.session_id,
                doc_filter=req.doc_filter,
                style=req.style,
            )
        except FileNotFoundError as exc:
            yield _format_event({"event": "error", "message": str(exc)})
            return
        except Exception as exc:  # noqa: BLE001
            yield _format_event({"event": "error", "message": f"调用失败：{exc}"})
            return

        try:
            for tok in handle.token_iter:
                if tok:
                    yield _format_event({"event": "token", "text": tok})
        except Exception as exc:  # noqa: BLE001
            yield _format_event({"event": "error", "message": f"流式中断：{exc}"})
            return

        try:
            result = handle.finalize()
        except Exception as exc:  # noqa: BLE001
            yield _format_event({"event": "error", "message": f"finalize 失败：{exc}"})
            return

        trace_model = _trace_to_model(result.trace)
        yield _format_event({
            "event": "done",
            "session_id": result.session_id,
            "answer": result.answer,
            "citations": [
                CitationModel.from_dataclass(c).model_dump() for c in result.citations
            ],
            "trace": trace_model.model_dump() if trace_model else None,
        })

    return StreamingResponse(
        _generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 关掉 Nginx 缓冲
        },
    )


@app.post("/chat/followups", response_model=FollowupsResponseModel, tags=["chat"])
def chat_followups(req: FollowupsRequest) -> FollowupsResponseModel:
    """根据上一轮 Q&A 生成 N 个相关追问。"""
    items = get_chat_service().suggest_followups(req.question, req.answer, n=req.n)
    return FollowupsResponseModel(suggestions=items)


# ============================================================================
# Sessions
# ============================================================================
@app.get("/sessions", response_model=List[SessionMetaModel], tags=["sessions"])
def list_sessions(
    q: str = Query("", description="按标题或消息内容关键字过滤"),
) -> List[SessionMetaModel]:
    metas = get_session_store().list(query=q)
    return [
        SessionMetaModel(
            session_id=m.session_id,
            title=m.title,
            created_at=m.created_at,
            updated_at=m.updated_at,
            message_count=m.message_count,
            pinned=m.pinned,
        )
        for m in metas
    ]


@app.post("/sessions", response_model=SessionResponse, tags=["sessions"])
def create_session() -> SessionResponse:
    sid = get_chat_service().new_session()
    return SessionResponse(session_id=sid)


@app.get("/sessions/{session_id}", response_model=StoredSessionModel, tags=["sessions"])
def get_session(session_id: str) -> StoredSessionModel:
    s = get_session_store().load(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="会话不存在")
    return StoredSessionModel.from_stored(s)


@app.patch("/sessions/{session_id}", response_model=StoredSessionModel, tags=["sessions"])
def patch_session(session_id: str, body: SessionPatch) -> StoredSessionModel:
    """重命名 / 置顶（两个字段都可选）。"""
    store = get_session_store()
    s = store.load(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="会话不存在")
    if body.title is not None and body.title.strip():
        store.rename(session_id, body.title.strip())
    if body.pinned is not None:
        store.set_pinned(session_id, body.pinned)
    return StoredSessionModel.from_stored(store.load(session_id))


@app.delete("/sessions/{session_id}", status_code=204, tags=["sessions"])
def delete_session(session_id: str) -> None:
    """同时清掉持久化文件 + 后端 engine memory。"""
    get_session_store().delete(session_id)
    try:
        get_chat_service().reset(session_id)
    except Exception:  # noqa: BLE001
        pass


@app.get("/sessions/{session_id}/export", tags=["sessions"])
def export_session(
    session_id: str,
    format: str = Query("json", description="json 或 md"),
):
    """导出单个会话。返回 attachment 形式的下载。"""
    s = get_session_store().load(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="会话不存在")

    fmt = format.lower()
    if fmt == "md" or fmt == "markdown":
        text = session_to_markdown(s)
        return PlainTextResponse(
            text,
            media_type="text/markdown; charset=utf-8",
            headers={
                "Content-Disposition":
                    f'attachment; filename="{s.title or session_id}.md"',
            },
        )
    if fmt == "json":
        body = session_to_json_bytes(s)
        return JSONResponse(
            content=json.loads(body.decode("utf-8")),
            headers={
                "Content-Disposition": f'attachment; filename="{session_id}.json"',
            },
        )
    raise HTTPException(status_code=400, detail="format 必须是 json 或 md")


@app.post("/sessions/import", response_model=SessionResponse, tags=["sessions"])
def import_session(payload: Dict[str, Any]) -> SessionResponse:
    """从 JSON 体导入会话；总是分配新 session_id 避免覆盖已有。"""
    try:
        sid = get_session_store().import_session(payload, overwrite_id=True)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"导入失败：{exc}")
    return SessionResponse(session_id=sid)


# ============================================================================
# Documents
# ============================================================================
@app.get("/documents", response_model=List[DocumentModel], tags=["documents"])
def list_documents() -> List[DocumentModel]:
    """列出向量库中按 file_hash 聚合的文档（含每个文件占多少节点）。"""
    docs = get_doc_store().list_documents()
    return [
        DocumentModel(
            file_hash=d.file_hash,
            original_name=d.original_name,
            saved_path=d.saved_path,
            node_count=d.node_count,
            uploaded_at=d.uploaded_at,
            size_bytes=d.size_bytes,
        )
        for d in docs
    ]


@app.delete("/documents/{file_hash}", tags=["documents"])
def delete_document(file_hash: str, delete_file: bool = Query(True)) -> dict:
    """按 ``file_hash`` 删除：

    - 始终清理 Chroma + docstore 中的所有匹配节点
    - ``delete_file=True``（默认）时同时删除磁盘上的原始文件
    """
    store = get_doc_store()
    stat = store.find_by_hash(file_hash)
    if not stat:
        raise HTTPException(status_code=404, detail="未找到对应文档")

    file_deleted = False
    if delete_file and stat.saved_path:
        p = Path(stat.saved_path)
        if p.exists():
            try:
                p.unlink()
                file_deleted = True
            except Exception as exc:  # noqa: BLE001
                logger.warning("删除原始文件失败：%s", exc)

    deleted_nodes = store.delete_by_hash(file_hash)
    reset_doc_store()
    return {
        "file_hash": file_hash,
        "deleted_nodes": deleted_nodes,
        "deleted_file": file_deleted,
    }


@app.post("/documents/upload", response_model=IngestResponseModel, tags=["documents"])
async def upload_documents(
    files: List[UploadFile] = File(..., description="一个或多个文件"),
    skip_existing: bool = Form(True, description="按内容 hash 去重，重复跳过"),
) -> IngestResponseModel:
    """上传一组文件 → 落盘到 ``data/uploads/`` → 调用 ingest 入库。

    返回处理结果（文件数 / 节点数 / 跳过数 / 耗时）。
    """
    cfg = get_settings()
    upload_dir = cfg.data_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved: List[Path] = []
    original_names: Dict[str, str] = {}
    skipped_dups: List[str] = []

    doc_store = get_doc_store()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    for f in files:
        suffix = Path(f.filename or "").suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件类型 {suffix}（支持 {sorted(SUPPORTED_SUFFIXES)}）",
            )
        data = await f.read()
        h = bytes_sha1(data)
        if skip_existing and doc_store.has_hash(h):
            skipped_dups.append(f.filename or "")
            continue
        out_path = upload_dir / f"{timestamp}_{f.filename}"
        out_path.write_bytes(data)
        saved.append(out_path)
        original_names[str(out_path)] = f.filename or out_path.name

    if not saved:
        return IngestResponseModel(
            files=0,
            nodes=0,
            skipped=len(skipped_dups),
            skipped_files=skipped_dups,
            elapsed=0.0,
        )

    try:
        stats = build_index_with_progress(
            file_paths=saved,
            original_names=original_names,
            skip_existing=skip_existing,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"入库失败：{exc}")

    # ingest 内部也有 hash 去重，把两层 skip 合并起来
    all_skipped = skipped_dups + list(stats.get("skipped_files") or [])
    return IngestResponseModel(
        files=stats.get("files", 0),
        nodes=stats.get("nodes", 0),
        skipped=len(all_skipped),
        skipped_files=all_skipped,
        elapsed=float(stats.get("elapsed", 0.0)),
    )


# ============================================================================
# Entry
# ============================================================================
def run() -> None:
    """通过 ``python -m scripts.run_api`` 调用。"""
    import uvicorn

    cfg = get_settings()
    uvicorn.run(
        "src.api:app",
        host=cfg.api_host,
        port=cfg.api_port,
        reload=False,
    )
