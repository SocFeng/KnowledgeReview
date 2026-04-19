"""多轮对话引擎：自定义"浓缩历史 + 检索上下文"流程。

LangChain 没有 LlamaIndex 的 ``CondensePlusContextChatEngine``。这里手工
组装了一个等价的小流程：

1. 如果历史不为空 → 用 LLM 把"历史 + 当前问题"浓缩成独立检索问句。
2. 用浓缩后的问句调用 ``HybridRetriever`` 拿上下文（含 trace）。
3. 把 system prompt + 风格指令 + 检索片段 + 历史 + 当前问题 拼成 messages。
4. 调用 LLM；返回完整答案，或者按 token 流式吐字。

提供两种调用方式：
- ``chat()``         一次性返回完整答案（非流式，FastAPI / 单次问答用）
- ``stream_chat()``  返回 token 生成器（流式，Streamlit 打字机效果用）

新增能力：
- ``answer_style``   切换回答格式（简洁/详细/对比表格/步骤化）
- ``doc_filter``     限定本次只在指定文件名内检索
- ``last_trace``     UI 拿来做"召回可视化"的诊断信息（来自 retriever）
- ``suggest_followups()`` 根据上一轮问答生成 N 个相关追问
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from threading import Lock
from typing import Callable, Dict, Iterator, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

from .retriever import (
    RetrievalTrace,
    ScoredDoc,
    get_retriever,
    reset_doc_filter,
    set_doc_filter,
)
from .settings import get_llm, init_settings

logger = logging.getLogger(__name__)


BASE_SYSTEM_PROMPT = (
    "你是一名严谨的知识库助手。请只依据下面提供的【参考资料】回答用户问题，"
    "若资料中没有答案，请明确说『资料中未提到』，不要编造。"
    "回答需准确，并在涉及具体内容时用 [n] 的形式标注引用片段编号。"
)

STYLE_PROMPTS: Dict[str, str] = {
    "concise": "请用 1-3 句话给出简洁、要点式的回答，避免冗长解释。",
    "detailed": "请给出详细、完整的回答，必要时分段展开背景、结论、注意事项。",
    "table": (
        "请尽可能用 Markdown 表格组织答案，列与列之间清晰对比；"
        "表格之外只写 1-2 句小结。"
    ),
    "steps": (
        "请用编号列表（1./2./3./...）逐步给出操作步骤，每步只写一句话，"
        "末尾追加一段『注意事项』小结。"
    ),
}
DEFAULT_STYLE = "concise"

CONDENSE_PROMPT = (
    "下面是用户和助手的对话历史，以及用户的最新问题。"
    "请把【最新问题】改写成一个能够独立检索的完整中文问句，"
    "包含必要的上下文实体；如果最新问题本身就是独立完整的，原样返回即可。"
    "只输出改写后的问句，不要任何前后缀。"
)

EMPTY_RESPONSE_HINT = (
    "未在知识库中找到与你的问题相关的内容。可以尝试：\n"
    "• 换种说法，使用文档中可能出现的关键词\n"
    "• 在「📤 文档上传」页确认相关文档已经入库\n"
    "• 提问得更具体，例如直接引用文件中的术语"
)


def _build_system_prompt(style: str) -> str:
    extra = STYLE_PROMPTS.get(style)
    if extra:
        return f"{BASE_SYSTEM_PROMPT}\n\n【输出格式】{extra}"
    return BASE_SYSTEM_PROMPT


SYSTEM_PROMPT = BASE_SYSTEM_PROMPT  # 兼容旧代码


# ---------- 数据结构 ----------
@dataclass
class Citation:
    index: int
    score: Optional[float]
    file_name: Optional[str]
    text: str


@dataclass
class ChatResponse:
    answer: str
    citations: List[Citation] = field(default_factory=list)
    session_id: str = ""
    trace: Optional[RetrievalTrace] = None


@dataclass
class ChatStreamHandle:
    """流式对话的句柄。

    UI 层先消费 ``token_iter`` 把字逐个写到屏幕，
    全部消费完后调用 ``finalize()`` 拿到完整文本 + 引用 + session_id。
    """
    session_id: str
    token_iter: Iterator[str]
    finalize: Callable[[], ChatResponse]


# ---------- 单会话状态 ----------
@dataclass
class _Session:
    sid: str
    style: str = DEFAULT_STYLE
    history: List[BaseMessage] = field(default_factory=list)


# ---------- 主服务 ----------
class ChatService:
    """对外提供 chat() / stream_chat() / new_session() / reset() 等方法。"""

    def __init__(self) -> None:
        init_settings()
        self._retriever = get_retriever()
        self._sessions: Dict[str, _Session] = {}
        self._lock = Lock()

    # ----- 会话管理 -----
    def new_session(self, style: str = DEFAULT_STYLE) -> str:
        sid = uuid.uuid4().hex
        with self._lock:
            self._sessions[sid] = _Session(sid=sid, style=style)
        return sid

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def set_style(self, session_id: str, style: str) -> None:
        if not session_id or style not in STYLE_PROMPTS:
            return
        with self._lock:
            sess = self._sessions.get(session_id)
            if sess is not None:
                sess.style = style
        logger.info("[session=%s] 切换回答风格 -> %s", session_id, style)

    def restore_session(
        self,
        session_id: str,
        history: List[Tuple[str, str]],
        style: str = DEFAULT_STYLE,
    ) -> None:
        """从 ``[(role, content), ...]`` 恢复一个会话的历史。"""
        msgs: List[BaseMessage] = []
        for role, content in history:
            if not (content or "").strip():
                continue
            if role == "user":
                msgs.append(HumanMessage(content=content))
            elif role == "assistant":
                msgs.append(AIMessage(content=content))
            elif role == "system":
                msgs.append(SystemMessage(content=content))
        with self._lock:
            self._sessions[session_id] = _Session(
                sid=session_id, style=style, history=msgs
            )
        logger.info(
            "[session=%s] 已从持久化历史恢复，载入 %d 条消息（style=%s）",
            session_id, len(msgs), style,
        )

    def _get_or_create(
        self, session_id: Optional[str], style: str = DEFAULT_STYLE,
    ) -> _Session:
        with self._lock:
            if session_id is None:
                sid = uuid.uuid4().hex
                sess = _Session(sid=sid, style=style)
                self._sessions[sid] = sess
                return sess
            sess = self._sessions.get(session_id)
            if sess is None:
                sess = _Session(sid=session_id, style=style)
                self._sessions[session_id] = sess
            elif style and style != sess.style and style in STYLE_PROMPTS:
                sess.style = style
        return sess

    # ----- 内部工具 -----
    def _condense_question(
        self, history: List[BaseMessage], question: str,
    ) -> str:
        """有历史时把当前问题改写成独立检索问句。"""
        if not history:
            return question
        llm = get_llm()
        # 取最近若干轮（防止 prompt 过长）
        recent = history[-8:]
        history_text_lines: List[str] = []
        for m in recent:
            role = (
                "用户" if isinstance(m, HumanMessage)
                else "助手" if isinstance(m, AIMessage) else "系统"
            )
            content = (m.content or "").strip()
            if not content:
                continue
            if len(content) > 400:
                content = content[:400] + "…"
            history_text_lines.append(f"{role}：{content}")
        history_text = "\n".join(history_text_lines)
        prompt = (
            f"{CONDENSE_PROMPT}\n\n"
            f"【对话历史】\n{history_text}\n\n"
            f"【最新问题】{question}\n\n"
            "改写后的问句："
        )
        try:
            resp = llm.invoke([HumanMessage(content=prompt)])
            condensed = (resp.content or "").strip().strip('"').strip("「」")
            return condensed or question
        except Exception as exc:  # noqa: BLE001
            logger.warning("浓缩问题失败，使用原问题：%s", exc)
            return question

    @staticmethod
    def _format_context(scored: List[ScoredDoc]) -> str:
        """把检索片段拼成 LLM 可读的上下文（带 [n] 编号 + 文件名）。"""
        lines: List[str] = []
        for i, s in enumerate(scored, start=1):
            fname = (
                (s.doc.metadata or {}).get("original_name")
                or (s.doc.metadata or {}).get("file_name")
                or "未知来源"
            )
            text = s.doc.page_content or ""
            lines.append(f"[{i}] 来源：{fname}\n{text}")
        return "\n\n".join(lines)

    def _build_messages(
        self,
        sess: _Session,
        question: str,
        scored: List[ScoredDoc],
    ) -> List[BaseMessage]:
        sys_prompt = _build_system_prompt(sess.style)
        context = self._format_context(scored) if scored else "（无相关资料）"
        sys_full = f"{sys_prompt}\n\n【参考资料】\n{context}"

        msgs: List[BaseMessage] = [SystemMessage(content=sys_full)]
        # 历史只放 user / assistant；不重复贴 system
        for m in sess.history:
            if isinstance(m, (HumanMessage, AIMessage)):
                msgs.append(m)
        msgs.append(HumanMessage(content=question))
        return msgs

    def _do_retrieve(
        self,
        sess: _Session,
        message: str,
        doc_filter: Optional[List[str]],
    ) -> Tuple[List[ScoredDoc], Optional[RetrievalTrace], str]:
        """执行"浓缩 + 检索"，返回 (scored_docs, trace, condensed_question)。"""
        condensed = self._condense_question(sess.history, message)
        if condensed != message:
            logger.info("[session=%s] condensed: %r", sess.sid, condensed)
        token = set_doc_filter(doc_filter)
        try:
            scored = self._retriever.retrieve_scored(condensed)
        finally:
            reset_doc_filter(token)
        trace = getattr(self._retriever, "last_trace", None)
        return scored, trace, condensed

    @staticmethod
    def _scored_to_citations(scored: List[ScoredDoc]) -> List[Citation]:
        out: List[Citation] = []
        for i, s in enumerate(scored, start=1):
            meta = s.doc.metadata or {}
            file_name = (
                meta.get("original_name")
                or meta.get("file_name")
                or meta.get("file_path")
            )
            text = s.doc.page_content or ""
            preview = text if len(text) <= 300 else text[:300] + "..."
            out.append(Citation(
                index=i, score=s.score, file_name=file_name, text=preview,
            ))
        return out

    # ----- 同步入口 -----
    def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        doc_filter: Optional[List[str]] = None,
        style: str = DEFAULT_STYLE,
    ) -> ChatResponse:
        sess = self._get_or_create(session_id, style)
        logger.info(
            "[session=%s] Q: %s | doc_filter=%s | style=%s",
            sess.sid, message, doc_filter, style,
        )

        scored, trace, _condensed = self._do_retrieve(sess, message, doc_filter)
        if scored:
            scores = [
                f"#{i + 1}={s.score:.3f}" if s.score is not None else f"#{i + 1}=N/A"
                for i, s in enumerate(scored)
            ]
            logger.info("[session=%s] node scores: %s", sess.sid, ", ".join(scores))

        msgs = self._build_messages(sess, message, scored)
        try:
            resp = get_llm().invoke(msgs)
            answer = (resp.content or "").strip()
        except Exception as exc:  # noqa: BLE001
            logger.exception("[session=%s] LLM 调用失败：%s", sess.sid, exc)
            answer = ""

        if not answer:
            answer = EMPTY_RESPONSE_HINT
        else:
            with self._lock:
                sess.history.append(HumanMessage(content=message))
                sess.history.append(AIMessage(content=answer))

        return ChatResponse(
            answer=answer,
            citations=self._scored_to_citations(scored),
            session_id=sess.sid,
            trace=trace,
        )

    # ----- 流式入口 -----
    def stream_chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        doc_filter: Optional[List[str]] = None,
        style: str = DEFAULT_STYLE,
    ) -> ChatStreamHandle:
        sess = self._get_or_create(session_id, style)
        logger.info(
            "[session=%s] (stream) Q: %s | doc_filter=%s | style=%s",
            sess.sid, message, doc_filter, style,
        )

        scored, trace, _condensed = self._do_retrieve(sess, message, doc_filter)
        if scored:
            scores = [
                f"#{i + 1}={s.score:.3f}" if s.score is not None else f"#{i + 1}=N/A"
                for i, s in enumerate(scored)
            ]
            logger.info("[session=%s] node scores: %s", sess.sid, ", ".join(scores))
        else:
            logger.warning("[session=%s] 检索结果为空", sess.sid)

        msgs = self._build_messages(sess, message, scored)
        collected: List[str] = []
        fallback: List[str] = []

        def _gen() -> Iterator[str]:
            try:
                for chunk in get_llm().stream(msgs):
                    tok = (
                        chunk.content
                        if isinstance(chunk.content, str)
                        else _flatten_chunk(chunk.content)
                    )
                    if tok:
                        collected.append(tok)
                        yield tok
            except Exception as exc:  # noqa: BLE001
                logger.exception("[session=%s] 流式生成异常: %s", sess.sid, exc)
                yield f"\n\n[流式生成中断：{exc}]"
                return

            # 流没抛异常，但一字未吐 —— 部分 LLM 后端在某些消息下会静默失败
            if not collected:
                logger.warning(
                    "[session=%s] 流式返回 0 字符，尝试非流式重试…", sess.sid
                )
                try:
                    resp = get_llm().invoke(msgs)
                    text = (resp.content or "").strip()
                    if text:
                        fallback.append(text)
                        yield text
                        logger.info(
                            "[session=%s] 非流式重试成功，total_chars=%d",
                            sess.sid, len(text),
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.exception(
                        "[session=%s] 非流式重试也失败: %s", sess.sid, exc
                    )

        def _finalize() -> ChatResponse:
            full = "".join(collected).strip() or "".join(fallback).strip()
            logger.info(
                "[session=%s] stream done, total_chars=%d (fallback=%s)",
                sess.sid, len(full), bool(fallback),
            )
            if full:
                with self._lock:
                    sess.history.append(HumanMessage(content=message))
                    sess.history.append(AIMessage(content=full))
            else:
                full = EMPTY_RESPONSE_HINT
            return ChatResponse(
                answer=full,
                citations=self._scored_to_citations(scored),
                session_id=sess.sid,
                trace=trace,
            )

        return ChatStreamHandle(
            session_id=sess.sid,
            token_iter=_gen(),
            finalize=_finalize,
        )

    # ----- 追问建议 -----
    def suggest_followups(
        self,
        question: str,
        answer: str,
        n: int = 3,
    ) -> List[str]:
        if not (question and answer):
            return []
        prompt = (
            "你是一名知识助手。基于下面给出的【用户问题】和【已给答案】，"
            f"提出 {n} 个用户接下来很可能想继续追问的、自然衔接的中文问题。\n\n"
            "要求：\n"
            "1) 每个问题不超过 25 字；\n"
            "2) 不重复用户原问题；\n"
            "3) 严格只返回一个 JSON 数组，"
            "例如 [\"问题1\", \"问题2\", \"问题3\"]，不要任何解释。\n\n"
            f"【用户问题】{question}\n"
            f"【已给答案】{answer[:800]}\n"
        )
        try:
            resp = get_llm().invoke([HumanMessage(content=prompt)])
            raw = (resp.content or "").strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("追问生成失败：%s", exc)
            return []

        followups = self._parse_followups(raw, n)
        logger.info("生成追问 %d 条：%s", len(followups), followups)
        return followups

    @staticmethod
    def _parse_followups(raw: str, n: int) -> List[str]:
        try:
            arr = json.loads(raw)
            if isinstance(arr, list):
                return [str(x).strip() for x in arr if str(x).strip()][:n]
        except Exception:  # noqa: BLE001
            pass
        m = re.search(r"\[.*?\]", raw, flags=re.S)
        if m:
            try:
                arr = json.loads(m.group(0))
                if isinstance(arr, list):
                    return [str(x).strip() for x in arr if str(x).strip()][:n]
            except Exception:  # noqa: BLE001
                pass
        items: List[str] = []
        for line in raw.splitlines():
            line = re.sub(r"^[\s\-\*\d\.\、\)\(]+", "", line).strip().strip('"').strip("「」")
            if line:
                items.append(line)
            if len(items) >= n:
                break
        return items[:n]


def _flatten_chunk(content) -> str:
    """LangChain 某些后端把 stream 内容包成 list[dict]；这里抽出文本。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out_parts: List[str] = []
        for x in content:
            if isinstance(x, str):
                out_parts.append(x)
            elif isinstance(x, dict):
                t = x.get("text") or x.get("content") or ""
                if isinstance(t, str):
                    out_parts.append(t)
        return "".join(out_parts)
    return ""


_service: Optional[ChatService] = None


def get_chat_service() -> ChatService:
    global _service
    if _service is None:
        _service = ChatService()
    return _service


def reset_chat_service() -> None:
    """ingest 完成后调用：让下一次 get_chat_service 重新连 retriever。"""
    global _service
    _service = None
