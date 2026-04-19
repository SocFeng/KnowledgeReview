"""多轮对话引擎：基于 CondensePlusContextChatEngine，支持会话历史 + 引用溯源。

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

from llama_index.core import Settings as LISettings
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.schema import NodeWithScore

from .retriever import RetrievalTrace, get_retriever, reset_doc_filter, set_doc_filter
from .settings import init_settings

logger = logging.getLogger(__name__)


BASE_SYSTEM_PROMPT = (
    "你是一名严谨的知识库助手。请只依据下面提供的【参考资料】回答用户问题，"
    "若资料中没有答案，请明确说『资料中未提到』，不要编造。"
    "回答需准确，并在涉及具体内容时用 [n] 的形式标注引用片段编号。"
)

# 不同回答风格对应追加的指令；UI 切换时在 system_prompt 末尾拼一段
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


def _build_system_prompt(style: str) -> str:
    extra = STYLE_PROMPTS.get(style)
    if extra:
        return f"{BASE_SYSTEM_PROMPT}\n\n【输出格式】{extra}"
    return BASE_SYSTEM_PROMPT


# 兼容老代码：保留这个常量名（早期测试脚本可能 import）
SYSTEM_PROMPT = BASE_SYSTEM_PROMPT

# CondensePlusContextChatEngine 在检索为空时会返回这个字面量字符串
EMPTY_RESPONSE_SENTINEL = "Empty Response"
EMPTY_RESPONSE_HINT = (
    "未在知识库中找到与你的问题相关的内容。可以尝试：\n"
    "• 换种说法，使用文档中可能出现的关键词\n"
    "• 在「📤 文档上传」页确认相关文档已经入库\n"
    "• 提问得更具体，例如直接引用文件中的术语"
)


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


class ChatService:
    """对外提供 chat() / new_session() / reset() 等方法。"""

    def __init__(self) -> None:
        init_settings()
        self._retriever = get_retriever()
        self._sessions: Dict[str, CondensePlusContextChatEngine] = {}
        self._session_styles: Dict[str, str] = {}
        self._lock = Lock()

    # ---------- 会话管理 ----------
    def new_session(self, style: str = DEFAULT_STYLE) -> str:
        session_id = uuid.uuid4().hex
        with self._lock:
            self._sessions[session_id] = self._build_engine(style)
            self._session_styles[session_id] = style
        return session_id

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
            self._session_styles.pop(session_id, None)

    def set_style(self, session_id: str, style: str) -> None:
        """切换某会话的回答风格：保留 memory，重建 engine 应用新 system_prompt。"""
        if not session_id or style not in STYLE_PROMPTS:
            return
        with self._lock:
            old = self._sessions.get(session_id)
            history: List[ChatMessage] = []
            if old is not None:
                try:
                    history = list(old.memory.get_all())
                except Exception:  # noqa: BLE001
                    history = []
            memory = ChatMemoryBuffer.from_defaults(
                token_limit=3000, chat_history=history
            )
            self._sessions[session_id] = CondensePlusContextChatEngine.from_defaults(
                retriever=self._retriever,
                memory=memory,
                system_prompt=_build_system_prompt(style),
            )
            self._session_styles[session_id] = style
        logger.info("[session=%s] 切换回答风格 -> %s", session_id, style)

    def restore_session(
        self,
        session_id: str,
        history: List[Tuple[str, str]],
        style: str = DEFAULT_STYLE,
    ) -> None:
        """从已有的对话历史恢复一个会话的 engine memory。

        ``history`` 是 ``[(role, content), ...]`` 列表，role 取值
        ``"user"`` / ``"assistant"`` / ``"system"``，按时间顺序排列。

        恢复后，下一次 ``chat()`` / ``stream_chat()`` 调用就能感知历史
        上下文，实现"切换到历史会话也能接着聊"的效果。
        """
        role_map = {
            "user": MessageRole.USER,
            "assistant": MessageRole.ASSISTANT,
            "system": MessageRole.SYSTEM,
        }
        chat_history = [
            ChatMessage(role=role_map.get(role, MessageRole.USER), content=content)
            for role, content in history
            if (content or "").strip()
        ]
        memory = ChatMemoryBuffer.from_defaults(
            token_limit=3000, chat_history=chat_history
        )
        engine = CondensePlusContextChatEngine.from_defaults(
            retriever=self._retriever,
            memory=memory,
            system_prompt=_build_system_prompt(style),
        )
        with self._lock:
            self._sessions[session_id] = engine
            self._session_styles[session_id] = style
        logger.info(
            "[session=%s] 已从持久化历史恢复，载入 %d 条消息（style=%s）",
            session_id, len(chat_history), style,
        )

    def _get_or_create(
        self, session_id: Optional[str], style: str = DEFAULT_STYLE,
    ) -> tuple[str, CondensePlusContextChatEngine]:
        if session_id is None or session_id not in self._sessions:
            sid = self.new_session(style) if session_id is None else session_id
            with self._lock:
                if sid not in self._sessions:
                    self._sessions[sid] = self._build_engine(style)
                    self._session_styles[sid] = style
            return sid, self._sessions[sid]
        # 已存在但风格不一致 → 切换
        if self._session_styles.get(session_id) != style and style in STYLE_PROMPTS:
            self.set_style(session_id, style)
        return session_id, self._sessions[session_id]

    def _build_engine(self, style: str = DEFAULT_STYLE) -> CondensePlusContextChatEngine:
        memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
        return CondensePlusContextChatEngine.from_defaults(
            retriever=self._retriever,
            memory=memory,
            system_prompt=_build_system_prompt(style),
        )

    # ---------- 主入口 ----------
    def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        doc_filter: Optional[List[str]] = None,
        style: str = DEFAULT_STYLE,
    ) -> ChatResponse:
        sid, engine = self._get_or_create(session_id, style)
        logger.info(
            "[session=%s] Q: %s | doc_filter=%s | style=%s",
            sid, message, doc_filter, style,
        )

        token = set_doc_filter(doc_filter)
        try:
            response = engine.chat(message)
        finally:
            reset_doc_filter(token)

        source_nodes = getattr(response, "source_nodes", []) or []
        citations = self._extract_citations(source_nodes)
        trace = getattr(self._retriever, "last_trace", None)

        # 真实 LLM 输出（可能为 None / 空串 / "Empty Response"）
        raw_response = getattr(response, "response", None)
        answer = str(response).strip()

        logger.info(
            "[session=%s] retrieved=%d nodes, response_type=%s, "
            "raw_response=%r, str_len=%d",
            sid,
            len(source_nodes),
            type(response).__name__,
            (raw_response[:200] + "...")
            if isinstance(raw_response, str) and len(raw_response) > 200
            else raw_response,
            len(answer),
        )

        # 把检索片段的相关度也打出来，帮助判断是否检索质量太差
        if source_nodes:
            scores = [
                f"#{i + 1}={n.score:.3f}" if n.score is not None else f"#{i + 1}=N/A"
                for i, n in enumerate(source_nodes)
            ]
            logger.info("[session=%s] node scores: %s", sid, ", ".join(scores))

        # 检索为空 / LLM 没生成内容时的兜底
        if not answer or answer == EMPTY_RESPONSE_SENTINEL:
            answer = EMPTY_RESPONSE_HINT

        return ChatResponse(
            answer=answer,
            citations=citations,
            session_id=sid,
            trace=trace,
        )

    # ---------- 流式入口 ----------
    def stream_chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        doc_filter: Optional[List[str]] = None,
        style: str = DEFAULT_STYLE,
    ) -> ChatStreamHandle:
        """流式对话：返回一个可迭代的 token 生成器 + 收尾回调。

        典型用法（Streamlit）::

            handle = service.stream_chat(question, sid, doc_filter=...)
            full_text = st.write_stream(handle.token_iter)
            result = handle.finalize()  # ChatResponse(answer=full_text, citations=...)
        """
        sid, engine = self._get_or_create(session_id, style)
        logger.info(
            "[session=%s] (stream) Q: %s | doc_filter=%s | style=%s",
            sid, message, doc_filter, style,
        )

        # engine.stream_chat 内部已经先做了检索，再启动 LLM 流。
        # 因此 source_nodes 在返回时就已经填充。doc_filter 的作用域在
        # set_doc_filter() 设置的 contextvar 上，整段流期间都生效。
        token = set_doc_filter(doc_filter)
        try:
            stream_resp = engine.stream_chat(message)
        except Exception:
            reset_doc_filter(token)
            raise
        source_nodes = list(getattr(stream_resp, "source_nodes", []) or [])
        trace = getattr(self._retriever, "last_trace", None)

        if source_nodes:
            scores = [
                f"#{i + 1}={n.score:.3f}" if n.score is not None else f"#{i + 1}=N/A"
                for i, n in enumerate(source_nodes)
            ]
            logger.info("[session=%s] node scores: %s", sid, ", ".join(scores))
        else:
            logger.warning("[session=%s] 检索结果为空", sid)

        collected: List[str] = []
        fallback_text: List[str] = []  # 流失败时的兜底答案

        def _gen() -> Iterator[str]:
            try:
                for tok in stream_resp.response_gen:
                    if tok:
                        collected.append(tok)
                        yield tok
            except Exception as exc:  # noqa: BLE001
                logger.exception("[session=%s] 流式生成异常: %s", sid, exc)
                yield f"\n\n[流式生成中断：{exc}]"
                reset_doc_filter(token)
                return

            # 流没抛异常，但一字未吐 —— 这是 DashScope stream 模式
            # 在某些 messages 序列下静默失败的情况。
            # 兜底：用非流式 chat() 重试一次，把答案一次性"流"出来。
            if not collected:
                logger.warning(
                    "[session=%s] 流式返回 0 字符，尝试非流式重试…", sid
                )
                try:
                    retry = engine.chat(message)
                    retry_text = str(retry).strip()
                    if retry_text and retry_text != EMPTY_RESPONSE_SENTINEL:
                        fallback_text.append(retry_text)
                        # 一次性 yield，UI 端会一次性渲染
                        yield retry_text
                        logger.info(
                            "[session=%s] 非流式重试成功，total_chars=%d",
                            sid,
                            len(retry_text),
                        )
                    else:
                        logger.warning(
                            "[session=%s] 非流式重试也返回空", sid
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.exception(
                        "[session=%s] 非流式重试也失败: %s", sid, exc
                    )

        def _finalize() -> ChatResponse:
            try:
                full = "".join(collected).strip() or "".join(fallback_text).strip()
                logger.info(
                    "[session=%s] stream done, total_chars=%d (fallback=%s)",
                    sid,
                    len(full),
                    bool(fallback_text),
                )
                if not full:
                    full = EMPTY_RESPONSE_HINT
                return ChatResponse(
                    answer=full,
                    citations=self._extract_citations(source_nodes),
                    session_id=sid,
                    trace=trace,
                )
            finally:
                reset_doc_filter(token)

        return ChatStreamHandle(
            session_id=sid,
            token_iter=_gen(),
            finalize=_finalize,
        )

    # ---------- 追问建议 ----------
    def suggest_followups(
        self,
        question: str,
        answer: str,
        n: int = 3,
    ) -> List[str]:
        """根据上一轮 Q&A 生成 ``n`` 个相关追问。

        采用一次性 ``LLM.complete`` 调用 + JSON 解析；解析失败时回退
        到按行/按编号提取，再不行就返回空列表（UI 端不渲染按钮即可）。
        """
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
            llm = LISettings.llm
            raw = str(llm.complete(prompt)).strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("追问生成失败：%s", exc)
            return []

        # 优先尝试 JSON 解析
        followups = self._parse_followups(raw, n)
        logger.info("生成追问 %d 条：%s", len(followups), followups)
        return followups

    @staticmethod
    def _parse_followups(raw: str, n: int) -> List[str]:
        # 1) 直接 JSON
        try:
            arr = json.loads(raw)
            if isinstance(arr, list):
                return [str(x).strip() for x in arr if str(x).strip()][:n]
        except Exception:  # noqa: BLE001
            pass
        # 2) 在文本里抓 [ ... ] 段
        m = re.search(r"\[.*?\]", raw, flags=re.S)
        if m:
            try:
                arr = json.loads(m.group(0))
                if isinstance(arr, list):
                    return [str(x).strip() for x in arr if str(x).strip()][:n]
            except Exception:  # noqa: BLE001
                pass
        # 3) 按行 + 编号
        items: List[str] = []
        for line in raw.splitlines():
            line = re.sub(r"^[\s\-\*\d\.\、\)\(]+", "", line).strip().strip('"').strip("「」")
            if line:
                items.append(line)
            if len(items) >= n:
                break
        return items[:n]

    @staticmethod
    def _extract_citations(nodes: List[NodeWithScore]) -> List[Citation]:
        citations: List[Citation] = []
        for i, node_with_score in enumerate(nodes, start=1):
            node = node_with_score.node
            file_name = node.metadata.get("file_name") or node.metadata.get("file_path")
            text = node.get_content()
            preview = text if len(text) <= 300 else text[:300] + "..."
            citations.append(
                Citation(
                    index=i,
                    score=node_with_score.score,
                    file_name=file_name,
                    text=preview,
                )
            )
        return citations


_service: ChatService | None = None


def get_chat_service() -> ChatService:
    global _service
    if _service is None:
        _service = ChatService()
    return _service
