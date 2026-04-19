"""Streamlit Web UI：

- 📤 文档管理：上传文件 → 解析切片 → 写入向量库；展示已入库文件 + 向量节点统计；
  支持联动删除（删原始文件同时清理 Chroma + docstore 中对应节点）。
- 💬 知识问答：基于已构建的知识库进行多轮对话，提供：
    * 检索范围多选（限定在选中的文件内检索）
    * 回答格式开关（简洁/详细/对比表格/步骤化）
    * AI 状态指示（检索中/思考中/回答中）
    * 引用 [n] 锚点跳转 + 高亮
    * 召回片段可视化面板（vector/BM25/融合/rerank 四阶段命中分数）
    * 追问建议按钮（点击即发）
    * 会话管理：选择/新建/删除/重命名/搜索/固定/导入/导出

侧边栏 radio 切换页面（不用 ``st.tabs``，否则 ``st.chat_input`` 不会
自动固定到屏幕底部）。

启动：
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import List, Optional

import streamlit as st

# 显式开启 INFO 级别，让 src.chat / src.retriever 里的 logger.info() 能打到控制台。
# 注意要在 import 我们的模块之前执行（避免被库内的 basicConfig 抢先）。
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
# 把噪音库的级别压回 WARNING
for noisy in ("httpx", "httpcore", "urllib3", "chromadb", "sentence_transformers"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
# 反过来：把 dashscope SDK 拉到 DEBUG，让 stream 失败时的真实错误能打印出来
logging.getLogger("dashscope").setLevel(logging.DEBUG)

from src.chat import Citation, STYLE_PROMPTS, get_chat_service  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.doc_store import bytes_sha1, get_doc_store, reset_doc_store  # noqa: E402
from src.ingest import SUPPORTED_SUFFIXES, build_index_with_progress  # noqa: E402
from src.sessions import (  # noqa: E402
    StoredSession,
    dict_to_messages,
    get_session_store,
    messages_to_dict,
    session_to_json_bytes,
    session_to_markdown,
)


STYLE_LABELS = {
    "concise": "🎯 简洁",
    "detailed": "📖 详细",
    "table": "📊 对比表格",
    "steps": "🪜 步骤化",
}

st.set_page_config(
    page_title="LlamaIndex RAG 知识库管理",
    page_icon="📚",
    layout="wide",
)


# ---------- 工具函数 ----------
def _save_uploaded_files(uploaded_files, target_dir: Path):
    """把 Streamlit 上传的文件落盘到 data/uploads/ 下，避免重名。

    会做"内容级去重"：上传内容 hash 已存在于向量库时，不重复落盘也不入库，
    UI 端展示一条提示。

    返回 ``(saved, original_names, skipped_dups)``：
        - ``saved``: 落盘后的 ``Path`` 列表
        - ``original_names``: ``{str(saved_path): original_filename}``
        - ``skipped_dups``: 已存在的 ``[(filename, existing_doc_stat), ...]``
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    original_names: dict = {}
    skipped: list = []
    doc_store = get_doc_store()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    for f in uploaded_files:
        suffix = Path(f.name).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            st.warning(f"跳过不支持的文件：{f.name}（仅支持 {sorted(SUPPORTED_SUFFIXES)}）")
            continue
        data = f.getbuffer().tobytes()
        h = bytes_sha1(data)
        existing = doc_store.find_by_hash(h)
        if existing:
            skipped.append((f.name, existing))
            continue
        out_path = target_dir / f"{timestamp}_{f.name}"
        out_path.write_bytes(data)
        saved.append(out_path)
        original_names[str(out_path)] = f.name
    return saved, original_names, skipped


def _get_collection_count() -> int:
    """读取当前 Chroma collection 中已索引的向量数。"""
    import chromadb

    cfg = get_settings()
    try:
        client = chromadb.PersistentClient(path=str(cfg.chroma_dir))
        col = client.get_or_create_collection(name=cfg.chroma_collection)
        return col.count()
    except Exception:
        return 0


def _format_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 ** 2:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / 1024 ** 2:.2f} MB"


def _list_persisted_files(upload_dir: Path) -> List[Path]:
    """扫描磁盘上已落盘的上传文件（按修改时间倒序）。"""
    if not upload_dir.exists():
        return []
    files = [
        p for p in upload_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


# ---------- 页面 ----------
def render_sidebar(cfg, page_key: str) -> None:
    with st.sidebar:
        st.header("📊 知识库状态")
        if st.button("🔄 刷新", use_container_width=True):
            reset_doc_store()
            st.rerun()
        st.metric("已索引向量数", _get_collection_count())

        # 在 chat 页时，提供"回答风格 + 检索范围"两组开关
        if page_key == "chat":
            st.divider()
            st.subheader("⚙️ 对话设置")
            style_keys = list(STYLE_LABELS.keys())
            # 确保默认值在选项里（防止旧 session_state 残留非法值）
            if st.session_state.get("answer_style") not in style_keys:
                st.session_state["answer_style"] = "concise"
            st.radio(
                "回答格式",
                style_keys,
                format_func=lambda k: STYLE_LABELS[k],
                key="answer_style",
                on_change=_on_style_change,
            )
            st.caption(STYLE_PROMPTS.get(st.session_state.answer_style, ""))

            st.divider()
            st.subheader("🎯 检索范围")
            doc_options = _list_doc_options()
            # 校准 widget state：移除已被删除的文档名，避免 multiselect 报错
            cur = [d for d in (st.session_state.get("doc_filter") or [])
                   if d in doc_options]
            if cur != st.session_state.get("doc_filter"):
                st.session_state["doc_filter"] = cur
            st.multiselect(
                "只在选中的文档中检索（留空=全部）",
                doc_options,
                key="doc_filter",
                placeholder="不选 = 全部文档参与检索",
            )

        st.divider()
        st.subheader("当前配置")
        st.write(f"**LLM**：`{cfg.llm_model}`")
        st.write(f"**Embedding**：`{cfg.embedding_model}`")
        st.write(f"**切片**：`{cfg.chunk_size} / {cfg.chunk_overlap}`")
        st.write(f"**Chroma 路径**：`{cfg.chroma_dir}`")
        st.write(f"**Collection**：`{cfg.chroma_collection}`")


def _list_doc_options() -> List[str]:
    """从 doc_store 读出所有已入库文档的"原始文件名"，给 multiselect 用。"""
    try:
        docs = get_doc_store().list_documents()
    except Exception:  # noqa: BLE001
        return []
    seen = set()
    out: List[str] = []
    for d in docs:
        name = d.original_name or (d.saved_path and Path(d.saved_path).name) or ""
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return sorted(out)


def render_upload_section() -> list:
    st.subheader("⬆️ 上传新文档")
    files = st.file_uploader(
        f"支持 {', '.join(sorted(s.lstrip('.') for s in SUPPORTED_SUFFIXES))}，可一次选多个",
        type=[s.lstrip(".") for s in SUPPORTED_SUFFIXES],
        accept_multiple_files=True,
        key="uploader",
    )

    if files:
        st.success(f"已选择 {len(files)} 个文件，总大小 "
                   f"{_format_size(sum(len(f.getbuffer()) for f in files))}")
        with st.expander("查看文件列表", expanded=True):
            for i, f in enumerate(files, 1):
                st.write(f"{i}. `{f.name}` — {_format_size(len(f.getbuffer()))}")
    return files or []


def render_persisted_files_section(cfg) -> None:
    """展示 ``data/uploads/`` 下已经入库的文件列表。

    每行附带：
      - 文件大小、入库时间
      - 在向量库中占多少节点
      - 「🗑️ 联动删除」按钮：同时删除原始文件 + Chroma + docstore 中的节点
    """
    st.subheader("📂 已上传文件")
    upload_dir = cfg.data_dir / "uploads"
    files = _list_persisted_files(upload_dir)

    # 用 doc_store 拿"原始名 -> 节点统计"
    doc_store = get_doc_store()
    docs = doc_store.list_documents()
    # 索引方式：既按 file_name 又按 original_name 都建索引，最大化命中
    stat_by_name: dict = {}
    for d in docs:
        if d.original_name:
            stat_by_name[d.original_name] = d
        # 上传时落盘的实际文件名带 timestamp 前缀，也做索引
        if d.saved_path:
            stat_by_name[Path(d.saved_path).name] = d

    total_nodes = doc_store.total_nodes()

    col_path, col_total, col_refresh = st.columns([3, 2, 1])
    with col_path:
        st.caption(f"目录：`{upload_dir}`  ·  磁盘文件 **{len(files)}** 个")
    with col_total:
        st.caption(f"向量库节点：**{total_nodes}** 个")
    with col_refresh:
        if st.button("🔄 刷新", use_container_width=True, key="refresh_files"):
            reset_doc_store()
            st.rerun()

    if not files and not docs:
        st.info("尚未上传任何文件。请在下方选择文件后点击『开始解析』。")
        return

    # 1) 磁盘上的文件 + 节点数
    if files:
        total_size = sum(p.stat().st_size for p in files)
        st.caption(f"磁盘总大小：{_format_size(total_size)}")
        for p in files:
            fstat = p.stat()
            mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(fstat.st_mtime))
            stat = stat_by_name.get(p.name)
            node_cnt = stat.node_count if stat else 0
            file_hash = stat.file_hash if stat else None
            pct = (node_cnt / total_nodes * 100) if total_nodes else 0

            col_name, col_meta, col_nodes, col_del = st.columns([4, 3, 2, 1])
            with col_name:
                badge = "🟢" if node_cnt > 0 else "⚪"
                st.markdown(f"{badge} 📄 `{p.name}`")
            with col_meta:
                st.caption(f"{_format_size(fstat.st_size)}  ·  {mtime}")
            with col_nodes:
                if node_cnt > 0:
                    st.caption(f"📦 {node_cnt} 节点 · {pct:.1f}%")
                else:
                    st.caption("⚠️ 未入库")
            with col_del:
                if st.button(
                    "🗑️",
                    key=f"del_{p.name}",
                    help="同时删除原始文件 + 向量库中对应节点",
                ):
                    _delete_file_and_vectors(p, file_hash)
                    st.rerun()

    # 2) 向量库里有，但磁盘上没找到的"孤儿节点"——给一个清理按钮
    on_disk_keys = {p.name for p in files}
    orphans = [
        d for d in docs
        if (d.original_name not in on_disk_keys)
        and (not d.saved_path or Path(d.saved_path).name not in on_disk_keys)
    ]
    if orphans:
        st.warning(f"发现 **{len(orphans)}** 个孤儿向量（原始文件已删除）")
        for d in orphans:
            col_name, col_nodes, col_del = st.columns([5, 3, 1])
            with col_name:
                st.markdown(f"👻 `{d.original_name}`")
            with col_nodes:
                st.caption(f"📦 {d.node_count} 节点  ·  hash `{d.file_hash[:10]}…`")
            with col_del:
                if st.button("🧹", key=f"orph_{d.file_hash}", help="清理向量库中的残留节点"):
                    deleted = get_doc_store().delete_by_hash(d.file_hash)
                    reset_doc_store()
                    st.toast(f"已清理 {deleted} 个节点", icon="🧹")
                    st.rerun()


def _delete_file_and_vectors(file_path: Path, file_hash: Optional[str]) -> None:
    """联动删除：原始文件 + Chroma 中匹配节点 + docstore 中匹配节点。"""
    file_deleted = False
    try:
        file_path.unlink()
        file_deleted = True
    except Exception as exc:  # noqa: BLE001
        st.error(f"删除原始文件失败：{exc}")

    nodes_deleted = 0
    if file_hash:
        try:
            nodes_deleted = get_doc_store().delete_by_hash(file_hash)
            reset_doc_store()
        except Exception as exc:  # noqa: BLE001
            st.warning(f"清理向量库节点失败：{exc}")

    if file_deleted:
        if nodes_deleted:
            st.toast(
                f"已删除 `{file_path.name}` + {nodes_deleted} 个向量节点",
                icon="✅",
            )
        else:
            st.toast(f"已删除 `{file_path.name}`（向量库无对应节点）", icon="✅")


def render_ingest_section(uploaded_files: list, cfg) -> None:
    st.subheader("⚙️ 解析并入库")

    col_btn, col_tip = st.columns([1, 4])
    with col_btn:
        run = st.button(
            "🚀 开始解析",
            type="primary",
            disabled=not uploaded_files,
            use_container_width=True,
        )
    with col_tip:
        if not uploaded_files:
            st.caption("请先在上方上传至少一个文件")
        else:
            st.caption("点击按钮后将依次执行：保存文件 → 加载 → 切片 → 生成向量 → 写入 Chroma")

    if not run:
        return

    upload_dir = cfg.data_dir / "uploads"

    with st.status("正在处理…", expanded=True) as status:
        # 阶段 0：保存文件 + 内容去重
        st.write("**Step 1/4** · 保存上传文件（按内容 hash 去重）")
        saved, original_names, dup_skipped = _save_uploaded_files(
            uploaded_files, upload_dir
        )
        if dup_skipped:
            st.warning(f"⏭️ 已跳过 {len(dup_skipped)} 个内容重复的文件：")
            for fname, existing in dup_skipped:
                st.write(
                    f"- `{fname}` 已存在为 `{existing.original_name}` "
                    f"（{existing.node_count} 个节点）"
                )
        if not saved:
            if dup_skipped:
                status.update(
                    label="所选文件均已存在，无需重复入库",
                    state="complete",
                )
            else:
                status.update(label="未保存任何有效文件", state="error")
            return
        st.write(f"✅ 已保存 {len(saved)} 个新文件到 `{upload_dir}`")

        # 进度组件
        st.write("**Step 2-4/4** · 加载 → 切片 → 生成向量")
        stage_label = st.empty()
        progress_bar = st.progress(0.0, text="准备中…")
        detail = st.empty()
        timer_start = time.time()

        def on_progress(stage: str, current: int, total: int) -> None:
            pct = current / max(total, 1)
            elapsed = time.time() - timer_start
            stage_label.markdown(f"### 当前阶段：**{stage}**")
            progress_bar.progress(min(pct, 1.0), text=f"{stage} — {current}/{total}")
            detail.caption(
                f"进度 {pct * 100:.1f}% · 已用时 {elapsed:.1f}s"
            )

        try:
            stats = build_index_with_progress(
                file_paths=saved,
                progress_cb=on_progress,
                original_names=original_names,
            )
        except Exception as exc:
            status.update(label=f"❌ 解析失败：{exc}", state="error")
            st.exception(exc)
            return

        status.update(label="✅ 解析完成", state="complete")

    # 完成统计
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("索引文件数", stats["files"])
    c2.metric("生成文本块", stats["nodes"])
    c3.metric("跳过重复", stats.get("skipped", 0))
    c4.metric("耗时", f"{stats['elapsed']:.1f} s")

    st.success("文档已成功写入向量库，可以在侧边栏点击『刷新』查看新的向量总数。")
    st.balloons()


# ---------- 对话页 ----------
def _cite_pattern_replace(text: str, max_index: int, msg_id: str) -> str:
    """把答案中的 ``[1]`` ``[2]`` 替换成可点击的 HTML anchor。

    ``msg_id`` 用来给同一条消息内的引用编号生成唯一 anchor id，避免不同
    回答之间的引用编号互相串扰。
    """
    if not text or max_index <= 0:
        return text

    import re

    def _sub(m):
        idx = int(m.group(1))
        if 1 <= idx <= max_index:
            return (
                f'<a href="#cite-{msg_id}-{idx}" '
                f'style="text-decoration:none;color:#3b82f6;'
                f'background:#eff6ff;border-radius:4px;padding:0 4px;'
                f'font-weight:600;">[{idx}]</a>'
            )
        return m.group(0)

    return re.sub(r"\[(\d+)\]", _sub, text)


def _render_assistant_message(
    content: str,
    citations,
    msg_id: str,
    partial: bool = False,
) -> None:
    """渲染一条 assistant 消息（正文 + 引用），把 [n] 处理成跳转链接。

    若 ``partial=True``，会追加一个"已中断"徽章并提示用户回答可能不完整。
    """
    rendered = _cite_pattern_replace(content or "", len(citations or []), msg_id)
    if partial:
        rendered += (
            '\n\n<span style="display:inline-block;background:#fee2e2;'
            'color:#b91c1c;border-radius:4px;padding:1px 8px;font-size:12px;'
            'font-weight:600;">⏹ 已中断</span>'
        )
    st.markdown(rendered, unsafe_allow_html=True)
    if partial:
        st.warning("此回答在生成过程中被中断，内容可能不完整。可重新提问或追问补全。")
    _render_citations(citations, msg_id)


def _render_citations(citations, msg_id: str = "x") -> None:
    """渲染引用片段。

    兼容两种 citation 来源：
    - 新生成的回答：``Citation`` dataclass 实例
    - 从磁盘加载的历史：``dict``（来自 ``asdict``）

    每条引用前面会加一个 ``<div id="cite-{msg_id}-{n}">`` 锚点，
    供答案中的 ``[n]`` 链接跳转。
    """
    if not citations:
        return
    with st.expander(f"📎 引用 {len(citations)} 个片段", expanded=False):
        for c in citations:
            if isinstance(c, dict):
                idx = c.get("index", "?")
                score_v = c.get("score")
                file_name = c.get("file_name")
                text = c.get("text", "")
            else:
                idx = c.index
                score_v = c.score
                file_name = c.file_name
                text = c.text
            score = f"{score_v:.4f}" if score_v is not None else "-"
            # anchor + 高亮样式（黄色背景的标题块）
            st.markdown(
                f'<div id="cite-{msg_id}-{idx}" '
                f'style="background:#fff7ed;border-left:4px solid #f59e0b;'
                f'padding:6px 10px;border-radius:4px;margin-top:4px;">'
                f'<b>[{idx}]</b> <code>{file_name or "未知来源"}</code> · '
                f'相关度 <code>{score}</code></div>',
                unsafe_allow_html=True,
            )
            st.text(text)
            st.divider()


def _render_trace_panel(trace) -> None:
    """召回片段可视化面板：vector / BM25 / 融合 / rerank 四阶段对比。"""
    if not trace:
        return
    with st.expander("🔬 召回详情（vector / BM25 / 融合 / rerank）", expanded=False):
        if trace.doc_filter:
            st.caption(f"🎯 已限定检索范围：{', '.join(trace.doc_filter)}")
        cols = st.columns(4)
        stages = [
            ("🔎 向量", trace.vector_hits),
            ("🧮 BM25", trace.bm25_hits),
            ("🔀 融合", trace.fused_hits),
            ("🏆 Rerank" + ("" if trace.rerank_used else "（未启用）"),
             trace.rerank_hits),
        ]
        for col, (title, hits) in zip(cols, stages):
            with col:
                st.markdown(f"**{title}**  ·  {len(hits)} 条")
                for i, h in enumerate(hits, start=1):
                    score = h.get("score")
                    score_s = f"{score:.4f}" if isinstance(score, (int, float)) else "-"
                    fname = h.get("file_name") or "未知"
                    st.markdown(
                        f"<div style='font-size:12px;line-height:1.4;"
                        f"margin-bottom:6px;'>"
                        f"<b>#{i}</b> <code>{score_s}</code><br>"
                        f"<span style='color:#6b7280;'>{fname}</span><br>"
                        f"<span style='color:#374151;'>{h.get('preview','')}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )


def _ensure_chat_state() -> None:
    if "chat_session_id" not in st.session_state:
        st.session_state.chat_session_id = None
    if "chat_messages" not in st.session_state:
        # 每条 message: {
        #   "role": "user|assistant", "content": str,
        #   "citations": List[Citation|dict], "trace": Optional[RetrievalTrace],
        #   "followups": List[str]
        # }
        st.session_state.chat_messages = []
    if "chat_session_title" not in st.session_state:
        st.session_state.chat_session_title = ""
    # selectbox 的 widget state 初始值（必须在 selectbox 首次渲染前设置，
    # 否则后面就只能在 callback 里改）
    if "session_selector" not in st.session_state:
        st.session_state.session_selector = None
    if "answer_style" not in st.session_state:
        st.session_state.answer_style = "concise"
    if "doc_filter" not in st.session_state:
        st.session_state.doc_filter = []
    if "session_search" not in st.session_state:
        st.session_state.session_search = ""
    # 由"追问按钮"或"导入会话"等触发的下一轮自动问句
    if "pending_user_input" not in st.session_state:
        st.session_state.pending_user_input = None
    # 控制重命名 popover 显示的临时缓存
    if "rename_buffer" not in st.session_state:
        st.session_state.rename_buffer = ""


def _persist_current_session() -> None:
    """把 ``st.session_state`` 里的当前对话写回磁盘。

    会话标题策略：用户首条消息的前 20 字（如果还没设置过）；保留之前的
    pinned / style 设置不被覆盖。
    """
    sid = st.session_state.chat_session_id
    if not sid:
        return
    msgs = st.session_state.chat_messages
    if not msgs:
        return

    store = get_session_store()
    title = st.session_state.chat_session_title or store.auto_title(
        next((m["content"] for m in msgs if m["role"] == "user"), "")
    )
    st.session_state.chat_session_title = title

    existing = store.load(sid)
    created_at = existing.created_at if existing else time.time()
    pinned = existing.pinned if existing else False

    session = StoredSession(
        session_id=sid,
        title=title,
        created_at=created_at,
        updated_at=time.time(),
        messages=dict_to_messages(msgs),
        pinned=pinned,
        style=st.session_state.answer_style,
    )
    try:
        store.save(session)
    except Exception as exc:  # noqa: BLE001
        st.toast(f"会话保存失败：{exc}", icon="⚠️")


def _switch_to_session(sid: str) -> bool:
    """切换到一个磁盘上已存在的会话：加载历史 + 恢复 engine memory。

    注意：这里**不**修改 ``session_selector`` 这个 widget state——调用方
    （selectbox 的 ``on_change`` 回调）已经先一步把它设成新值了，重复
    赋值会触发 StreamlitAPIException。
    """
    store = get_session_store()
    loaded = store.load(sid)
    if not loaded:
        st.error(f"会话 `{sid}` 已不存在或已损坏")
        return False

    st.session_state.chat_session_id = loaded.session_id
    st.session_state.chat_session_title = loaded.title
    st.session_state.chat_messages = messages_to_dict(loaded.messages)
    st.session_state.answer_style = loaded.style or "concise"

    try:
        get_chat_service().restore_session(
            loaded.session_id,
            [(m.role, m.content) for m in loaded.messages],
            style=loaded.style or "concise",
        )
    except Exception as exc:  # noqa: BLE001
        st.toast(f"恢复后端记忆失败：{exc}", icon="⚠️")
    return True


def _start_new_session() -> None:
    """开新会话：清空当前 UI 状态，并把旧 engine 释放。

    注意：``session_selector`` widget state 的同步留给调用方处理
    （只有 callback 能安全修改 widget state）。
    """
    old_sid = st.session_state.chat_session_id
    if old_sid:
        try:
            get_chat_service().reset(old_sid)
        except Exception:
            pass
    st.session_state.chat_session_id = None
    st.session_state.chat_session_title = ""
    st.session_state.chat_messages = []


# ---------- selectbox / button callbacks ----------
def _on_session_selector_change() -> None:
    """selectbox 选项变化时的回调：值已经被写到 widget state 里了，
    这里只负责把对应的对话状态切换/重建。"""
    selected = st.session_state.session_selector
    if selected == st.session_state.chat_session_id:
        return
    if selected is None:
        _start_new_session()
    else:
        if not _switch_to_session(selected):
            # 加载失败 → 把 selectbox 拉回 None（callback 内部允许改 widget state）
            st.session_state.session_selector = None
            _start_new_session()


def _on_new_session_clicked() -> None:
    """『➕ 新建』按钮回调。"""
    _start_new_session()
    st.session_state.session_selector = None


def _on_delete_session_clicked() -> None:
    """『🗑️』按钮回调：删除当前会话的持久化文件，并重置为新会话。"""
    sid = st.session_state.chat_session_id
    if not sid:
        return
    try:
        get_session_store().delete(sid)
    except Exception as exc:  # noqa: BLE001
        st.toast(f"删除失败：{exc}", icon="⚠️")
        return
    _start_new_session()
    st.session_state.session_selector = None


def _on_followup_clicked(text: str) -> None:
    """追问按钮点击：把追问文本塞到 pending_user_input，下次 rerun 自动当作输入触发。"""
    st.session_state.pending_user_input = text


def _on_pin_toggle_clicked() -> None:
    sid = st.session_state.chat_session_id
    if not sid:
        return
    store = get_session_store()
    s = store.load(sid)
    if not s:
        return
    store.set_pinned(sid, not s.pinned)


def _on_rename_submit() -> None:
    sid = st.session_state.chat_session_id
    new_title = (st.session_state.get("rename_input") or "").strip()
    if not sid or not new_title:
        return
    if get_session_store().rename(sid, new_title):
        st.session_state.chat_session_title = new_title
        st.toast(f"已重命名为「{new_title}」", icon="✏️")


def _on_style_change() -> None:
    new_style = st.session_state.answer_style
    sid = st.session_state.chat_session_id
    if sid:
        try:
            get_chat_service().set_style(sid, new_style)
        except Exception as exc:  # noqa: BLE001
            st.toast(f"切换风格失败：{exc}", icon="⚠️")


def _import_session_from_upload(uploaded_file) -> None:
    try:
        payload = json.loads(uploaded_file.getvalue().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        st.error(f"JSON 解析失败：{exc}")
        return
    try:
        new_sid = get_session_store().import_session(payload, overwrite_id=True)
    except Exception as exc:  # noqa: BLE001
        st.error(f"导入失败：{exc}")
        return
    st.toast(f"已导入会话，ID `{new_sid[:8]}…`", icon="📥")
    # 标记下一次渲染时自动切换到新会话；因为这里仍在 button 回调外，
    # 直接改 chat_session_id 即可，让 render_chat_tab 顶部的同步逻辑接手。
    _switch_to_session(new_sid)


def _format_session_label(sid: Optional[str], sessions) -> str:
    """selectbox 用：把 session_id -> "📌 标题（n 条 · 时间）"。"""
    if sid is None:
        return "➕ 新建会话"
    meta = next((s for s in sessions if s.session_id == sid), None)
    if meta is None:
        return f"💬 {sid[:8]}…"
    ts = time.strftime("%m-%d %H:%M", time.localtime(meta.updated_at))
    pin = "📌 " if meta.pinned else ""
    return f"{pin}💬 {meta.title}  ·  {meta.message_count} 条  ·  {ts}"


def _render_session_toolbar(store, sessions, current_sid: Optional[str]) -> None:
    """会话顶部工具栏：选择 / 新建 / 删除 / 重命名 / 固定 / 导出 / 导入。"""
    options = [None] + [s.session_id for s in sessions]

    # 在 selectbox 实例化【之前】校准 widget state，避免 StreamlitAPIException
    if (st.session_state.session_selector is not None
            and st.session_state.session_selector not in options):
        st.session_state.session_selector = None
    if st.session_state.session_selector != current_sid:
        if current_sid in options:
            st.session_state.session_selector = current_sid
        else:
            st.session_state.session_selector = None

    # 第一行：选择 + 新建 + 删除
    col_sel, col_new, col_del = st.columns([6, 1, 1])
    with col_sel:
        st.selectbox(
            "会话选择",
            options,
            format_func=lambda s: _format_session_label(s, sessions),
            key="session_selector",
            on_change=_on_session_selector_change,
            label_visibility="collapsed",
        )
    with col_new:
        st.button(
            "➕ 新建",
            use_container_width=True,
            help="开始一个全新的会话",
            on_click=_on_new_session_clicked,
            key="new_session_btn",
        )
    with col_del:
        st.button(
            "🗑️",
            use_container_width=True,
            disabled=not current_sid,
            help="删除当前会话的持久化记录",
            on_click=_on_delete_session_clicked,
            key="del_session_btn",
        )

    # 第二行：搜索 + 重命名 + 固定 + 导出 + 导入
    col_search, col_rename, col_pin, col_export_md, col_export_json, col_import = (
        st.columns([4, 1, 1, 1, 1, 2])
    )
    with col_search:
        st.text_input(
            "搜索会话",
            key="session_search",
            placeholder="🔍 按标题或内容搜索…",
            label_visibility="collapsed",
        )

    cur_loaded = store.load(current_sid) if current_sid else None
    is_pinned = bool(cur_loaded and cur_loaded.pinned)

    with col_rename:
        with st.popover(
            "✏️",
            use_container_width=True,
            disabled=not current_sid,
            help="重命名当前会话",
        ):
            st.markdown("**重命名当前会话**")
            st.text_input(
                "新标题",
                value=st.session_state.chat_session_title or "",
                key="rename_input",
                max_chars=50,
                label_visibility="collapsed",
            )
            st.button(
                "保存",
                key="rename_submit",
                use_container_width=True,
                on_click=_on_rename_submit,
            )

    with col_pin:
        st.button(
            "📌" if is_pinned else "📍",
            use_container_width=True,
            disabled=not current_sid,
            help="取消置顶" if is_pinned else "置顶此会话",
            on_click=_on_pin_toggle_clicked,
            key="pin_btn",
        )

    with col_export_md:
        if cur_loaded:
            st.download_button(
                "⬇️MD",
                data=session_to_markdown(cur_loaded).encode("utf-8"),
                file_name=f"{cur_loaded.title or 'session'}.md",
                mime="text/markdown",
                use_container_width=True,
                help="导出为 Markdown",
                key="export_md_btn",
            )
        else:
            st.button("⬇️MD", disabled=True, use_container_width=True, key="export_md_disabled")

    with col_export_json:
        if cur_loaded:
            st.download_button(
                "⬇️JSON",
                data=session_to_json_bytes(cur_loaded),
                file_name=f"{cur_loaded.session_id}.json",
                mime="application/json",
                use_container_width=True,
                help="导出为 JSON（可再导入）",
                key="export_json_btn",
            )
        else:
            st.button("⬇️JSON", disabled=True, use_container_width=True, key="export_json_disabled")

    with col_import:
        with st.popover("📤 导入", use_container_width=True, help="导入 JSON 会话"):
            up = st.file_uploader(
                "选择 JSON 文件",
                type=["json"],
                key="import_session_uploader",
                label_visibility="collapsed",
            )
            if up is not None and st.button("确认导入", key="confirm_import_btn"):
                _import_session_from_upload(up)
                st.rerun()


def render_chat_tab() -> None:
    _ensure_chat_state()
    store = get_session_store()
    # 列表 + 搜索过滤
    sessions = store.list(query=st.session_state.session_search)
    current_sid = st.session_state.chat_session_id

    # ----- 会话管理栏 -----
    _render_session_toolbar(store, sessions, current_sid)

    # 当前会话信息行
    cur_title = st.session_state.chat_session_title or "(尚未保存)"
    cur_id_short = (current_sid[:8] + "…") if current_sid else "尚未生成"
    style_label = STYLE_LABELS.get(st.session_state.answer_style, st.session_state.answer_style)
    st.caption(
        f"当前：**{cur_title}**  ·  ID `{cur_id_short}`  ·  "
        f"风格 {style_label}  ·  共 **{len(sessions)}** 个历史会话"
        + (f"  ·  搜索：`{st.session_state.session_search}`"
           if st.session_state.session_search else "")
    )

    st.divider()

    # 提示：是否有可用知识库
    if _get_collection_count() == 0:
        st.info("当前向量库为空，请先在左侧「📤 文档管理」页面上传文档并解析。")
        return

    # 历史消息渲染（assistant 部分用富渲染：引用跳转 + trace 面板 + 追问）
    for i, msg in enumerate(st.session_state.chat_messages):
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                _render_assistant_message(
                    msg["content"],
                    msg.get("citations", []),
                    msg_id=str(i),
                    partial=bool(msg.get("partial", False)),
                )
                _render_trace_panel(msg.get("trace"))
                _render_followups(msg.get("followups") or [], msg_idx=i)
            else:
                st.markdown(msg["content"])

    # 输入框 / 追问按钮触发的输入
    pending = st.session_state.pending_user_input
    if pending:
        user_input = pending
        st.session_state.pending_user_input = None
    else:
        user_input = st.chat_input("请输入你的问题…")

    if not user_input:
        return

    # 立刻显示用户消息
    st.session_state.chat_messages.append(
        {"role": "user", "content": user_input, "citations": []}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    # ----- 调用后端（流式 + 检索范围 + 风格）-----
    # 关键：先 append 一个 placeholder assistant message，再开始 stream；
    # 流式期间把每个 token 实时写入 placeholder["content"]。这样即便用户
    # 在 streamlit 右上角点了 Stop（会抛 RerunException/StopException），
    # 已经生成的 partial 内容也会留在 chat_messages 里 + 持久化到磁盘。
    placeholder: Optional[dict] = None
    collected: List[str] = []
    completed = False

    with st.chat_message("assistant"):
        # 顶部一行：左侧状态文字（思考中/回答中），右侧"⏹ 停止"按钮。
        # 关键机制：流式期间用户点击此按钮 → Streamlit 接收到 widget 交互
        # 事件 → 在 stream 的下一次 streamlit API 调用处抛出 RerunException
        # → 我们的 finally 块把已收到的 partial 内容落盘 → 新一轮 run 时
        # 历史渲染那条 partial 消息会带"⏹ 已中断"徽章。
        # 这里 button 的返回值我们不直接读——按钮的"被点击"这个动作本身
        # 就是中断信号；点击会被 streamlit 转化为 rerun，自然中断当前流。
        head_cols = st.columns([0.78, 0.22])
        with head_cols[0]:
            status_box = st.empty()
        with head_cols[1]:
            stop_slot = st.empty()  # stream 之前 fill；正常完成后 .empty()

        try:
            # ---- 1. 先做检索（拿到 handle），失败要撤回 user msg ----
            try:
                service = get_chat_service()
                status_box.markdown("🔎 **正在检索知识库…**")
                handle = service.stream_chat(
                    user_input,
                    session_id=st.session_state.chat_session_id,
                    doc_filter=st.session_state.doc_filter or None,
                    style=st.session_state.answer_style,
                )
            except FileNotFoundError as exc:
                status_box.empty()
                st.error(str(exc))
                st.session_state.chat_messages.pop()  # 撤回 user
                return
            except Exception as exc:
                status_box.empty()
                st.error(f"调用失败：{exc}")
                st.exception(exc)
                st.session_state.chat_messages.pop()
                return

            # ---- 2. 追加 assistant placeholder + 渲染停止按钮 ----
            placeholder = {
                "role": "assistant",
                "content": "",
                "citations": [],
                "trace": None,
                "followups": [],
                "partial": True,
            }
            st.session_state.chat_messages.append(placeholder)
            status_box.markdown("🤔 **AI 正在思考…**")
            # 停止按钮：仅在 stream 期间存在；点击会触发 rerun → 中断当前流
            stop_slot.button(
                "⏹ 停止",
                key="stop_streaming_btn",
                type="primary",
                use_container_width=True,
                help="停止当前回答（已生成的内容会保留）",
            )

            # ---- 3. 流式生成（每个 token 同步写回 placeholder）----
            def _status_iter():
                first = True
                for token in handle.token_iter:
                    if first and token:
                        status_box.markdown("✍️ **AI 正在回答…**")
                        first = False
                    if token:
                        collected.append(token)
                        placeholder["content"] = "".join(collected)
                    yield token

            full_text = st.write_stream(_status_iter()) or ""
            status_box.empty()
            stop_slot.empty()  # 正常完成 → 立刻收掉停止按钮

            # ---- 4. 收尾：finalize → 替换 placeholder ----
            result = handle.finalize()
            st.session_state.chat_session_id = result.session_id

            if not full_text.strip():
                st.info(result.answer)

            placeholder.update({
                "content": result.answer,
                "citations": result.citations,
                "trace": result.trace,
                "partial": False,
            })
            completed = True

            # ---- 5. 追问建议（独立 try，失败不影响主流程）----
            try:
                with st.spinner("生成追问建议…"):
                    placeholder["followups"] = service.suggest_followups(
                        user_input, result.answer, n=3
                    )
            except Exception:  # noqa: BLE001
                placeholder["followups"] = []

        finally:
            # 不管完成还是被中断（streamlit stop / 异常），都尝试把当前
            # 状态写到磁盘。这一段必须能容忍 placeholder is None 的情况
            # （检索阶段就失败了，return 前 placeholder 还没创建）。
            if placeholder is not None and not completed:
                partial = "".join(collected).strip()
                if partial:
                    placeholder["content"] = partial
                    placeholder["partial"] = True
                else:
                    # 一个 token 都没生成 → 把 placeholder + user 一并撤回，
                    # 避免下次刷新看到一对孤立的"提问 + 空回答"
                    try:
                        if (st.session_state.chat_messages
                                and st.session_state.chat_messages[-1] is placeholder):
                            st.session_state.chat_messages.pop()
                        if (st.session_state.chat_messages
                                and st.session_state.chat_messages[-1].get("role") == "user"):
                            st.session_state.chat_messages.pop()
                    except Exception:  # noqa: BLE001
                        pass
            try:
                _persist_current_session()
            except Exception:  # noqa: BLE001
                pass

    if not completed:
        # 走到这里说明被中断/出错（partial 已经在 finally 落盘），
        # 不需要再 st.rerun()，让 streamlit 完成本轮即可
        return

    # 答完且未中断 → rerun 让新消息走"历史消息渲染"路径，
    # 这样引用跳转锚点 / 追问按钮 / trace 面板才会真正显示出来
    st.rerun()


def _render_followups(followups: List[str], msg_idx: int) -> None:
    """追问按钮组：3 个候选问句，点击直接发出。"""
    if not followups:
        return
    st.caption("💡 你可能还想问：")
    cols = st.columns(len(followups))
    for i, (col, q) in enumerate(zip(cols, followups)):
        with col:
            st.button(
                q,
                key=f"followup_{msg_idx}_{i}",
                on_click=_on_followup_clicked,
                args=(q,),
                use_container_width=True,
            )


def render_upload_page(cfg) -> None:
    st.title("📤 文档管理")
    st.caption("上传 → 解析 → 入库；同时查看磁盘上已存在的文件")

    render_persisted_files_section(cfg)
    st.divider()
    files = render_upload_section()
    st.divider()
    render_ingest_section(files, cfg)


def render_chat_page() -> None:
    st.title("💬 知识问答")
    st.caption("基于已构建的知识库进行多轮对话，自动给出引用片段")
    render_chat_tab()


PAGES = {
    "📤 文档管理": "upload",
    "💬 知识问答": "chat",
}


def main() -> None:
    try:
        cfg = get_settings()
    except Exception as exc:
        st.error(
            "配置加载失败：请确认项目根目录存在 `.env` 文件，并已填入 "
            "`DASHSCOPE_API_KEY`。"
        )
        st.exception(exc)
        st.stop()

    # 侧边栏：页面路由 + 知识库状态
    with st.sidebar:
        st.title("📚 LlamaIndex × Qwen")
        st.caption("知识库管理")
        page_label = st.radio(
            "页面导航",
            list(PAGES.keys()),
            label_visibility="collapsed",
        )
        st.divider()

    page_key = PAGES[page_label]
    render_sidebar(cfg, page_key)

    if page_key == "upload":
        render_upload_page(cfg)
    elif page_key == "chat":
        render_chat_page()


if __name__ == "__main__":
    main()
