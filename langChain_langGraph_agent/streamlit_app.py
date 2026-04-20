"""智能旅游规划 Agent —— Streamlit Web UI。

启动方式：
    streamlit run streamlit_app.py

特性：
- 类聊天对话：左侧会话列表（多 thread）、右侧消息流
- 工具调用 trace 实时展开（可折叠）
- 支持上下文修改：只要在同一 thread 内继续聊，Agent 就会自动衔接
- 一键新建 / 删除 / 重命名会话
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agent import get_history, reset_thread, stream_agent_tokens
from src.config import settings
from src.tools import ALL_TOOLS
from src.usage import load_usage, summarize_total

# ---------- 页面基础设置 ----------
st.set_page_config(
    page_title="智能旅游规划 · LangGraph Agent",
    page_icon="🧳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- 会话存储（用本地 JSON 维护 thread 元信息）----------
SESSIONS_FILE = settings.data_dir / "sessions.json"


def _load_sessions() -> dict[str, dict]:
    if not SESSIONS_FILE.exists():
        return {}
    try:
        return json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_sessions(sessions: dict[str, dict]) -> None:
    SESSIONS_FILE.write_text(
        json.dumps(sessions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _new_session(title: str | None = None) -> str:
    sessions = _load_sessions()
    tid = uuid.uuid4().hex
    sessions[tid] = {
        "title": title or f"行程 {datetime.now().strftime('%m-%d %H:%M')}",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    _save_sessions(sessions)
    return tid


def _rename_session(tid: str, new_title: str) -> None:
    sessions = _load_sessions()
    if tid in sessions:
        sessions[tid]["title"] = new_title
        _save_sessions(sessions)


def _delete_session(tid: str) -> None:
    sessions = _load_sessions()
    sessions.pop(tid, None)
    _save_sessions(sessions)
    reset_thread(tid)


# ---------- 初始化 session_state ----------
if "active_thread_id" not in st.session_state:
    sessions = _load_sessions()
    if sessions:
        st.session_state.active_thread_id = next(iter(sessions.keys()))
    else:
        st.session_state.active_thread_id = _new_session("我的第一个行程")

if "show_trace" not in st.session_state:
    st.session_state.show_trace = settings.SHOW_TOOL_TRACE


# =============================================================
# 侧边栏
# =============================================================
with st.sidebar:
    st.markdown("## 🧳 智能旅游规划")
    st.caption(f"模型：`{settings.LLM_MODEL}`  ·  Agent：LangGraph")

    st.divider()
    st.markdown("### 📋 会话列表")

    if st.button("➕ 新建会话", use_container_width=True):
        new_tid = _new_session()
        st.session_state.active_thread_id = new_tid
        st.rerun()

    sessions = _load_sessions()
    for tid, meta in list(sessions.items())[::-1]:
        is_active = tid == st.session_state.active_thread_id
        cols = st.columns([5, 1])
        with cols[0]:
            label = ("🟢 " if is_active else "💬 ") + meta["title"]
            if st.button(label, key=f"sw_{tid}", use_container_width=True):
                st.session_state.active_thread_id = tid
                st.rerun()
        with cols[1]:
            if st.button("🗑", key=f"del_{tid}", help="删除该会话"):
                _delete_session(tid)
                if tid == st.session_state.active_thread_id:
                    remaining = list(_load_sessions().keys())
                    st.session_state.active_thread_id = (
                        remaining[0] if remaining else _new_session()
                    )
                st.rerun()

    st.divider()
    st.markdown("### ⚙️ 设置")
    st.session_state.show_trace = st.toggle(
        "显示工具调用过程",
        value=st.session_state.show_trace,
        help="开启后会展开 Agent 每一步调用了哪些工具、得到了什么结果",
    )

    if st.button("🔄 清空当前会话历史", use_container_width=True):
        reset_thread(st.session_state.active_thread_id)
        st.success("已清空")
        time.sleep(0.4)
        st.rerun()

    st.divider()
    st.markdown("### 🛠 可用工具")
    for t in ALL_TOOLS:
        st.markdown(f"- **{t.name}** —— {t.description.splitlines()[0][:40]}")

    st.divider()
    st.markdown("### 📊 Token & 费用")
    _total = summarize_total(st.session_state.active_thread_id)
    _last_turn = st.session_state.get("last_turn_usage")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("↑ 输入", f"{_total.get('input', 0):,}")
    col_b.metric("↓ 输出", f"{_total.get('output', 0):,}")
    col_c.metric("累计费用", f"¥{_total.get('cost_cny', 0.0):.4f}")
    if _last_turn:
        st.caption(
            f"本轮：↑{_last_turn['input']:,} ↓{_last_turn['output']:,} tokens"
            f"  ·  ¥{_last_turn['cost_cny']:.4f}"
        )
    with st.expander("📈 逐轮明细", expanded=False):
        _usage_detail = load_usage(st.session_state.active_thread_id)
        _turns = _usage_detail.get("turns", [])
        if _turns:
            st.caption(f"模型：`{_usage_detail.get('model', settings.LLM_MODEL)}`")
            st.dataframe(
                _turns[-20:][::-1],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("暂无用量记录。发一条消息试试～")

    st.divider()
    with st.expander("🔌 数据源状态", expanded=False):
        st.write(f"DashScope LLM：{'✅' if settings.DASHSCOPE_API_KEY else '❌ 未配置'}")
        st.write(f"高德地图：{'✅ 已启用' if settings.has_amap else '⚪ 未配置（已 fallback）'}")
        st.write(f"Open-Meteo 天气：✅ 免费无 key")
        st.write(f"OSM Nominatim：✅ 免费无 key")
        st.write(f"Wikipedia：✅ 免费无 key")


# =============================================================
# 主区域
# =============================================================
active_tid = st.session_state.active_thread_id
active_meta = _load_sessions().get(active_tid, {})

# 顶部：标题 + 重命名
title_col, edit_col = st.columns([6, 1])
with title_col:
    st.markdown(f"## 💬 {active_meta.get('title', '新会话')}")
with edit_col:
    with st.popover("✏️ 重命名"):
        new_name = st.text_input("新名称", value=active_meta.get("title", ""))
        if st.button("保存"):
            if new_name.strip():
                _rename_session(active_tid, new_name.strip())
                st.rerun()

st.caption(
    "告诉我你的出发地、目的地、出行天数和偏好，我会调天气 / 景点 / 路线工具帮你规划。"
    "随时改需求都可以，我会基于上下文增量调整。"
)


# =============================================================
# 渲染历史对话
# =============================================================
def _render_message(msg, key_prefix: str = ""):
    if isinstance(msg, HumanMessage):
        with st.chat_message("user", avatar="🧑"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant", avatar="🧳"):
            # 如果有 tool_calls，先渲染它们的标签
            if getattr(msg, "tool_calls", None) and st.session_state.show_trace:
                with st.expander(
                    f"🔧 调用了 {len(msg.tool_calls)} 个工具",
                    expanded=False,
                ):
                    for tc in msg.tool_calls:
                        st.markdown(f"**{tc['name']}**")
                        st.json(tc.get("args", {}), expanded=False)
            if msg.content:
                st.markdown(msg.content)
    elif isinstance(msg, ToolMessage):
        if st.session_state.show_trace:
            with st.chat_message("tool", avatar="🛠"):
                st.markdown(f"**{msg.name}** 返回：")
                # 尝试当作 JSON 渲染，不行就 code
                try:
                    st.json(json.loads(msg.content) if isinstance(msg.content, str) else msg.content,
                            expanded=False)
                except Exception:
                    st.code(str(msg.content)[:2000])


history = get_history(active_tid)
for i, m in enumerate(history):
    _render_message(m, key_prefix=f"hist_{i}")


# =============================================================
# 输入区 + 流式响应
# =============================================================
prompt = st.chat_input("例如：我想周末从北京去天津玩 2 天，喜欢历史和小吃")
if prompt:
    # 1) 立即显示用户消息
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    # 2) Agent 流式响应（token 级打字机 + 工具 trace + 用量）
    with st.chat_message("assistant", avatar="🧳"):
        trace_box = st.expander(
            "🔧 Agent 思考过程（实时）", expanded=st.session_state.show_trace
        )
        answer_box = st.empty()

        final_text = ""
        turn_usage: dict | None = None
        try:
            for ev in stream_agent_tokens(prompt, thread_id=active_tid):
                ev_type = ev.get("type")

                if ev_type == "token":
                    # 真·token 级增量，累加并渲染带光标的文本
                    final_text += ev["content"]
                    answer_box.markdown(final_text + "▌")

                elif ev_type == "tool_call":
                    with trace_box:
                        st.markdown(f"🤔 决定调用 **{ev['name']}**")
                        st.json(ev.get("args", {}), expanded=False)

                elif ev_type == "tool_result":
                    with trace_box:
                        st.markdown(f"📦 **{ev['name']}** 返回：")
                        content = ev.get("content", "")
                        try:
                            parsed = (
                                json.loads(content)
                                if isinstance(content, str)
                                else content
                            )
                            st.json(parsed, expanded=False)
                        except Exception:
                            st.code(str(content)[:1500])

                elif ev_type == "usage":
                    turn_usage = {
                        "input": ev["input"],
                        "output": ev["output"],
                        "cost_cny": ev["cost_cny"],
                    }

                elif ev_type == "done":
                    # 把尾部光标去掉
                    if final_text:
                        answer_box.markdown(final_text)
                    elif ev.get("text"):
                        final_text = ev["text"]
                        answer_box.markdown(final_text)
        except Exception as e:
            st.error(f"调用 Agent 出错：{type(e).__name__}: {e}")
            raise

        if not final_text:
            answer_box.info("（本轮 Agent 未给出文本回答，请尝试换种问法）")

        # 显示本轮 token/费用
        if turn_usage:
            st.caption(
                f"📊 本轮：↑{turn_usage['input']:,} ↓{turn_usage['output']:,} tokens"
                f"  ·  ¥{turn_usage['cost_cny']:.4f}"
            )
            st.session_state["last_turn_usage"] = turn_usage

    # 3) 一轮完成后强制 rerun，让"已完成"的对话以静态形式重排（去掉实时 placeholder）
    st.rerun()
