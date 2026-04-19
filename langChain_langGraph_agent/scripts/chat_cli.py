"""命令行版本的对话界面，方便没有浏览器的环境调试。

用法：
    python -m scripts.chat_cli                  # 新会话
    python -m scripts.chat_cli --thread my-tid  # 指定 thread 续聊
"""

from __future__ import annotations

import argparse
import io
import sys
import uuid
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402

from src.agent import stream_agent  # noqa: E402
from src.config import settings  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="智能旅游规划 Agent 命令行")
    ap.add_argument("--thread", default=None, help="会话 ID，省略则随机新建")
    ap.add_argument("--no-trace", action="store_true", help="不显示工具调用 trace")
    args = ap.parse_args()

    thread_id = args.thread or uuid.uuid4().hex[:12]
    show_trace = not args.no_trace

    print("=" * 60)
    print(f"  🧳 智能旅游规划 Agent  ·  Thread: {thread_id}")
    print(f"  Model: {settings.LLM_MODEL}  ·  输入 q / quit / exit 退出")
    print("=" * 60)

    while True:
        try:
            user = input("\n你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见，旅途愉快！")
            return
        if not user:
            continue
        if user.lower() in {"q", "quit", "exit", ":q"}:
            print("再见，旅途愉快！")
            return

        print("\n助手 > ", end="", flush=True)
        final_text = ""
        try:
            for event in stream_agent(user, thread_id=thread_id):
                for node, state in event.items():
                    msgs = state.get("messages", []) if isinstance(state, dict) else []
                    for m in msgs:
                        if isinstance(m, AIMessage):
                            if show_trace and getattr(m, "tool_calls", None):
                                for tc in m.tool_calls:
                                    print(f"\n  🔧 调用 {tc['name']}({tc.get('args')})", flush=True)
                            if m.content:
                                final_text = m.content
                        elif isinstance(m, ToolMessage) and show_trace:
                            preview = str(m.content)[:120].replace("\n", " ")
                            print(f"  📦 {m.name} → {preview}...", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"\n[调用出错] {type(e).__name__}: {e}")
            continue

        print()
        print(final_text or "（本轮 Agent 未生成文本回答）")


if __name__ == "__main__":
    main()
