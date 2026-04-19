"""不启动服务，直接命令行问一题（用于本地快速联调）。

用法：
    python -m scripts.ask_once "什么是 RAG？"
"""

import sys

from src.chat import get_chat_service


def main() -> None:
    if len(sys.argv) < 2:
        print('用法: python -m scripts.ask_once "你的问题"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    service = get_chat_service()
    result = service.chat(question)

    print("\n=== Answer ===")
    print(result.answer)
    print("\n=== Citations ===")
    for c in result.citations:
        score = f"{c.score:.4f}" if c.score is not None else "-"
        print(f"[{c.index}] ({score}) {c.file_name}")
        print(f"    {c.text}\n")


if __name__ == "__main__":
    main()
