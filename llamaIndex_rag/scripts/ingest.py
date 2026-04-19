"""命令行入口：构建/重建知识库索引。

用法：
    python -m scripts.ingest
"""

from src.ingest import main

if __name__ == "__main__":
    main()
