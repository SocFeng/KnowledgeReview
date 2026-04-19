"""命令行入口：扫描 ``data/`` 目录构建索引。

用法：
    python -m scripts.ingest
"""

import logging
import sys
from pathlib import Path

# 允许 `python scripts/ingest.py` 直接运行
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.ingest import build_index_with_progress  # noqa: E402


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    build_index_with_progress()
