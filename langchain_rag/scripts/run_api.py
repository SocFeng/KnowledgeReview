"""命令行入口：启动 FastAPI 服务。

用法：
    python -m scripts.run_api
"""

import logging
import sys
from pathlib import Path

# 允许 `python scripts/run_api.py` 直接运行
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.api import run  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run()
