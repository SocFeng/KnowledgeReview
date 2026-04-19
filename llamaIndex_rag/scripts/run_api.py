"""命令行入口：启动 FastAPI 服务。

用法：
    python -m scripts.run_api
"""

import logging

from src.api import run

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run()
