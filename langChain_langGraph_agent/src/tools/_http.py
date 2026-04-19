"""所有工具共用的 HTTP 封装：超时、重试、UA、错误格式统一。"""

from __future__ import annotations

import time
from typing import Any

import requests

DEFAULT_TIMEOUT = 10
DEFAULT_RETRIES = 2  # 重试次数（不含首次）
DEFAULT_BACKOFF = 0.6  # 指数退避起始秒数

DEFAULT_HEADERS = {
    # 用一个真实浏览器风格的 UA，比自定义 "TravelAgent/0.1" 通过率高得多
    # （Nominatim、Wikipedia 都会拦截 Python-requests 默认 UA 或来历不明的 UA）
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.6",
}


def http_get(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
) -> dict[str, Any]:
    """统一 GET：成功返回 {"ok": True, "data": ...}；失败返回 {"ok": False, "error": "..."}。

    - 自带指数退避重试（默认 3 次：首发 + 2 次重试），针对短暂 SSL / 网络抖动
    - 工具层永远不要把异常抛回给 Agent，否则 LangGraph 整个 step 会崩
    """
    merged = {**DEFAULT_HEADERS, **(headers or {})}
    last_error = "unknown"

    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, params=params, headers=merged, timeout=timeout)
            resp.raise_for_status()
            ctype = resp.headers.get("Content-Type", "")
            if "application/json" in ctype:
                return {"ok": True, "data": resp.json()}
            return {"ok": True, "data": resp.text}
        except requests.exceptions.Timeout:
            last_error = f"请求超时（>{timeout}s）：{url}"
        except requests.exceptions.HTTPError as e:
            # 4xx 类一般重试也没用，直接退出
            code = e.response.status_code
            last_error = f"HTTP 错误 {code}：{url}"
            if 400 <= code < 500 and code != 429:
                break
        except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
            last_error = f"网络错误：{type(e).__name__}: {e}"
        except Exception as e:  # noqa: BLE001
            last_error = f"请求失败：{type(e).__name__}: {e}"

        if attempt < retries:
            time.sleep(DEFAULT_BACKOFF * (2 ** attempt))

    return {"ok": False, "error": last_error}


__all__ = ["http_get"]
