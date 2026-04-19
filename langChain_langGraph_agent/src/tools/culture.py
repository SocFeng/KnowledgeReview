"""文化背景工具：查询某地的历史 / 文化 / 风俗简介。

数据源：Wikipedia REST API（zh.wikipedia 优先，失败回落 en.wikipedia）。
返回的是简介摘要，由 Agent 进一步整合到行程介绍中。
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from ._http import http_get


def _wiki_summary(title: str, lang: str = "zh") -> dict[str, Any] | None:
    """Wikipedia REST summary endpoint。"""
    res = http_get(
        f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title}",
    )
    if not res["ok"]:
        return None
    data = res["data"]
    if not isinstance(data, dict) or "extract" not in data:
        return None
    return {
        "title": data.get("title", title),
        "summary": data.get("extract", ""),
        "url": data.get("content_urls", {}).get("desktop", {}).get("page"),
        "lang": lang,
    }


@tool("lookup_culture", return_direct=False)
def lookup_culture(place_or_topic: str) -> dict[str, Any]:
    """查询某地 / 某主题的文化、历史、风俗简介（数据源：Wikipedia）。

    Args:
        place_or_topic: 地名或主题，例如 "西安"、"敦煌莫高窟"、"川菜"、"傣族泼水节"。

    Returns:
        含 summary（简介）和 url（原文链接）的字典；失败返回 {"error": "..."}。
    """
    # 中文 wiki 优先
    info = _wiki_summary(place_or_topic, "zh")
    if info and info["summary"]:
        return info

    # 回落英文 wiki
    info_en = _wiki_summary(place_or_topic, "en")
    if info_en and info_en["summary"]:
        return info_en

    return {
        "error": f"未在 Wikipedia 找到 '{place_or_topic}' 的条目，请用模型常识补充。",
        "topic": place_or_topic,
    }
