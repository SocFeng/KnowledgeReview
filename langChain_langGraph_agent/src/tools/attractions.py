"""景点查询工具：返回某地的热门景点列表。

数据源策略：
- 优先：高德 POI 搜索（中文准、含地址 / 评分 / 类型）
- 兜底：Wikipedia API + 经纬度，附近 wiki 条目（geosearch）
- 最终兜底：返回空列表 + 一段提示，由 Agent 用自身知识补充
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from ..config import settings
from ._http import http_get
from .geocode import geocode_address as _geocode_tool

# 高德 POI 类型代码：旅游景点
# https://lbs.amap.com/api/webservice/download
_AMAP_POI_TYPE_SCENIC = "110000"  # 风景名胜


def _attractions_amap(city: str, keyword: str | None, max_results: int) -> list[dict] | None:
    if not settings.has_amap:
        return None
    res = http_get(
        "https://restapi.amap.com/v3/place/text",
        params={
            "key": settings.AMAP_API_KEY,
            "keywords": keyword or "景点",
            "city": city,
            "types": _AMAP_POI_TYPE_SCENIC,
            "offset": max_results,
            "page": 1,
            "extensions": "all",
        },
    )
    if not res["ok"] or res["data"].get("status") != "1":
        return None
    pois = res["data"].get("pois", [])
    items: list[dict] = []
    for p in pois[:max_results]:
        items.append({
            "name": p.get("name"),
            "address": p.get("address"),
            "type": p.get("type"),
            "tel": p.get("tel") or None,
            "biz_ext": (p.get("biz_ext") or {}).get("rating") or None,
            "location": p.get("location"),  # "lon,lat"
            "source": "amap",
        })
    return items


def _attractions_wiki(lat: float, lon: float, max_results: int) -> list[dict]:
    """Wikipedia geosearch：返回坐标附近的 wiki 条目。"""
    res = http_get(
        "https://zh.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "list": "geosearch",
            "gscoord": f"{lat}|{lon}",
            "gsradius": 10000,  # 10km
            "gslimit": max_results,
            "format": "json",
        },
    )
    if not res["ok"]:
        return []
    pages = res["data"].get("query", {}).get("geosearch", [])
    return [
        {
            "name": p["title"],
            "address": f"距中心约 {p.get('dist', 0)} 米",
            "wiki_url": f"https://zh.wikipedia.org/wiki/{p['title']}",
            "source": "wikipedia",
        }
        for p in pages
    ]


@tool("search_attractions", return_direct=False)
def search_attractions(
    city: str,
    keyword: str | None = None,
    max_results: int = 8,
) -> dict[str, Any]:
    """查询某城市的热门景点 / 兴趣点。

    Args:
        city: 城市名，例如 "西安"、"杭州"。
        keyword: 可选关键词，例如 "博物馆"、"古城"、"亲子"。
        max_results: 最多返回多少条，默认 8。

    Returns:
        {"city": ..., "items": [...], "count": N}
        items 为空时，建议 Agent 用自身知识补充该城市的代表景点。
    """
    max_results = max(1, min(int(max_results or 8), 15))

    # 1) 高德优先
    items = _attractions_amap(city, keyword, max_results)

    # 2) 兜底用 Wikipedia geosearch
    if not items:
        geo = _geocode_tool.invoke({"address": city})
        if "error" not in geo:
            items = _attractions_wiki(geo["latitude"], geo["longitude"], max_results)
        else:
            items = []

    if not items:
        return {
            "city": city,
            "items": [],
            "count": 0,
            "note": (
                "未能从外部数据源拉到景点（可能未配置 AMAP_API_KEY 且 Wikipedia 不可达）。"
                "请基于你自身知识列出该地代表景点，并注明数据来源为模型常识。"
            ),
        }

    return {"city": city, "keyword": keyword, "items": items, "count": len(items)}
