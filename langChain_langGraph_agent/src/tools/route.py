"""路线规划工具：出发地 → 目的地的真实道路路线。

数据源策略：
- 高德 driving / walking / bicycling / transit（4 种模式）
- 没 key 时，回退到 distance 工具的"直线 + 推荐交通方式"
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.tools import tool

from ..config import settings
from ._http import http_get
from .distance import compute_distance as _distance_tool
from .geocode import geocode_address as _geocode_tool

Mode = Literal["driving", "walking", "bicycling", "transit"]

_MODE_TO_AMAP_PATH = {
    "driving": "driving",
    "walking": "walking",
    "bicycling": "bicycling",
    "transit": "transit/integrated",
}


def _route_amap(
    o_lon: float,
    o_lat: float,
    d_lon: float,
    d_lat: float,
    mode: Mode,
    city: str | None,
) -> dict[str, Any] | None:
    if not settings.has_amap:
        return None
    path = _MODE_TO_AMAP_PATH.get(mode, "driving")

    params = {
        "key": settings.AMAP_API_KEY,
        "origin": f"{o_lon},{o_lat}",
        "destination": f"{d_lon},{d_lat}",
    }
    if mode == "transit":
        if not city:
            return {"error": "公交模式需提供 city（高德要求）"}
        params["city"] = city

    res = http_get(f"https://restapi.amap.com/v3/direction/{path}", params=params)
    if not res["ok"]:
        return {"error": res["error"]}
    data = res["data"]
    if data.get("status") != "1":
        return {"error": data.get("info", "高德路线规划失败")}

    route = data.get("route", {})
    paths = route.get("paths") or route.get("transits") or []
    if not paths:
        return {"error": "高德返回空路径"}
    p = paths[0]

    # 不同模式字段不一致，做下兼容
    distance_m = int(p.get("distance", 0))
    duration_s = int(p.get("duration", 0))
    return {
        "mode": mode,
        "distance_km": round(distance_m / 1000, 2),
        "duration_min": round(duration_s / 60, 1),
        "tolls_yuan": p.get("tolls"),
        "steps_summary": [s.get("instruction") for s in p.get("steps", [])][:8],
        "source": "amap",
    }


@tool("plan_route", return_direct=False)
def plan_route(
    origin: str,
    destination: str,
    mode: str = "driving",
    city: str | None = None,
) -> dict[str, Any]:
    """规划从出发地到目的地的具体路线（含距离、用时、关键步骤）。

    Args:
        origin: 出发地，地址或地名。
        destination: 目的地。
        mode: 出行方式，可选 "driving"（驾车）/ "walking"（步行）/ "bicycling"（骑行）/ "transit"（公交）。
        city: 公交模式必填，例如 "北京"。

    Returns:
        {mode, distance_km, duration_min, steps_summary, source}；
        若无 AMAP_API_KEY，会回退为 compute_distance 的直线估算。
    """
    mode = (mode or "driving").lower()
    if mode not in _MODE_TO_AMAP_PATH:
        mode = "driving"

    g1 = _geocode_tool.invoke({"address": origin})
    g2 = _geocode_tool.invoke({"address": destination})
    if "error" in g1:
        return {"error": f"出发地解析失败：{g1['error']}"}
    if "error" in g2:
        return {"error": f"目的地解析失败：{g2['error']}"}

    # 高德优先
    amap = _route_amap(
        g1["longitude"], g1["latitude"],
        g2["longitude"], g2["latitude"],
        mode, city,
    )
    if amap and "distance_km" in amap:
        amap.update({"origin": g1["formatted_address"], "destination": g2["formatted_address"]})
        return amap

    # 回退：直线距离 + 经验估算
    fallback = _distance_tool.invoke({"origin": origin, "destination": destination})
    if "error" in fallback:
        return fallback
    # 把字段名对齐高德返回，避免上层 LLM 看到两套字段名困惑
    fallback["mode"] = mode
    fallback["source"] = "fallback (haversine)"
    fallback["duration_min"] = (fallback.get("transport_suggestion") or {}).get("estimate_time_min")
    fallback["steps_summary"] = [
        f"建议方式：{(fallback.get('transport_suggestion') or {}).get('primary')}",
        (fallback.get('transport_suggestion') or {}).get('reason', ''),
    ]
    fallback["note"] = (
        "未配置 AMAP_API_KEY，已回退为直线距离 + 经验估算；"
        "如需真实道路里程，请在 .env 中填入 AMAP_API_KEY。"
    )
    return fallback
