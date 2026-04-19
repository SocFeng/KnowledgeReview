"""距离计算工具：两点直线距离 + 推荐交通方式。

不依赖任何外部 API，纯 haversine 公式，离线可用，
非常适合作为"非外部 API 工具"的示范。
"""

from __future__ import annotations

import math
from typing import Any

from langchain_core.tools import tool

from .geocode import geocode_address as _geocode_tool

EARTH_RADIUS_KM = 6371.0


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """球面两点距离（公里）。"""
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return EARTH_RADIUS_KM * c


def _suggest_transport(km: float) -> dict[str, Any]:
    """根据直线距离粗略给出交通建议。"""
    if km < 5:
        return {
            "primary": "步行 / 共享单车 / 出租车",
            "estimate_time_min": int(km * 12),  # 步行 ≈ 5km/h
            "reason": "短途，市内不堵的话打车 5~15 分钟",
        }
    if km < 50:
        return {
            "primary": "出租车 / 网约车 / 地铁",
            "estimate_time_min": int(km * 2.5),  # 假设市内 25 km/h
            "reason": "市内或近郊，公共交通比较合适",
        }
    if km < 300:
        return {
            "primary": "高铁 / 城际 / 自驾",
            "estimate_time_min": int(km / 200 * 60),  # 高铁 ~200 km/h
            "reason": "城际短途，高铁通常 1~2 小时",
        }
    if km < 1500:
        return {
            "primary": "高铁 / 飞机",
            "estimate_time_min": int(km / 250 * 60),
            "reason": "中距离，高铁 4~6 小时或飞机 2 小时左右",
        }
    return {
        "primary": "飞机",
        "estimate_time_min": int(km / 750 * 60) + 90,  # 含值机/安检
        "reason": "长距离，建议直飞",
    }


@tool("compute_distance", return_direct=False)
def compute_distance(origin: str, destination: str) -> dict[str, Any]:
    """计算两个地点之间的直线距离，并推荐交通方式。

    Args:
        origin: 出发地名称或地址，例如 "北京"。
        destination: 目的地名称或地址，例如 "上海"。

    Returns:
        {
            "origin": ..., "destination": ...,
            "distance_km": float,         # 球面直线距离
            "transport_suggestion": {...}, # 推荐交通方式 + 估算时间
        }
    """
    g1 = _geocode_tool.invoke({"address": origin})
    g2 = _geocode_tool.invoke({"address": destination})
    if "error" in g1:
        return {"error": f"出发地解析失败：{g1['error']}"}
    if "error" in g2:
        return {"error": f"目的地解析失败：{g2['error']}"}

    km = round(
        _haversine_km(g1["latitude"], g1["longitude"], g2["latitude"], g2["longitude"]),
        2,
    )
    return {
        "origin": g1.get("formatted_address", origin),
        "destination": g2.get("formatted_address", destination),
        "distance_km": km,
        "transport_suggestion": _suggest_transport(km),
        "note": "直线距离仅供参考，实际道路里程会更长（通常 ×1.2~1.4）",
    }
