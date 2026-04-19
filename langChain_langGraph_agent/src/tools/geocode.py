"""地理编码工具：地址 → (经度, 纬度, 标准化名称)。

数据源策略（按优先级 fallback）：
1. 高德地图 geocode    —— 中文地址识别准、字段细（需 AMAP_API_KEY）
2. Open-Meteo geocoding —— 免费无 key，对 UA / IP 限制宽，全球可用
3. OpenStreetMap Nominatim —— 最后兜底（部分网络下会 403）
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from ..config import settings
from ._http import http_get


def _geocode_amap(address: str, city: str | None = None) -> dict[str, Any] | None:
    """高德地理编码 API。https://lbs.amap.com/api/webservice/guide/api/georegeo"""
    if not settings.has_amap:
        return None
    res = http_get(
        "https://restapi.amap.com/v3/geocode/geo",
        params={"key": settings.AMAP_API_KEY, "address": address, "city": city or ""},
    )
    if not res["ok"]:
        return {"error": res["error"], "source": "amap"}
    data = res["data"]
    if data.get("status") != "1" or not data.get("geocodes"):
        return None
    g = data["geocodes"][0]
    lon, lat = g["location"].split(",")
    return {
        "address_input": address,
        "formatted_address": g.get("formatted_address", address),
        "country": "中国",
        "province": g.get("province"),
        "city": g.get("city"),
        "district": g.get("district"),
        "longitude": float(lon),
        "latitude": float(lat),
        "source": "amap",
    }


def _query_open_meteo(name: str) -> dict[str, Any] | None:
    """单次查询 Open-Meteo geocoding。

    实测细节：
      - language=zh 时返回的结果**不含 population 字段**，常常拿不到真正的"北京/上海"，
        会被同名小村镇覆盖；
      - 不传 language → 返回英文 + 完整 population，便于按行政等级 + 人口排序；
    """
    res = http_get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": name, "count": 10, "format": "json"},
    )
    if not res["ok"]:
        return None
    results = (res["data"] or {}).get("results") or []
    if not results:
        return None

    # PPLC=首都, PPLA=一级行政中心(省会), PPLA2=二级(地级市)
    _RANK = {"PPLC": 4, "PPLA": 3, "PPLA2": 2, "PPLA3": 1}

    def _score(it: dict) -> tuple[int, int, int]:
        cc_match = 1 if it.get("country_code") == "CN" else 0
        rank = _RANK.get(it.get("feature_code", ""), 0)
        pop = int(it.get("population") or 0)
        return (cc_match, rank, pop)

    item = max(results, key=_score)
    parts = [item.get(k) for k in ("country", "admin1", "admin2", "name") if item.get(k)]
    return {
        "address_input": name,
        "formatted_address": " · ".join(parts) or item.get("name", name),
        "country": item.get("country"),
        "province": item.get("admin1"),
        "city": item.get("admin2") or item.get("name"),
        "longitude": float(item["longitude"]),
        "latitude": float(item["latitude"]),
        "source": "open-meteo",
    }


_LANDMARK_SUFFIXES = [
    "天安门", "故宫", "外滩", "钟楼", "鼓楼", "城墙", "古城", "古镇",
    "博物馆", "美术馆", "图书馆", "公园", "广场", "景区", "风景区",
    "大剧院", "体育馆", "新区", "高新区", "经开区", "开发区",
    "国际机场", "机场", "高铁站", "火车站", "汽车站", "码头", "港口",
]
_ADMIN_SUFFIXES = ["自治区", "自治州", "自治县", "省", "市", "区", "县"]


def _smart_keywords(address: str) -> list[str]:
    """把"北京天安门" / "西安钟楼" 这类长地名拆出几个候选关键词。

    Open-Meteo geocoding 走的是"行政区/地名"匹配，对具体景点支持有限，
    所以我们要：
      1) 优先去掉景点后缀，让"北京天安门" → "北京" 直接命中行政区
      2) 短地名直接用原文
      3) 最后兜底取前 2~3 字
    """
    addr = address.strip()
    candidates: list[str] = []

    # 1) 去掉"景点级"后缀的 stem 优先（这样"北京天安门"先尝试"北京"）
    for suffix in _LANDMARK_SUFFIXES:
        if addr.endswith(suffix) and len(addr) > len(suffix):
            stem = addr[: -len(suffix)].strip()
            if stem and stem not in candidates:
                candidates.append(stem)
            break

    # 2) 原文（如"成都""上海"这种短行政区名直接命中）
    if addr not in candidates:
        candidates.append(addr)

    # 3) 去掉行政区后缀（"北京市" → "北京"）
    for suffix in _ADMIN_SUFFIXES:
        if addr.endswith(suffix) and len(addr) > len(suffix):
            stem = addr[: -len(suffix)].strip()
            if stem and stem not in candidates:
                candidates.append(stem)
            break

    # 4) 兜底前 2~3 字
    if len(addr) >= 3 and addr[:2] not in candidates:
        candidates.append(addr[:2])
    if len(addr) >= 4 and addr[:3] not in candidates:
        candidates.append(addr[:3])
    return candidates


def _geocode_open_meteo(address: str) -> dict[str, Any] | None:
    """Open-Meteo geocoding：免费无 key，对中文支持也很好。

    https://open-meteo.com/en/docs/geocoding-api
    """
    for kw in _smart_keywords(address):
        item = _query_open_meteo(kw)
        if item:
            # 标识真实匹配的关键词，方便排查
            item["matched_keyword"] = kw
            item["address_input"] = address
            return item
    return None


def _geocode_osm(address: str) -> dict[str, Any]:
    """OpenStreetMap Nominatim，免费但请求频率受限（≤1 QPS）。"""
    res = http_get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": address, "format": "json", "limit": 1, "accept-language": "zh-CN"},
    )
    if not res["ok"]:
        return {"error": res["error"], "source": "osm"}
    data = res["data"]
    if not data:
        return {"error": f"未能解析地址：{address}", "source": "osm"}
    item = data[0]
    return {
        "address_input": address,
        "formatted_address": item.get("display_name", address),
        "longitude": float(item["lon"]),
        "latitude": float(item["lat"]),
        "source": "osm",
    }


@tool("geocode_address", return_direct=False)
def geocode_address(address: str, city: str | None = None) -> dict[str, Any]:
    """把地址 / 地名转成经纬度。

    Args:
        address: 待解析的地址或地名，例如 "北京天安门"、"上海外滩"、"东京塔"。
        city: 可选，限定城市以提高精度，例如 "上海市"。

    Returns:
        含 longitude / latitude / formatted_address 的字典；失败时返回 {"error": "..."}。
    """
    # 1) 高德优先
    amap_result = _geocode_amap(address, city)
    if amap_result and "longitude" in amap_result:
        return amap_result
    # 2) Open-Meteo geocoding（更稳定）
    om = _geocode_open_meteo(address)
    if om:
        return om
    # 3) 兜底 OSM Nominatim
    return _geocode_osm(address)
