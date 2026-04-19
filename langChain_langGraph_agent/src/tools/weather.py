"""天气查询工具：基于经纬度查询未来几天的天气预报。

数据源：Open-Meteo（开源、免费、无需 key、全球覆盖）
官网：https://open-meteo.com/en/docs
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from ._http import http_get
from .geocode import geocode_address as _geocode_tool

# WMO weather code → 中文描述
# 参考：https://open-meteo.com/en/docs#weathervariables
_WEATHER_CODE_ZH = {
    0: "晴",
    1: "多云", 2: "局部多云", 3: "阴",
    45: "雾", 48: "凇雾",
    51: "小毛毛雨", 53: "中等毛毛雨", 55: "大毛毛雨",
    56: "冻毛毛雨", 57: "大冻毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    66: "冻雨", 67: "大冻雨",
    71: "小雪", 73: "中雪", 75: "大雪",
    77: "雪粒",
    80: "阵雨", 81: "中阵雨", 82: "强阵雨",
    85: "阵雪", 86: "强阵雪",
    95: "雷阵雨", 96: "雷阵雨伴小冰雹", 99: "雷阵雨伴大冰雹",
}


def _weather_code_text(code: int | None) -> str:
    if code is None:
        return "未知"
    return _WEATHER_CODE_ZH.get(int(code), f"代码 {code}")


@tool("get_weather_forecast", return_direct=False)
def get_weather_forecast(
    location: str,
    days: int = 3,
) -> dict[str, Any]:
    """查询某个地点未来 N 天的天气预报（数据源：Open-Meteo，免费）。

    Args:
        location: 地名或地址，例如 "成都"、"杭州西湖"。会先内部地理编码。
        days: 预报天数，1~7。默认 3 天。

    Returns:
        含 location / forecast 的字典：
            forecast: List[{date, weather, t_max, t_min, precipitation_mm}]
        失败时返回 {"error": "..."}。
    """
    days = max(1, min(int(days or 3), 7))

    # 1) 地理编码（直接调用底层函数；@tool 包装的只是供 LLM，内部互相调用走原函数）
    geo = _geocode_tool.invoke({"address": location})
    if "error" in geo:
        return {"error": f"地理编码失败：{geo['error']}", "location": location}

    lat, lon = geo["latitude"], geo["longitude"]

    # 2) 调 Open-Meteo
    res = http_get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "daily": "weathercode,temperature_2m_max,temperature_2m_min,precipitation_sum",
            "timezone": "Asia/Shanghai",
            "forecast_days": days,
        },
    )
    if not res["ok"]:
        return {"error": res["error"], "location": location}

    daily = res["data"].get("daily", {})
    dates = daily.get("time", [])
    codes = daily.get("weathercode", [])
    tmax = daily.get("temperature_2m_max", [])
    tmin = daily.get("temperature_2m_min", [])
    rain = daily.get("precipitation_sum", [])

    forecast = []
    for i, date in enumerate(dates):
        forecast.append({
            "date": date,
            "weather": _weather_code_text(codes[i] if i < len(codes) else None),
            "t_max": tmax[i] if i < len(tmax) else None,
            "t_min": tmin[i] if i < len(tmin) else None,
            "precipitation_mm": rain[i] if i < len(rain) else None,
        })

    return {
        "location": geo.get("formatted_address", location),
        "longitude": lon,
        "latitude": lat,
        "days": len(forecast),
        "forecast": forecast,
        "source": "open-meteo",
    }
