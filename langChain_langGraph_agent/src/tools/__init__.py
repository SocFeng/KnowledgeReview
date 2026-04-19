"""所有自定义工具的统一出口。

设计哲学：
1. 每个工具独立成文件，便于调试 / 单测 / 替换数据源；
2. 一律用 `@tool` 装饰，自动暴露 OpenAI tool-calling 格式；
3. 工具内部自己处理失败（永远返回 dict / str，不抛异常给 Agent，
   失败时返回 {"error": "..."}），Agent 看到 error 字段就能自己重试或换路；
4. 优先免费、无 key 的数据源（OSM / Open-Meteo / Wikipedia），
   有 AMAP_API_KEY 时再升级到高德数据。
"""

from .geocode import geocode_address
from .weather import get_weather_forecast
from .distance import compute_distance
from .attractions import search_attractions
from .culture import lookup_culture
from .route import plan_route

ALL_TOOLS = [
    geocode_address,
    get_weather_forecast,
    compute_distance,
    search_attractions,
    lookup_culture,
    plan_route,
]

__all__ = [
    "ALL_TOOLS",
    "geocode_address",
    "get_weather_forecast",
    "compute_distance",
    "search_attractions",
    "lookup_culture",
    "plan_route",
]
