"""一键体检：验证环境 / LLM / 各工具是否正常工作。

用法：
    python -m scripts.diagnose
"""

from __future__ import annotations

import io
import sys
import traceback
from pathlib import Path

# Windows 默认控制台是 gbk，输出 emoji 会崩；强制 utf-8
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# 让脚本能直接 `python -m scripts.diagnose` 跑
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _ok(msg: str) -> None:
    print(f"  ✅ {msg}")


def _warn(msg: str) -> None:
    print(f"  ⚠️  {msg}")


def _fail(msg: str) -> None:
    print(f"  ❌ {msg}")


def _section(title: str) -> None:
    print(f"\n=== {title} ===")


def check_config() -> bool:
    _section("Step 1 / 6  配置")
    try:
        from src.config import settings
        if not settings.DASHSCOPE_API_KEY:
            _fail("未设置 DASHSCOPE_API_KEY，请编辑 .env")
            return False
        _ok(f"LLM 模型：{settings.LLM_MODEL}")
        _ok(f"高德 Key：{'已配置' if settings.has_amap else '未配置（路线/景点会 fallback）'}")
        _ok(f"数据目录：{settings.data_dir}")
        return True
    except Exception:
        traceback.print_exc()
        return False


def check_llm() -> bool:
    _section("Step 2 / 6  LLM (Qwen via DashScope)")
    try:
        from src.llm import get_llm
        resp = get_llm().invoke("用一句话介绍北京。")
        _ok(f"模型回答：{str(resp.content)[:80]}...")
        return True
    except Exception as e:
        _fail(f"LLM 调用失败：{type(e).__name__}: {e}")
        return False


def check_tool_geocode() -> bool:
    _section("Step 3 / 6  geocode_address")
    try:
        from src.tools import geocode_address
        r = geocode_address.invoke({"address": "北京天安门"})
        if "error" in r:
            _warn(f"返回错误：{r['error']}")
            return False
        _ok(f"天安门坐标：lat={r['latitude']}, lon={r['longitude']}  (源：{r['source']})")
        return True
    except Exception as e:
        _fail(f"{type(e).__name__}: {e}")
        return False


def check_tool_weather() -> bool:
    _section("Step 4 / 6  get_weather_forecast")
    try:
        from src.tools import get_weather_forecast
        r = get_weather_forecast.invoke({"location": "成都", "days": 3})
        if "error" in r:
            _warn(f"返回错误：{r['error']}")
            return False
        days = r.get("forecast", [])
        if not days:
            _warn("forecast 为空")
            return False
        _ok(f"成都未来 {len(days)} 天天气：")
        for d in days:
            print(f"     · {d['date']}  {d['weather']}  {d['t_min']}~{d['t_max']}°C  雨量 {d['precipitation_mm']}mm")
        return True
    except Exception as e:
        _fail(f"{type(e).__name__}: {e}")
        return False


def check_tool_distance() -> bool:
    _section("Step 5 / 6  compute_distance")
    try:
        from src.tools import compute_distance
        r = compute_distance.invoke({"origin": "北京", "destination": "上海"})
        if "error" in r:
            _warn(f"返回错误：{r['error']}")
            return False
        _ok(f"北京 → 上海 ≈ {r['distance_km']} km，建议 {r['transport_suggestion']['primary']}")
        return True
    except Exception as e:
        _fail(f"{type(e).__name__}: {e}")
        return False


def check_tool_attractions() -> bool:
    _section("Step 6 / 6  search_attractions / lookup_culture / plan_route（综合）")
    ok = True
    try:
        from src.tools import search_attractions, lookup_culture, plan_route
        r = search_attractions.invoke({"city": "西安", "max_results": 3})
        if r["count"] == 0:
            _warn(f"景点为空：{r.get('note')}")
        else:
            _ok(f"西安景点 Top {r['count']}：{[i['name'] for i in r['items']]}")
    except Exception as e:
        _fail(f"search_attractions：{e}")
        ok = False

    try:
        r = lookup_culture.invoke({"place_or_topic": "西安"})
        if "error" in r:
            _warn(f"lookup_culture: {r['error']}")
        else:
            _ok(f"西安文化简介：{r['summary'][:60]}...")
    except Exception as e:
        _fail(f"lookup_culture：{e}")
        ok = False

    try:
        r = plan_route.invoke({"origin": "西安钟楼", "destination": "西安大雁塔", "mode": "driving"})
        if "error" in r:
            _warn(f"plan_route: {r['error']}")
        else:
            _ok(f"路线：{r['distance_km']} km，{r['duration_min']} 分钟（源：{r['source']}）")
    except Exception as e:
        _fail(f"plan_route：{e}")
        ok = False

    return ok


def main() -> int:
    print("\n🩺  TravelAgent 体检脚本\n" + "=" * 40)
    results = [
        check_config(),
        check_llm(),
        check_tool_geocode(),
        check_tool_weather(),
        check_tool_distance(),
        check_tool_attractions(),
    ]
    print("\n" + "=" * 40)
    if all(results):
        print("🎉 全部通过，可以启动 Web UI：")
        print("    streamlit run streamlit_app.py\n")
        return 0
    print("⚠️  存在失败项，请按上面提示修复后再启动。\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
