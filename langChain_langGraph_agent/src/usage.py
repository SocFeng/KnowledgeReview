"""Token 用量统计 + 成本估算 + 按 thread 持久化。

设计要点
--------
- 每一轮对话（一次用户输入 → Agent 最终回答）在 LangGraph 里会触发
  多次 LLM 调用（ReAct：决策 → 工具 → 决策 → ...），我们在 agent 层
  把每个 AIMessage / AIMessageChunk 的 ``usage_metadata`` 汇总，一轮结束
  再把 (input_tokens, output_tokens, cost_cny) 追加到 ``data/usage/{thread}.json``。
- 价格表手工维护在 ``PRICING`` 里，模型换新或官方调价时更新这张表即可。
- 小工具函数全部是纯 IO，不依赖 LangChain，方便单测。

文件结构：
    data/usage/{thread_id}.json
    {
        "model": "qwen-plus",
        "turns": [
            {"time": "2026-04-19T21:00:00", "input": 1234, "output": 567,
             "cost_cny": 0.00212},
            ...
        ],
        "total": {"input": N, "output": N, "total": N, "cost_cny": X}
    }
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import settings

# DashScope / 通义千问 定价（元 / 1K tokens）
# 数字为 2026 年初官方公开价格，若官方调价请自行更新。
# key 使用模型名前缀匹配，未命中则用 "_default"。
PRICING: dict[str, dict[str, float]] = {
    "qwen-max": {"input": 0.02, "output": 0.06},
    "qwen-plus": {"input": 0.0008, "output": 0.002},
    "qwen-turbo": {"input": 0.0003, "output": 0.0006},
    "qwen-long": {"input": 0.0005, "output": 0.002},
    "qwen2.5-72b-instruct": {"input": 0.004, "output": 0.012},
    "qwen2.5-32b-instruct": {"input": 0.0035, "output": 0.007},
    "qwen2.5-14b-instruct": {"input": 0.001, "output": 0.003},
    "qwen2.5-7b-instruct": {"input": 0.0005, "output": 0.001},
    "_default": {"input": 0.001, "output": 0.002},
}


def get_pricing(model: str) -> dict[str, float]:
    """按前缀匹配模型价格，未匹配到回落到 ``_default``。"""
    if model in PRICING:
        return PRICING[model]
    for key, price in PRICING.items():
        if key != "_default" and model.startswith(key):
            return price
    return PRICING["_default"]


def compute_cost(input_tokens: int, output_tokens: int, model: str | None = None) -> float:
    """按千 token 单价算本轮费用，单位：人民币元。"""
    model = model or settings.LLM_MODEL
    price = get_pricing(model)
    cost = (input_tokens * price["input"] + output_tokens * price["output"]) / 1000.0
    return round(cost, 6)


# ---------- 持久化 ----------
def _usage_dir() -> Path:
    d = settings.data_dir / "usage"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _usage_file(thread_id: str) -> Path:
    return _usage_dir() / f"{thread_id}.json"


def _empty_record() -> dict[str, Any]:
    return {
        "model": settings.LLM_MODEL,
        "turns": [],
        "total": {"input": 0, "output": 0, "total": 0, "cost_cny": 0.0},
    }


def load_usage(thread_id: str) -> dict[str, Any]:
    """读取某 thread 的用量记录，文件不存在返回空壳。"""
    f = _usage_file(thread_id)
    if not f.exists():
        return _empty_record()
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return _empty_record()


def save_turn(
    thread_id: str,
    input_tokens: int,
    output_tokens: int,
    model: str | None = None,
) -> dict[str, Any]:
    """追加一轮用量并更新累计。返回"本轮"统计，方便前端立即显示。"""
    model = model or settings.LLM_MODEL
    cost = compute_cost(input_tokens, output_tokens, model)
    record = load_usage(thread_id)
    record["model"] = model
    turn = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "input": int(input_tokens),
        "output": int(output_tokens),
        "cost_cny": cost,
    }
    record["turns"].append(turn)
    total = record["total"]
    total["input"] = int(total.get("input", 0)) + int(input_tokens)
    total["output"] = int(total.get("output", 0)) + int(output_tokens)
    total["total"] = total["input"] + total["output"]
    total["cost_cny"] = round(float(total.get("cost_cny", 0.0)) + cost, 6)
    _usage_file(thread_id).write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return turn


def reset_usage(thread_id: str) -> None:
    """清空某 thread 的用量记录，配合 reset_thread 一起用。"""
    f = _usage_file(thread_id)
    if f.exists():
        try:
            f.unlink()
        except Exception:
            pass


def summarize_total(thread_id: str) -> dict[str, Any]:
    """只返回累计值，前端侧边栏用。"""
    return load_usage(thread_id).get(
        "total", {"input": 0, "output": 0, "total": 0, "cost_cny": 0.0}
    )


__all__ = [
    "PRICING",
    "get_pricing",
    "compute_cost",
    "load_usage",
    "save_turn",
    "reset_usage",
    "summarize_total",
]
