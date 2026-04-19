"""依赖诊断脚本：逐一测试 Embedding / LLM / Rerank，打印每一步的真实结果或错误。

用法（项目根目录、激活 venv 后）::

    python -m scripts.diagnose

会打印：
1. .env 加载到的关键配置
2. 本地 / DashScope embedding 是否可用
3. Qwen LLM 是否能正常返回内容
4. DashScope rerank 是否能正常返回（直接走原生 SDK，能拿到详细错误码）
"""

from __future__ import annotations

import sys
import traceback
from typing import Any


def _section(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def _ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def _info(msg: str) -> None:
    print(f"         {msg}")


# ---------------------------------------------------------------- 1. 配置
def check_config() -> Any:
    _section("1) 加载配置（.env）")
    from src.config import get_settings

    cfg = get_settings()
    _info(f"LLM_MODEL           = {cfg.llm_model}")
    _info(f"EMBEDDING_PROVIDER  = {cfg.embedding_provider}")
    if cfg.embedding_provider == "huggingface":
        _info(f"HF_EMBEDDING_MODEL  = {cfg.hf_embedding_model}")
        _info(f"HF_EMBEDDING_DEVICE = {cfg.hf_embedding_device}")
    else:
        _info(f"EMBEDDING_MODEL     = {cfg.embedding_model}")
    _info(f"RERANK_MODEL        = {cfg.rerank_model}")
    if cfg.dashscope_api_key:
        masked = cfg.dashscope_api_key[:6] + "..." + cfg.dashscope_api_key[-4:]
        _info(f"DASHSCOPE_API_KEY   = {masked}")
    else:
        _fail("DASHSCOPE_API_KEY 未配置")
    return cfg


# ---------------------------------------------------------------- 2. Embedding
def check_embedding(cfg) -> None:
    _section("2) 测试 Embedding")
    try:
        from src.settings import _build_embedding  # type: ignore[attr-defined]

        embed = _build_embedding(cfg)
        vec = embed.get_query_embedding("测试一下中文 embedding 是否能正常工作。")
        _ok(f"维度={len(vec)}, 前 5 维={[round(v, 4) for v in vec[:5]]}")
    except Exception as exc:  # noqa: BLE001
        _fail(f"embedding 调用失败: {exc!r}")
        traceback.print_exc()


# ---------------------------------------------------------------- 3. LLM
def check_llm(cfg) -> None:
    _section("3) 测试 LLM（Qwen via DashScope）")
    try:
        from llama_index.llms.dashscope import DashScope

        llm = DashScope(
            model_name=cfg.llm_model,
            api_key=cfg.dashscope_api_key,
        )
        resp = llm.complete("用一句话说明你是谁。")
        text = str(resp).strip()
        if text:
            _ok(f"LLM 输出：{text[:200]}")
        else:
            _fail("LLM 返回了空字符串！这通常意味着 API key 没有该模型权限，"
                  "或模型名错误，或被限流。")
            _info(f"原始 response 对象：{resp!r}")
    except Exception as exc:  # noqa: BLE001
        _fail(f"LLM 调用失败: {exc!r}")
        traceback.print_exc()


# ---------------------------------------------------------------- 4. Rerank
def check_rerank(cfg) -> None:
    _section("4) 测试 Rerank（直接走 dashscope SDK，能看到详细错误码）")
    try:
        import dashscope
        from dashscope import TextReRank

        dashscope.api_key = cfg.dashscope_api_key
        resp = TextReRank.call(
            model=cfg.rerank_model,
            query="衣服怎么洗",
            documents=[
                "衣服洗涤方式：建议使用 30 度温水手洗，避免高温缩水。",
                "今天天气很好，适合出门散步。",
                "牛仔裤建议反面冷水洗，避免褪色。",
            ],
            top_n=2,
            return_documents=True,
        )
        _info(f"status_code = {resp.status_code}")
        _info(f"request_id  = {getattr(resp, 'request_id', None)}")
        _info(f"code        = {getattr(resp, 'code', None)}")
        _info(f"message     = {getattr(resp, 'message', None)}")
        _info(f"output      = {resp.output!r}")
        if resp.status_code == 200 and resp.output is not None:
            _ok(f"rerank 调用成功，共 {len(resp.output.results)} 条结果")
            for r in resp.output.results:
                idx = r.index
                score = r.relevance_score
                doc = r.document.get("text") if isinstance(r.document, dict) else r.document
                _info(f"  #{idx} score={score:.4f}  {doc}")
        else:
            _fail("rerank 调用失败 —— 看上面 message 字段定位原因。")
            _info("常见原因：")
            _info("  • model 名称错误：在百炼控制台 -> 模型广场 找到正确名称")
            _info("  • api_key 没开通该模型权限：去模型详情页点『立即开通』")
            _info("  • 余额不足：去百炼控制台 -> 费用充值")
            _info("  • 海外账号 vs 国内账号 endpoint 不匹配")
    except Exception as exc:  # noqa: BLE001
        _fail(f"rerank 调用过程出现 Python 异常: {exc!r}")
        traceback.print_exc()


def main() -> int:
    print("RAG 项目依赖自检 ……")
    cfg = check_config()
    check_embedding(cfg)
    check_llm(cfg)
    check_rerank(cfg)
    print("\n诊断完成。\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
