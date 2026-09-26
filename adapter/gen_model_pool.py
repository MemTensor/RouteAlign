"""
模型池配置模板生成工具（可选便捷脚本）。

模型服务属性标签本应由用户自行配置；本脚本仅从 bench 数据中估算价格
（对 cost = pt*p_in/1e6 + ct*p_out/1e6 做最小二乘拟合），并结合内置的
公开上下文窗口/多模态信息生成一份模板，供用户检查和修改。

sensitive_data_allowed 全部赋值为 true（敏感数据检测暂不实现）。

用法：
    python -m adapter.gen_model_pool \
        --config config/adapter_config.yaml \
        --output config/model_pool.yaml
"""

import argparse
from collections import defaultdict

import yaml
from loguru import logger

try:
    from baselines.data_loader import BaselineDataLoader
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "gen_model_pool requires the LLMRouterBench package "
        "(baselines.data_loader) to read benchmark usage records. A curated "
        "model pool already ships at config/model_pool.yaml, so this script is "
        "only needed when regenerating the pool template from raw records."
    ) from exc

from .config import AdapterConfig

# 公开资料中的上下文窗口与多模态支持（用户可在生成后修改）
KNOWN_MODEL_INFO = {
    "claude-sonnet-4": {"context_window": 200000, "multimodal": ["vision"]},
    "deepseek-v3-0324": {"context_window": 131072, "multimodal": []},
    "deepseek-v3.1-terminus": {"context_window": 131072, "multimodal": []},
    "deepseek-r1-0528": {"context_window": 131072, "multimodal": []},
    "gemini-2.5-flash": {"context_window": 1048576, "multimodal": ["vision", "audio", "video"]},
    "gemini-2.5-pro": {"context_window": 1048576, "multimodal": ["vision", "audio", "video"]},
    "gpt-5-chat": {"context_window": 128000, "multimodal": ["vision"]},
    "gpt-5": {"context_window": 400000, "multimodal": ["vision"]},
    "qwen3-235b-a22b-2507": {"context_window": 262144, "multimodal": []},
    "qwen3-235b-a22b-thinking-2507": {"context_window": 262144, "multimodal": []},
    "glm-4.6": {"context_window": 200000, "multimodal": []},
    "kimi-k2-0905": {"context_window": 262144, "multimodal": []},
    "intern-s1": {"context_window": 131072, "multimodal": ["vision"]},
}

DEFAULT_CONTEXT_WINDOW = 131072


def estimate_prices(records_by_model):
    """对每个模型用最小二乘估算每百万 token 的输入/输出价格。"""
    try:
        import numpy as np
    except ImportError:
        logger.warning("numpy not available; prices will be left as 0")
        return {}

    prices = {}
    for model, rows in records_by_model.items():
        rows = [r for r in rows if r[2] > 0]  # 只用 cost>0 的记录
        if len(rows) < 5:
            prices[model] = (0.0, 0.0)
            continue
        A = np.array([[r[0] / 1e6, r[1] / 1e6] for r in rows])
        b = np.array([r[2] for r in rows])
        solution, *_ = np.linalg.lstsq(A, b, rcond=None)
        p_in, p_out = max(0.0, float(solution[0])), max(0.0, float(solution[1]))
        prices[model] = (round(p_in, 4), round(p_out, 4))
    return prices


def main():
    parser = argparse.ArgumentParser(description="Generate model pool config template")
    parser.add_argument(
        "--config",
        default="config/adapter_config.yaml",
    )
    parser.add_argument(
        "--output",
        default="config/model_pool.yaml",
    )
    args = parser.parse_args()

    adapter_config = AdapterConfig.load(args.config)
    loader = BaselineDataLoader(config_path=adapter_config.baseline_config)

    records_by_model = defaultdict(list)
    for record in loader.load_records_iter():
        records_by_model[record.model_name].append(
            (record.prompt_tokens, record.completion_tokens, record.cost)
        )

    prices = estimate_prices(records_by_model)

    models = []
    for model in sorted(records_by_model):
        info = KNOWN_MODEL_INFO.get(model, {})
        p_in, p_out = prices.get(model, (0.0, 0.0))
        models.append({
            "model_id": model,
            "service_attributes": {
                "cost": {
                    "input_price": p_in,
                    "output_price": p_out,
                    "cache_read_price": 0.0,
                    "cache_write_price": 0.0,
                },
                "latency": {
                    "avg_first_token_latency_ms": 1000,
                    "avg_output_tokens_per_second": 50,
                    "p95_latency_ms": 10000,
                },
                "context_window": info.get("context_window", DEFAULT_CONTEXT_WINDOW),
                "tool_supported": True,
                "multimodal_supported": info.get("multimodal", []),
                "sensitive_data_allowed": True,
                "available": True,
            },
            # 线上真正调用该模型时的端点配置（离线查表评估不需要）
            # "endpoint": {"model": "...", "base_url": "...", "api_key": "ENV_VAR_NAME"},
        })

    output = {"models": models}
    with open(args.output, "w", encoding="utf-8") as f:
        yaml.safe_dump(output, f, allow_unicode=True, sort_keys=False, width=100)

    logger.info(f"Model pool template written to {args.output} ({len(models)} models)")
    logger.info("请人工核对各模型的价格、上下文窗口、多模态支持等属性")


if __name__ == "__main__":
    main()
