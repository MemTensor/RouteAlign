"""
RouteAlign 配置加载。

配置文件（YAML）集中管理各步骤所需的 LLM 信息与其他参数：
打分模型、任务画像模型、画像聚合参数、路由策略、数据路径等。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .llm_client import LLMEndpoint
from .router import RoutingPolicy


@dataclass
class DemoRouterConfig:
    """RouteAlign 全局配置。"""

    # LLM 端点
    grading_model: Dict[str, Any] = field(default_factory=dict)
    task_profile_model: Dict[str, Any] = field(default_factory=dict)

    # 画像聚合参数
    relevance_threshold: float = 0.5
    confidence_thresholds: Dict[str, int] = field(
        default_factory=lambda: {"low": 3, "medium": 10, "high": 30}
    )
    tier_ratios: Dict[str, float] = field(
        default_factory=lambda: {"expert": 0.25, "strong": 0.33, "standard": 0.50}
    )

    # 路由策略
    routing_mode: str = "best_performance"
    max_cost_per_million_output: Optional[float] = None
    max_first_token_latency_ms: Optional[float] = None
    difficulty_tier_map: Optional[Dict[str, str]] = None
    disable_upward_fallback: bool = False

    # 数据路径
    question_bank_path: str = ""
    model_pool_path: str = ""
    grading_results_path: str = ""
    model_profiles_path: str = ""

    # 并发
    max_workers: int = 8

    @classmethod
    def load(cls, path: str) -> "DemoRouterConfig":
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"RouteAlign config not found: {path}")
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        cfg = data.get("demo_router", data)

        profiling = cfg.get("profiling") or {}
        routing = cfg.get("routing") or {}
        paths = cfg.get("paths") or {}

        return cls(
            grading_model=cfg.get("grading_model") or {},
            task_profile_model=cfg.get("task_profile_model") or {},
            relevance_threshold=float(profiling.get("relevance_threshold", 0.5)),
            confidence_thresholds=dict(
                profiling.get("confidence_thresholds")
                or {"low": 3, "medium": 10, "high": 30}
            ),
            tier_ratios=dict(
                profiling.get("tier_ratios")
                or {"expert": 0.25, "strong": 0.33, "standard": 0.50}
            ),
            routing_mode=routing.get("mode", "best_performance"),
            max_cost_per_million_output=routing.get("max_cost_per_million_output"),
            max_first_token_latency_ms=routing.get("max_first_token_latency_ms"),
            difficulty_tier_map=routing.get("difficulty_tier_map"),
            disable_upward_fallback=bool(routing.get("disable_upward_fallback", False)),
            question_bank_path=paths.get("question_bank", ""),
            model_pool_path=paths.get("model_pool", ""),
            grading_results_path=paths.get("grading_results", ""),
            model_profiles_path=paths.get("model_profiles", ""),
            max_workers=int(cfg.get("max_workers", 8)),
        )

    def grading_endpoint(self) -> LLMEndpoint:
        return LLMEndpoint.from_config(self.grading_model)

    def task_profile_endpoint(self) -> LLMEndpoint:
        return LLMEndpoint.from_config(self.task_profile_model)

    def task_profile_prompt_style(self) -> str:
        """任务画像提示词风格：default / sft_v3（微调模型专用）。"""
        return str(self.task_profile_model.get("prompt_style", "default"))

    def routing_policy(self) -> RoutingPolicy:
        return RoutingPolicy(
            routing_mode=self.routing_mode,
            max_cost_per_million_output=self.max_cost_per_million_output,
            max_first_token_latency_ms=self.max_first_token_latency_ms,
            difficulty_tier_map=self.difficulty_tier_map,
            disable_upward_fallback=self.disable_upward_fallback,
        )
