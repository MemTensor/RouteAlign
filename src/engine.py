"""
线上路由引擎（方案 §2.2）。

将各组件串联为完整的线上流程：
用户输入 → 自动结构化检测 → 任务画像生成 → 硬过滤/档位匹配/质量评分 →
按运行模式选定模型 →（可选）调用最终模型生成回答。
"""

from typing import Any, Dict, List, Optional

from loguru import logger

from .config import DemoRouterConfig
from .llm_client import LLMClient, LLMEndpoint
from .model_pool import ModelPool
from .profiling import load_profiles
from .request_features import detect_request_features
from .router import DemoRouter
from .schemas import RoutingDecision
from .task_profiler import TaskProfiler


class RoutingEngine:
    """Demo Router 线上引擎。"""

    def __init__(self, config: DemoRouterConfig):
        self.config = config
        self.model_pool = ModelPool.load(config.model_pool_path)
        profiles = load_profiles(config.model_profiles_path)
        self.router = DemoRouter(profiles, policy=config.routing_policy())
        self.task_profiler = TaskProfiler(
            LLMClient(config.task_profile_endpoint())
        )
        self._answer_clients: Dict[str, LLMClient] = {}

    def route(
        self,
        query: str,
        context: str = "",
        attachments: Optional[List[Dict[str, Any]]] = None,
        user_tier: Optional[str] = None,
        high_risk: bool = False,
    ) -> RoutingDecision:
        """只做路由决策，不调用最终模型。"""
        features = detect_request_features(
            query, context=context, attachments=attachments, high_risk=high_risk
        )
        task = self.task_profiler.profile(query, features)
        decision = self.router.route(task, features, user_tier=user_tier)
        logger.info(
            f"Routed to {decision.selected_model} "
            f"(difficulty={decision.difficulty_level}, "
            f"primary={decision.primary_capability}, tier={decision.matched_tier})"
        )
        return decision

    def answer(
        self,
        query: str,
        context: str = "",
        attachments: Optional[List[Dict[str, Any]]] = None,
        user_tier: Optional[str] = None,
        high_risk: bool = False,
    ) -> Dict[str, Any]:
        """完整流程：路由决策 + 调用最终选定模型生成回答。"""
        decision = self.route(
            query, context=context, attachments=attachments,
            user_tier=user_tier, high_risk=high_risk,
        )
        if decision.selected_model is None:
            return {"decision": decision.to_dict(), "answer": None,
                    "error": "no candidate model after routing"}

        client = self._get_answer_client(decision.selected_model)
        messages = []
        if context:
            messages.append({"role": "system", "content": context})
        messages.append({"role": "user", "content": query})
        content = client.chat(messages)
        return {"decision": decision.to_dict(), "answer": content, "error": None}

    def _get_answer_client(self, model_id: str) -> LLMClient:
        if model_id not in self._answer_clients:
            entry = self.model_pool.get(model_id)
            if not entry.endpoint:
                raise ValueError(
                    f"Model {model_id} has no endpoint config in model pool; "
                    f"cannot generate answer"
                )
            self._answer_clients[model_id] = LLMClient(
                LLMEndpoint.from_config(entry.endpoint)
            )
        return self._answer_clients[model_id]
