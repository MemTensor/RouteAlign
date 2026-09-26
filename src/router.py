"""
路由决策函数（方案 §8-§9）。

两阶段路由：硬条件过滤 + 档位匹配 + 质量匹配评分。
运行模式：best_performance（已实现）、cost_first / speed_first（预留接口，暂不实现）。
"""

from typing import Dict, List, Optional, Tuple

from loguru import logger

from .capabilities import DEFAULT_DIFFICULTY_TIER_MAP, TIERS
from .schemas import (
    ModelProfile,
    RequestFeatures,
    RoutingDecision,
    TaskProfile,
)

ROUTING_MODES = ("best_performance", "cost_first", "speed_first")


class RoutingPolicy:
    """路由策略参数。"""

    def __init__(
        self,
        routing_mode: str = "best_performance",
        max_cost_per_million_output: Optional[float] = None,
        max_first_token_latency_ms: Optional[float] = None,
        difficulty_tier_map: Optional[Dict[str, str]] = None,
        disable_upward_fallback: bool = False,
    ):
        if routing_mode not in ROUTING_MODES:
            raise ValueError(f"Unknown routing mode: {routing_mode}")
        self.routing_mode = routing_mode
        self.max_cost_per_million_output = max_cost_per_million_output
        self.max_first_token_latency_ms = max_first_token_latency_ms
        self.difficulty_tier_map = difficulty_tier_map or dict(DEFAULT_DIFFICULTY_TIER_MAP)
        self.disable_upward_fallback = disable_upward_fallback


class DemoRouter:
    """基于能力画像与任务需求匹配的模型路由器。"""

    def __init__(
        self,
        profiles: Dict[str, ModelProfile],
        policy: Optional[RoutingPolicy] = None,
    ):
        self.profiles = profiles
        self.policy = policy or RoutingPolicy()

    # ------------------------------------------------------------------
    # 硬过滤
    # ------------------------------------------------------------------

    def hard_filter(
        self,
        task: TaskProfile,
        features: RequestFeatures,
    ) -> Tuple[List[str], Dict[str, str]]:
        """
        剔除不满足硬性服务约束的模型。

        Returns:
            (通过过滤的模型列表, {被剔除模型: 剔除原因})
        """
        passed: List[str] = []
        reasons: Dict[str, str] = {}
        total_tokens = features.input_tokens + features.context_tokens

        for model_id, profile in self.profiles.items():
            attrs = profile.service_attributes

            if not attrs.available:
                reasons[model_id] = "model unavailable"
                continue
            if attrs.context_window < total_tokens:
                reasons[model_id] = (
                    f"context window {attrs.context_window} < input {total_tokens}"
                )
                continue
            if task.tool_required and not attrs.tool_supported:
                reasons[model_id] = "tool required but not supported"
                continue
            if features.contains_image and "vision" not in attrs.multimodal_supported:
                reasons[model_id] = "image input but vision not supported"
                continue
            if features.contains_audio and "audio" not in attrs.multimodal_supported:
                reasons[model_id] = "audio input but audio not supported"
                continue
            if features.contains_video and "video" not in attrs.multimodal_supported:
                reasons[model_id] = "video input but video not supported"
                continue
            if features.contains_sensitive_data and not attrs.sensitive_data_allowed:
                reasons[model_id] = "sensitive data not allowed"
                continue
            if self.policy.max_cost_per_million_output is not None:
                output_price = attrs.cost.get("output_price")
                if output_price is not None and output_price > self.policy.max_cost_per_million_output:
                    reasons[model_id] = (
                        f"output price {output_price} exceeds cap "
                        f"{self.policy.max_cost_per_million_output}"
                    )
                    continue
            if self.policy.max_first_token_latency_ms is not None:
                latency = attrs.latency.get("avg_first_token_latency_ms")
                if latency is not None and latency > self.policy.max_first_token_latency_ms:
                    reasons[model_id] = (
                        f"first token latency {latency}ms exceeds cap "
                        f"{self.policy.max_first_token_latency_ms}ms"
                    )
                    continue

            passed.append(model_id)

        return passed, reasons

    # ------------------------------------------------------------------
    # 档位匹配
    # ------------------------------------------------------------------

    def match_tier(
        self,
        candidates: List[str],
        task: TaskProfile,
        features: RequestFeatures,
        user_tier: Optional[str] = None,
    ) -> Tuple[List[str], str, str, bool]:
        """
        按任务难度匹配模型档位（主能力维度优先原则）。

        Args:
            candidates: 硬过滤后的候选模型
            task: 任务画像
            features: 请求特征（高风险任务不允许降档）
            user_tier: 用户指定档位（指定后只在该档位中选择，不回退）

        Returns:
            (匹配档位后的候选模型, 目标档位, 实际使用档位, 是否发生了回退)
        """
        primary_cap = task.primary_capability()

        if user_tier is not None:
            if user_tier not in TIERS:
                raise ValueError(f"Unknown tier: {user_tier}")
            target_tier = user_tier
        else:
            target_tier = self.policy.difficulty_tier_map.get(
                task.difficulty_level, "standard"
            )

        def models_in_tier(tier: str) -> List[str]:
            return [
                m for m in candidates
                if self.profiles[m].tier_of(primary_cap) == tier
            ]

        matched = models_in_tier(target_tier)
        if matched or user_tier is not None:
            # 用户指定档位时不回退：该档位为空即路由失败
            return matched, target_tier, target_tier, False

        # 回退策略：优先向上升级；高风险任务不允许降档
        target_idx = TIERS.index(target_tier)
        fallback_order: List[int] = []
        if not self.policy.disable_upward_fallback:
            fallback_order += list(range(target_idx + 1, len(TIERS)))  # 向上
        if not features.high_risk:
            fallback_order += list(range(target_idx - 1, -1, -1))  # 向下

        for idx in fallback_order:
            matched = models_in_tier(TIERS[idx])
            if matched:
                logger.info(
                    f"Tier fallback: {target_tier} -> {TIERS[idx]} "
                    f"(primary capability: {primary_cap})"
                )
                return matched, target_tier, TIERS[idx], True

        return [], target_tier, target_tier, False

    # ------------------------------------------------------------------
    # 质量匹配评分
    # ------------------------------------------------------------------

    def quality_score(self, model_id: str, task: TaskProfile) -> float:
        """Q(m, x) = Σ_k R_x[k] × A_{m,k,d}"""
        profile = self.profiles[model_id]
        level = task.difficulty_level
        total = 0.0
        for cap, weight in task.capability_need.items():
            if weight <= 0:
                continue
            score = profile.score_at_with_fallback(cap, level)
            if score is None:
                continue  # 全难度均无数据的能力不计入
            total += weight * score
        return total

    # ------------------------------------------------------------------
    # 完整路由
    # ------------------------------------------------------------------

    def route(
        self,
        task: TaskProfile,
        features: RequestFeatures,
        user_tier: Optional[str] = None,
    ) -> RoutingDecision:
        """
        执行完整路由决策：硬过滤 → 档位匹配 → 质量匹配评分 → 按运行模式选择。
        """
        filtered, filter_reasons = self.hard_filter(task, features)
        tier_candidates, target_tier, matched_tier, fallback = self.match_tier(
            filtered, task, features, user_tier=user_tier
        )

        quality_scores = {
            m: round(self.quality_score(m, task), 4) for m in tier_candidates
        }

        selected = self._select_by_mode(quality_scores)

        decision = RoutingDecision(
            selected_model=selected,
            routing_mode=self.policy.routing_mode,
            difficulty_level=task.difficulty_level,
            primary_capability=task.primary_capability(),
            target_tier=target_tier,
            matched_tier=matched_tier,
            candidates_after_filter=filtered,
            candidates_after_tier=tier_candidates,
            quality_scores=quality_scores,
            filter_reasons=filter_reasons,
            task_profile=task.to_dict(),
            request_features=features.to_dict(),
            fallback_applied=fallback,
        )

        if selected is None:
            logger.warning(
                f"Routing failed: no candidate model "
                f"(after filter: {len(filtered)}, after tier: {len(tier_candidates)})"
            )
        return decision

    def _select_by_mode(self, quality_scores: Dict[str, float]) -> Optional[str]:
        if not quality_scores:
            return None
        if self.policy.routing_mode == "best_performance":
            return max(quality_scores.items(), key=lambda kv: kv[1])[0]
        if self.policy.routing_mode == "cost_first":
            raise NotImplementedError("cost_first mode is reserved but not implemented")
        if self.policy.routing_mode == "speed_first":
            raise NotImplementedError("speed_first mode is reserved but not implemented")
        raise ValueError(f"Unknown routing mode: {self.policy.routing_mode}")
