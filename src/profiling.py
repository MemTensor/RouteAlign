"""
模型画像构建与全局分档（方案 §5-§6）。

- 能力得分聚合：A_{m,k,l} = Σ(w_{i,k} × s_{m,i,k}) / Σ(w_{i,k})
  仅纳入 w_{i,k} > 0 且 relevance > 阈值 且 s >= 0 且 validity=valid 的样本。
- 置信度：按有效样本数与可配置阈值划分 low/medium/high。
- 全局档位：按相对排序划分 专家/强/标准/弱 四档。
- 专长标签：基于档位生成，如 "coding_expert"。
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from .capabilities import CAPABILITY_FIELDS, DIFFICULTY_ORDER, TIERS
from .model_pool import ModelPool
from .question_bank import QuestionBank
from .schemas import (
    CapabilityLevelScore,
    GradingResult,
    ModelProfile,
)


class ProfileBuilder:
    """从打分结果聚合出模型画像，并对全模型池做能力分档。"""

    def __init__(
        self,
        relevance_threshold: float = 0.5,
        confidence_thresholds: Optional[Dict[str, int]] = None,
        expert_ratio: float = 0.25,
        strong_ratio: float = 0.33,
        standard_ratio: float = 0.50,
    ):
        """
        Args:
            relevance_threshold: 打分相关性阈值，超过才纳入能力计算（初始 0.5）
            confidence_thresholds: 样本置信度阈值，如 {"low": 3, "medium": 10, "high": 30}
                有效样本数 < low → none；< medium → low；< high → medium；否则 high
            expert_ratio: 按 L4 得分排序，前 25% 标为专家模型
            strong_ratio: 剔除专家后按 L3 排序，前 33% 标为强模型
            standard_ratio: 剔除专家/强后按 L2 排序，前 50% 标为标准模型
        """
        self.relevance_threshold = relevance_threshold
        self.confidence_thresholds = confidence_thresholds or {
            "low": 3, "medium": 10, "high": 30,
        }
        self.expert_ratio = expert_ratio
        self.strong_ratio = strong_ratio
        self.standard_ratio = standard_ratio

    # ------------------------------------------------------------------
    # 能力得分聚合
    # ------------------------------------------------------------------

    def aggregate_capability_scores(
        self,
        bank: QuestionBank,
        grading_results: List[GradingResult],
    ) -> Dict[str, Dict[str, Dict[str, CapabilityLevelScore]]]:
        """
        计算所有模型的 A_{m,k,l}。

        Returns:
            {model_id: {capability: {level: CapabilityLevelScore}}}
        """
        # accum[model][cap][level] = [weighted_score_sum, weight_sum, sample_count]
        accum = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0])))

        for result in grading_results:
            if result.validity != "valid":
                continue
            try:
                item = bank.get(result.item_id)
            except KeyError:
                logger.warning(f"Grading result references unknown item: {result.item_id}")
                continue
            level = item.difficulty_level
            for cap, grade in result.capability_list.items():
                weight = item.capability_weights.get(cap, 0.0)
                # 纳入条件：w>0 且 relevance>阈值 且 s>=0（validity 已在上方检查）
                if weight <= 0.0:
                    continue
                if grade.relevance <= self.relevance_threshold:
                    continue
                if grade.score < 0:
                    continue
                slot = accum[result.model_id][cap][level]
                slot[0] += weight * grade.score
                slot[1] += weight
                slot[2] += 1

        scores: Dict[str, Dict[str, Dict[str, CapabilityLevelScore]]] = {}
        for model_id, cap_map in accum.items():
            scores[model_id] = {}
            for cap in CAPABILITY_FIELDS:
                scores[model_id][cap] = {}
                for level in DIFFICULTY_ORDER:
                    slot = cap_map.get(cap, {}).get(level)
                    if not slot or slot[1] <= 0:
                        scores[model_id][cap][level] = CapabilityLevelScore(
                            score=None, effective_sample_count=0, confidence="none"
                        )
                        continue
                    scores[model_id][cap][level] = CapabilityLevelScore(
                        score=round(slot[0] / slot[1], 4),
                        effective_sample_count=slot[2],
                        confidence=self._confidence(slot[2]),
                    )
        return scores

    def _confidence(self, sample_count: int) -> str:
        t = self.confidence_thresholds
        if sample_count < t.get("low", 3):
            return "none"
        if sample_count < t.get("medium", 10):
            return "low"
        if sample_count < t.get("high", 30):
            return "medium"
        return "high"

    # ------------------------------------------------------------------
    # 全局档位划分（相对排序）
    # ------------------------------------------------------------------

    def assign_tiers(
        self,
        scores: Dict[str, Dict[str, Dict[str, CapabilityLevelScore]]],
    ) -> Dict[str, Dict[str, str]]:
        """
        对每个能力维度按相对排序划分四档。

        Returns:
            {capability: {model_id: tier}}
        """
        model_ids = list(scores.keys())
        tiers: Dict[str, Dict[str, str]] = {}

        def sort_key(model_id: str, cap: str, level: str) -> float:
            entry = scores[model_id][cap].get(level)
            if entry is None or entry.score is None:
                return float("-inf")
            return entry.score

        for cap in CAPABILITY_FIELDS:
            cap_tiers: Dict[str, str] = {}
            remaining = list(model_ids)

            # 1) 按 L4 得分，前 expert_ratio 标为专家
            remaining.sort(key=lambda m: sort_key(m, cap, "L4"), reverse=True)
            n_expert = max(1, round(len(model_ids) * self.expert_ratio)) if model_ids else 0
            # 无 L4 数据的模型不应仅凭排序进入专家档
            experts = [m for m in remaining[:n_expert] if sort_key(m, cap, "L4") > float("-inf")]
            for m in experts:
                cap_tiers[m] = "expert"
            remaining = [m for m in remaining if m not in cap_tiers]

            # 2) 剔除专家后按 L3 得分，前 strong_ratio 标为强
            remaining.sort(key=lambda m: sort_key(m, cap, "L3"), reverse=True)
            n_strong = max(1, round(len(remaining) * self.strong_ratio)) if remaining else 0
            strongs = [m for m in remaining[:n_strong] if sort_key(m, cap, "L3") > float("-inf")]
            for m in strongs:
                cap_tiers[m] = "strong"
            remaining = [m for m in remaining if m not in cap_tiers]

            # 3) 剔除专家/强后按 L2 得分，前 standard_ratio 标为标准
            remaining.sort(key=lambda m: sort_key(m, cap, "L2"), reverse=True)
            n_standard = max(1, round(len(remaining) * self.standard_ratio)) if remaining else 0
            standards = [m for m in remaining[:n_standard] if sort_key(m, cap, "L2") > float("-inf")]
            for m in standards:
                cap_tiers[m] = "standard"

            # 4) 其余为弱模型
            for m in model_ids:
                cap_tiers.setdefault(m, "weak")

            tiers[cap] = cap_tiers

        return tiers

    # ------------------------------------------------------------------
    # 完整画像构建
    # ------------------------------------------------------------------

    def build_profiles(
        self,
        bank: QuestionBank,
        grading_results: List[GradingResult],
        model_pool: ModelPool,
    ) -> Dict[str, ModelProfile]:
        """
        构建全部模型的完整画像：能力得分 + 档位 + 专长标签 + 服务属性。

        模型池中没有打分结果的模型也会生成画像（全部维度视为无数据、弱档）。
        """
        scores = self.aggregate_capability_scores(bank, grading_results)

        # 模型池中未被打分的模型补空得分
        for model_id in model_pool.model_ids():
            if model_id not in scores:
                logger.warning(f"Model {model_id} has no grading results; profile will be empty")
                scores[model_id] = {
                    cap: {
                        level: CapabilityLevelScore(None, 0, "none")
                        for level in DIFFICULTY_ORDER
                    }
                    for cap in CAPABILITY_FIELDS
                }

        tiers = self.assign_tiers(scores)

        profiles: Dict[str, ModelProfile] = {}
        for model_id, cap_map in scores.items():
            capability_scores: Dict[str, Dict] = {}
            specialty_tags: List[str] = []
            for cap in CAPABILITY_FIELDS:
                tier = tiers[cap].get(model_id, "weak")
                entry: Dict = {level: cap_map[cap][level] for level in DIFFICULTY_ORDER}
                entry["tier"] = tier
                capability_scores[cap] = entry
                if tier in ("expert", "strong"):
                    specialty_tags.append(f"{cap}_{tier}")

            service_attrs = (
                model_pool.service_attributes(model_id)
                if model_id in model_pool
                else None
            )
            if service_attrs is None:
                logger.warning(
                    f"Model {model_id} not in model pool; using default service attributes"
                )
                from .schemas import ServiceAttributes
                service_attrs = ServiceAttributes()

            profiles[model_id] = ModelProfile(
                model_id=model_id,
                capability_scores=capability_scores,
                specialty_tags=specialty_tags,
                service_attributes=service_attrs,
            )

        logger.info(f"Built profiles for {len(profiles)} models")
        return profiles


# ---------------------------------------------------------------------------
# 画像持久化
# ---------------------------------------------------------------------------

def save_profiles(profiles: Dict[str, ModelProfile], path: str):
    out = {model_id: p.to_dict() for model_id, p in profiles.items()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(profiles)} model profiles to {path}")


def load_profiles(path: str) -> Dict[str, ModelProfile]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    profiles = {
        model_id: ModelProfile.from_dict(p) for model_id, p in data.items()
    }
    logger.info(f"Loaded {len(profiles)} model profiles from {path}")
    return profiles
