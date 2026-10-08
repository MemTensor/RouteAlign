"""
RouteAlign 核心数据结构定义。

涵盖：题库题目、多维打分结果、模型画像、任务画像、请求结构化特征、路由决策。
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from .capabilities import CAPABILITY_FIELDS, DIFFICULTY_ORDER, normalize_capability_key


@dataclass
class QuestionItem:
    """动态评测题库中的一道题（含完整结构化元数据与能力权重）。"""

    item_id: str
    question: str
    reference_answer: str = ""
    rubric: str = ""

    task_type: str = ""
    difficulty_level: str = "L2"
    difficulty_score: float = 0.5
    primary_capabilities: List[str] = field(default_factory=list)
    secondary_capabilities: List[str] = field(default_factory=list)
    # w_{i,k}：题目对每个能力维度的考察权重（由强模型标注）
    capability_weights: Dict[str, float] = field(default_factory=dict)

    objective_metric: str = "llm_judge"
    input_token_bucket: str = ""
    expected_output_length: str = "medium"
    tool_required: bool = False
    multimodal_required: bool = False
    safety_sensitive: bool = False
    domain: str = ""
    language: str = "en"
    answer_format_requirement: str = ""
    bank_type: str = "core"  # core / dynamic / hidden
    metadata_version: str = "v1.0"
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.difficulty_level not in DIFFICULTY_ORDER:
            raise ValueError(
                f"Invalid difficulty_level '{self.difficulty_level}' for item {self.item_id}"
            )
        # 只保留合法能力维度的权重
        self.capability_weights = {
            k: float(v) for k, v in (self.capability_weights or {}).items()
            if k in CAPABILITY_FIELDS
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuestionItem":
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in data.items() if k in known}
        extra = {k: v for k, v in data.items() if k not in known}
        if extra:
            kwargs.setdefault("extra", {}).update(extra)
        return cls(**kwargs)


@dataclass
class CapabilityGrade:
    """打分模型对某一能力维度的判定：相关性 + 表现得分。"""

    relevance: float
    score: float


@dataclass
class GradingResult:
    """打分模型对某个模型在某道题上的多维打分输出。"""

    item_id: str
    model_id: str
    validity: str  # "valid" / "invalid"
    capability_list: Dict[str, CapabilityGrade]
    brief_rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "model_id": self.model_id,
            "validity": self.validity,
            "capability_list": {
                k: {"relevance": g.relevance, "scores": g.score}
                for k, g in self.capability_list.items()
            },
            "brief_rationale": self.brief_rationale,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GradingResult":
        caps = {}
        for k, v in (data.get("capability_list") or {}).items():
            k = normalize_capability_key(k)
            if not k:
                continue
            caps[k] = CapabilityGrade(
                relevance=float(v.get("relevance", 0.0)),
                score=float(v.get("scores", v.get("score", -1))),
            )
        return cls(
            item_id=data["item_id"],
            model_id=data["model_id"],
            validity=str(data.get("validity", "invalid")),
            capability_list=caps,
            brief_rationale=str(data.get("brief_rationale", "")),
        )


@dataclass
class CapabilityLevelScore:
    """模型在某能力、某难度上的聚合得分。"""

    score: Optional[float]
    effective_sample_count: int
    confidence: str  # low / medium / high / none


@dataclass
class ServiceAttributes:
    """模型服务属性标签（硬指标），由模型池配置文件提供。"""

    cost: Dict[str, float] = field(default_factory=dict)
    latency: Dict[str, float] = field(default_factory=dict)
    context_window: int = 128000
    tool_supported: bool = True
    multimodal_supported: List[str] = field(default_factory=list)
    sensitive_data_allowed: bool = True
    available: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ServiceAttributes":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class ModelProfile:
    """完整模型画像：分能力分难度得分 + 档位 + 专长标签 + 服务属性。"""

    model_id: str
    # capability_scores[cap] = {"L1": CapabilityLevelScore, ..., "tier": str}
    capability_scores: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    specialty_tags: List[str] = field(default_factory=list)
    service_attributes: ServiceAttributes = field(default_factory=ServiceAttributes)

    def tier_of(self, capability: str) -> str:
        cap = self.capability_scores.get(capability) or {}
        return cap.get("tier", "weak")

    def score_at(self, capability: str, level: str) -> Optional[float]:
        """取模型在指定能力、指定难度上的得分；无数据返回 None。"""
        cap = self.capability_scores.get(capability) or {}
        entry = cap.get(level)
        if entry is None:
            return None
        if isinstance(entry, CapabilityLevelScore):
            return entry.score
        return entry.get("score")

    def score_at_with_fallback(self, capability: str, level: str) -> Optional[float]:
        """
        取指定难度得分；该难度无有效样本时，回退到最邻近的有数据难度
        （距离相同优先更高难度，避免高估低难度表现）。
        """
        order = DIFFICULTY_ORDER
        target = order.index(level)
        candidates = sorted(
            range(len(order)),
            key=lambda i: (abs(i - target), -i),
        )
        for i in candidates:
            score = self.score_at(capability, order[i])
            if score is not None:
                return score
        return None

    def to_dict(self) -> Dict[str, Any]:
        cap_scores = {}
        for cap, levels in self.capability_scores.items():
            cap_scores[cap] = {}
            for key, val in levels.items():
                if isinstance(val, CapabilityLevelScore):
                    cap_scores[cap][key] = asdict(val)
                else:
                    cap_scores[cap][key] = val
        return {
            "model_id": self.model_id,
            "capability_scores": cap_scores,
            "specialty_tags": self.specialty_tags,
            "service_attributes": self.service_attributes.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelProfile":
        cap_scores = {}
        for cap, levels in (data.get("capability_scores") or {}).items():
            cap_scores[cap] = {}
            for key, val in levels.items():
                if key in DIFFICULTY_ORDER and isinstance(val, dict):
                    cap_scores[cap][key] = CapabilityLevelScore(
                        score=val.get("score"),
                        effective_sample_count=int(val.get("effective_sample_count", 0)),
                        confidence=str(val.get("confidence", "none")),
                    )
                else:
                    cap_scores[cap][key] = val
        return cls(
            model_id=data["model_id"],
            capability_scores=cap_scores,
            specialty_tags=list(data.get("specialty_tags") or []),
            service_attributes=ServiceAttributes.from_dict(
                data.get("service_attributes") or {}
            ),
        )


@dataclass
class TaskProfile:
    """任务画像生成模型输出：能力需求权重、难度等级、工具需求。"""

    capability_need: Dict[str, float]
    difficulty_level: str
    tool_required: bool = False

    def primary_capability(self) -> str:
        """任务画像中权重最高的能力维度（主能力维度优先原则）。"""
        if not self.capability_need:
            return "instruction_following"
        return max(self.capability_need.items(), key=lambda kv: kv[1])[0]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskProfile":
        need = {}
        for k, v in (data.get("capability_need") or {}).items():
            field_name = normalize_capability_key(k)
            if not field_name:
                continue
            try:
                need[field_name] = need.get(field_name, 0.0) + float(v)
            except (TypeError, ValueError):
                continue
        level = data.get("difficulty_level", "L2")
        if level not in DIFFICULTY_ORDER:
            level = "L2"
        return cls(
            capability_need=need,
            difficulty_level=level,
            tool_required=bool(data.get("tool_required", False)),
        )


@dataclass
class RequestFeatures:
    """系统自动检测的请求结构化信息（代码计算，不经过 LLM）。"""

    input_tokens: int = 0
    context_tokens: int = 0
    file_count: int = 0
    file_types: List[str] = field(default_factory=list)
    contains_image: bool = False
    contains_audio: bool = False
    contains_video: bool = False
    # 数据隐私/敏感数据检测暂不实现，恒为 False
    contains_sensitive_data: bool = False
    high_risk: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RoutingDecision:
    """一次路由决策的完整记录。"""

    selected_model: Optional[str]
    routing_mode: str
    difficulty_level: str
    primary_capability: str
    target_tier: str
    matched_tier: str
    candidates_after_filter: List[str]
    candidates_after_tier: List[str]
    quality_scores: Dict[str, float]
    filter_reasons: Dict[str, str] = field(default_factory=dict)
    task_profile: Optional[Dict[str, Any]] = None
    request_features: Optional[Dict[str, Any]] = None
    fallback_applied: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
