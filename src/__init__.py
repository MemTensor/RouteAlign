"""
Demo Router：基于能力画像与任务需求匹配的模型路由框架。

独立于任何具体基准测试的通用路由项目：
- 离线阶段：读取即插即用题库 → 多维能力打分 → 模型画像与全局分档
- 线上阶段：请求结构化检测 → 任务画像生成 → 硬过滤 → 档位匹配 → 质量评分 → 选模型
"""

from .capabilities import (
    CAPABILITIES,
    CAPABILITY_FIELDS,
    DIFFICULTY_LEVELS,
    DIFFICULTY_ORDER,
    TIERS,
    format_capability_descriptions,
    format_difficulty_descriptions,
)
from .config import DemoRouterConfig
from .engine import RoutingEngine
from .grading import AnswerGrader
from .llm_client import LLMClient, LLMEndpoint
from .model_pool import ModelPool, ModelPoolEntry
from .offline import (
    generate_answers,
    grade_all,
    load_answers,
    load_grading_results,
    run_offline_pipeline,
)
from .profiling import ProfileBuilder, load_profiles, save_profiles
from .question_bank import QuestionBank
from .request_features import detect_request_features
from .router import DemoRouter, RoutingPolicy
from .schemas import (
    CapabilityGrade,
    CapabilityLevelScore,
    GradingResult,
    ModelProfile,
    QuestionItem,
    RequestFeatures,
    RoutingDecision,
    ServiceAttributes,
    TaskProfile,
)
from .task_profiler import TaskProfiler

__all__ = [
    "CAPABILITIES", "CAPABILITY_FIELDS", "DIFFICULTY_LEVELS", "DIFFICULTY_ORDER",
    "TIERS", "format_capability_descriptions", "format_difficulty_descriptions",
    "DemoRouterConfig", "RoutingEngine", "AnswerGrader",
    "LLMClient", "LLMEndpoint", "ModelPool", "ModelPoolEntry",
    "generate_answers", "grade_all", "load_answers", "load_grading_results",
    "run_offline_pipeline",
    "ProfileBuilder", "load_profiles", "save_profiles",
    "QuestionBank", "detect_request_features",
    "DemoRouter", "RoutingPolicy",
    "CapabilityGrade", "CapabilityLevelScore", "GradingResult", "ModelProfile",
    "QuestionItem", "RequestFeatures", "RoutingDecision", "ServiceAttributes",
    "TaskProfile", "TaskProfiler",
]
