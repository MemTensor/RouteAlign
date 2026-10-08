"""
适配模块配置加载。

适配模块有独立于 RouteAlign 的配置文件（YAML），
管理数据来源、划分参数、题库抽样规模、标注 LLM、评估参数等。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class AdapterConfig:
    """适配模块全局配置。"""

    # LLMRouterBench 数据来源
    baseline_config: str = "config/baseline_config_performance_cost.yaml"

    # 训练/测试划分
    seed: int = 42
    train_ratio: float = 0.7
    ood_datasets: List[str] = field(default_factory=list)

    # 输出目录（题库、回答、划分数据、画像、评估结果均在其下）
    output_dir: str = "data/seed42_split0.7"

    # 题库分层抽样规模（按数据集分层）
    per_dataset_samples: int = 10
    per_dataset_overrides: Dict[str, int] = field(default_factory=dict)
    # 题目/回答文本长度上限（超长截断，避免超出标注与打分模型上下文）
    max_question_chars: int = 30000
    max_answer_chars: int = 30000

    # 强模型：题库元数据与能力权重标注
    metadata_model: Dict[str, Any] = field(default_factory=dict)

    # RouteAlign 侧配置文件（驱动离线画像与线上路由）
    demo_router_config: str = "config/demo_router_config.yaml"

    # 评估参数
    eval_max_samples_per_dataset: Optional[int] = None  # None = 全量测试集
    # 评估标签：非空时路由记录与评估结果文件名带后缀
    # （routing_records_{tag}.jsonl / eval_result_{tag}.json），
    # 用于在同一数据目录下并行评估多个任务画像模型而互不覆盖。
    eval_tag: Optional[str] = None

    # 并发
    max_workers: int = 8

    @classmethod
    def load(cls, path: str) -> "AdapterConfig":
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Adapter config not found: {path}")
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        cfg = data.get("adapter", data)

        qb = cfg.get("question_bank") or {}
        evaluation = cfg.get("evaluation") or {}

        return cls(
            baseline_config=cfg.get(
                "baseline_config", "config/baseline_config_performance_cost.yaml"
            ),
            seed=int(cfg.get("seed", 42)),
            train_ratio=float(cfg.get("train_ratio", 0.7)),
            ood_datasets=list(cfg.get("ood_datasets") or []),
            output_dir=cfg.get(
                "output_dir", "data/seed42_split0.7"
            ),
            per_dataset_samples=int(qb.get("per_dataset_samples", 10)),
            per_dataset_overrides=dict(qb.get("per_dataset_overrides") or {}),
            max_question_chars=int(qb.get("max_question_chars", 30000)),
            max_answer_chars=int(qb.get("max_answer_chars", 30000)),
            metadata_model=cfg.get("metadata_model") or {},
            demo_router_config=cfg.get(
                "demo_router_config",
                "config/demo_router_config.yaml",
            ),
            eval_max_samples_per_dataset=evaluation.get("max_samples_per_dataset"),
            eval_tag=evaluation.get("eval_tag") or None,
            max_workers=int(cfg.get("max_workers", 8)),
        )

    # 输出路径约定
    @property
    def train_path(self) -> str:
        return str(Path(self.output_dir) / "train.jsonl")

    @property
    def test_path(self) -> str:
        return str(Path(self.output_dir) / "test.jsonl")

    @property
    def sampled_questions_path(self) -> str:
        return str(Path(self.output_dir) / "sampled_questions.jsonl")

    @property
    def question_bank_path(self) -> str:
        return str(Path(self.output_dir) / "question_bank.jsonl")

    @property
    def answers_path(self) -> str:
        return str(Path(self.output_dir) / "answers.jsonl")

    @property
    def grading_results_path(self) -> str:
        return str(Path(self.output_dir) / "grading_results.jsonl")

    @property
    def model_profiles_path(self) -> str:
        return str(Path(self.output_dir) / "model_profiles.json")

    @property
    def routing_records_path(self) -> str:
        name = (
            f"routing_records_{self.eval_tag}.jsonl"
            if self.eval_tag
            else "routing_records.jsonl"
        )
        return str(Path(self.output_dir) / name)

    @property
    def eval_result_path(self) -> str:
        name = (
            f"eval_result_{self.eval_tag}.json" if self.eval_tag else "eval_result.json"
        )
        return str(Path(self.output_dir) / name)
