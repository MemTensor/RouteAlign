"""
驱动 RouteAlign 离线建模：多维打分 → 模型画像与全局分档。

题库与回答由适配模块准备（data_prep + build_question_bank），
打分与画像聚合完全由 RouteAlign 项目完成，本脚本只负责串联。

用法：
    python -m adapter.run_offline \
        --config config/adapter_config.yaml
"""

import argparse

from dotenv import find_dotenv, load_dotenv
from loguru import logger

from src import (
    AnswerGrader,
    DemoRouterConfig,
    LLMClient,
    ModelPool,
    ProfileBuilder,
    QuestionBank,
    run_offline_pipeline,
)

from .config import AdapterConfig


def run_offline(adapter_config: AdapterConfig):
    router_config = DemoRouterConfig.load(adapter_config.demo_router_config)

    bank = QuestionBank.load(adapter_config.question_bank_path)
    model_pool = ModelPool.load(router_config.model_pool_path)
    grader = AnswerGrader(
        LLMClient(router_config.grading_endpoint()),
        max_answer_chars=adapter_config.max_answer_chars,
    )
    builder = ProfileBuilder(
        relevance_threshold=router_config.relevance_threshold,
        confidence_thresholds=router_config.confidence_thresholds,
        expert_ratio=router_config.tier_ratios.get("expert", 0.25),
        strong_ratio=router_config.tier_ratios.get("strong", 0.33),
        standard_ratio=router_config.tier_ratios.get("standard", 0.50),
    )

    profiles = run_offline_pipeline(
        bank=bank,
        model_pool=model_pool,
        grader=grader,
        profile_builder=builder,
        grading_output_path=adapter_config.grading_results_path,
        profiles_output_path=adapter_config.model_profiles_path,
        answers_path=adapter_config.answers_path,
        max_workers=adapter_config.max_workers,
    )

    logger.info(f"Offline profiling complete: {len(profiles)} model profiles")
    for model_id, profile in sorted(profiles.items()):
        logger.info(f"  {model_id}: specialty_tags={profile.specialty_tags}")


def main():
    parser = argparse.ArgumentParser(description="Run RouteAlign offline profiling")
    parser.add_argument(
        "--config",
        default="config/adapter_config.yaml",
        help="Path to adapter config YAML",
    )
    args = parser.parse_args()

    load_dotenv(find_dotenv(usecwd=True))
    adapter_config = AdapterConfig.load(args.config)
    run_offline(adapter_config)


if __name__ == "__main__":
    main()
