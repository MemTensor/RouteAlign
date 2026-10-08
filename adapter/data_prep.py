"""
数据准备：results/bench 格式转换、训练/测试划分、分层抽样、回答导出。

- 使用 LLMRouterBench 的 BaselineDataLoader 读取 results/bench 数据；
- 按数据集+prompt 划分训练/测试集（与其他 baseline 一致，防数据泄漏）；
- 从训练集中按数据集分层抽样，形成题库候选题目；
- 从既有基准结果中导出被抽样题目上各模型的回答（复用回答，不重新生成）。

用法：
    python -m adapter.data_prep \
        --config config/adapter_config.yaml
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from loguru import logger

try:
    from baselines.data_loader import BaselineDataLoader
    from baselines.schema import BaselineRecord
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "data_prep requires the LLMRouterBench package (baselines.data_loader, "
        "baselines.schema) to convert raw benchmark records. The prepared "
        "train/test splits used in the paper already ship under data/, so this "
        "script is only needed when rebuilding the split from scratch."
    ) from exc

from .config import AdapterConfig


def _truncate(text: str, limit: int) -> str:
    if text and len(text) > limit:
        return text[:limit] + "\n...（文本过长，已截断）"
    return text


def make_item_id(dataset_id: str, index: int) -> str:
    return f"{dataset_id}-{index:05d}"


def _normalize_answer(raw_output) -> str:
    """将 bench 中的 raw_output 统一转为文本（tau2 等数据集为消息列表）。"""
    if raw_output is None:
        return ""
    if isinstance(raw_output, str):
        return raw_output
    if isinstance(raw_output, (list, dict)):
        return json.dumps(raw_output, ensure_ascii=False, indent=2)
    return str(raw_output)


def _group_by_prompt(
    records: List[BaselineRecord],
) -> Dict[Tuple[str, str], List[BaselineRecord]]:
    groups = defaultdict(list)
    for r in records:
        groups[(r.dataset_id, r.prompt)].append(r)
    return groups


def convert_to_jsonl_items(records: List[BaselineRecord]) -> List[Dict]:
    """
    转换为按题目聚合的 JSONL 条目（与 AvengersPro 格式兼容）：
    {"query", "dataset", "index", "ground_truth", "records": {model: score},
     "usages": {model: {cost, prompt_tokens, completion_tokens}}}
    """
    groups = _group_by_prompt(records)
    items = []
    for (dataset_id, prompt), group in groups.items():
        index = min(r.record_index for r in group)
        scores = {}
        usages = {}
        ground_truth = ""
        for r in group:
            scores[r.model_name] = r.score
            usages[r.model_name] = {
                "cost": r.cost,
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
            }
            if r.ground_truth and r.ground_truth != "None":
                ground_truth = r.ground_truth
        items.append({
            "query": prompt,
            "dataset": dataset_id,
            "index": index,
            "ground_truth": ground_truth,
            "records": scores,
            "usages": usages,
        })
    items.sort(key=lambda x: (x["dataset"], x["index"]))
    return items


def write_jsonl(path: str, items: List[Dict]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(items)} items to {path}")


def stratified_sample(
    train_items: List[Dict],
    per_dataset_samples: int,
    per_dataset_overrides: Dict[str, int],
    seed: int,
) -> List[Dict]:
    """从训练集中按数据集分层随机抽样，形成题库候选题目。"""
    rng = random.Random(seed)
    by_dataset = defaultdict(list)
    for item in train_items:
        by_dataset[item["dataset"]].append(item)

    sampled = []
    for dataset_id in sorted(by_dataset):
        pool = by_dataset[dataset_id]
        n = per_dataset_overrides.get(dataset_id, per_dataset_samples)
        n = min(n, len(pool))
        chosen = rng.sample(pool, n)
        chosen.sort(key=lambda x: x["index"])
        sampled.extend(chosen)
        logger.info(f"Sampled {n}/{len(pool)} questions from dataset {dataset_id}")

    return sampled


def export_answers(
    sampled_items: List[Dict],
    train_records: List[BaselineRecord],
    max_answer_chars: int,
) -> List[Dict]:
    """
    导出被抽样题目上各模型的既有回答（复用 bench 中的 raw_output）。

    Returns:
        [{"item_id", "model_id", "answer"}]
    """
    sampled_keys = {
        (item["dataset"], item["query"]): make_item_id(item["dataset"], item["index"])
        for item in sampled_items
    }

    answers = []
    for r in train_records:
        item_id = sampled_keys.get((r.dataset_id, r.prompt))
        if item_id is None:
            continue
        answers.append({
            "item_id": item_id,
            "model_id": r.model_name,
            "answer": _truncate(_normalize_answer(r.raw_output), max_answer_chars),
        })

    logger.info(f"Exported {len(answers)} answers for {len(sampled_items)} sampled questions")
    return answers


def run_data_prep(config: AdapterConfig) -> Dict[str, str]:
    """执行完整数据准备流程。"""
    loader = BaselineDataLoader(config_path=config.baseline_config)
    all_records = loader.load_all_records()
    logger.info(f"Loaded {len(all_records)} bench records")

    train_records, test_records = loader.split_by_dataset_then_prompt(
        all_records,
        train_ratio=config.train_ratio,
        random_seed=config.seed,
        ood_datasets=config.ood_datasets,
    )

    train_items = convert_to_jsonl_items(train_records)
    test_items = convert_to_jsonl_items(test_records)
    write_jsonl(config.train_path, train_items)
    write_jsonl(config.test_path, test_items)

    sampled = stratified_sample(
        train_items,
        per_dataset_samples=config.per_dataset_samples,
        per_dataset_overrides=config.per_dataset_overrides,
        seed=config.seed,
    )
    # 抽样题目文件：题目 + 参考答案（元数据由 build_question_bank 标注）
    sampled_questions = [
        {
            "item_id": make_item_id(item["dataset"], item["index"]),
            "dataset": item["dataset"],
            "index": item["index"],
            "question": _truncate(item["query"], config.max_question_chars),
            "reference_answer": item["ground_truth"],
        }
        for item in sampled
    ]
    write_jsonl(config.sampled_questions_path, sampled_questions)

    answers = export_answers(sampled, train_records, config.max_answer_chars)
    write_jsonl(config.answers_path, answers)

    return {
        "train": config.train_path,
        "test": config.test_path,
        "sampled_questions": config.sampled_questions_path,
        "answers": config.answers_path,
    }


def main():
    parser = argparse.ArgumentParser(description="RouteAlign adapter data preparation")
    parser.add_argument(
        "--config",
        default="config/adapter_config.yaml",
        help="Path to adapter config YAML",
    )
    args = parser.parse_args()

    config = AdapterConfig.load(args.config)
    outputs = run_data_prep(config)
    logger.info("Data preparation complete:")
    for key, path in outputs.items():
        logger.info(f"  {key}: {path}")


if __name__ == "__main__":
    main()
