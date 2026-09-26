"""
题库构建：调用强模型为每道抽样题目标注元数据与能力权重，
生成符合 Demo Router 要求的完整题库（QuestionItem JSONL）。

支持断点续跑：已标注的 item_id 会被跳过。

用法：
    python -m adapter.build_question_bank \
        --config config/adapter_config.yaml
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Set

from dotenv import find_dotenv, load_dotenv
from loguru import logger

from src.capabilities import (
    DIFFICULTY_ORDER,
    normalize_capability_key,
    score_to_level,
)

from .config import AdapterConfig
from .llm_client import AdapterLLMClient
from .prompts import build_metadata_messages

# 输入 token 数区间（按字符近似估算，约 3 字符/token）
_TOKEN_BUCKETS = [
    (2000, "0-2k"),
    (8000, "2k-8k"),
    (32000, "8k-32k"),
    (128000, "32k-128k"),
]


def _token_bucket(text: str) -> str:
    approx_tokens = len(text) // 3
    for limit, label in _TOKEN_BUCKETS:
        if approx_tokens < limit:
            return label
    return "128k+"


def _normalize_weights(raw_weights: Dict) -> Dict[str, float]:
    """清洗并归一化能力权重。"""
    weights = {}
    for key, value in (raw_weights or {}).items():
        field = normalize_capability_key(key)
        if not field:
            continue
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        if v > 0:
            weights[field] = weights.get(field, 0.0) + v
    total = sum(weights.values())
    if total > 0:
        weights = {k: round(v / total, 4) for k, v in weights.items()}
    return weights


def _derive_capability_lists(weights: Dict[str, float]) -> (List[str], List[str]):
    """从权重派生主要/次要考察能力：最高权重为主要，其余 >=0.1 为次要。"""
    if not weights:
        return [], []
    ordered = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
    primary = [ordered[0][0]]
    secondary = [k for k, v in ordered[1:] if v >= 0.1]
    return primary, secondary


def annotate_question(
    client: AdapterLLMClient,
    sampled: Dict,
) -> Dict:
    """调用强模型标注一道题目，输出完整 QuestionItem dict。"""
    question = sampled["question"]
    reference_answer = sampled.get("reference_answer", "")

    raw = client.chat_json(build_metadata_messages(question, reference_answer))

    weights = _normalize_weights(raw.get("capability_weights"))
    if not weights:
        logger.warning(
            f"Empty capability weights for {sampled['item_id']}; "
            f"defaulting to instruction_following"
        )
        weights = {"instruction_following": 1.0}
    primary, secondary = _derive_capability_lists(weights)

    # 难度等级与连续难度分的一致性校验：以 difficulty_score 为准做兜底
    level = raw.get("difficulty_level", "")
    try:
        difficulty_score = float(raw.get("difficulty_score", 0.5))
    except (TypeError, ValueError):
        difficulty_score = 0.5
    difficulty_score = max(0.0, min(1.0, difficulty_score))
    if level not in DIFFICULTY_ORDER:
        level = score_to_level(difficulty_score)

    return {
        "item_id": sampled["item_id"],
        "question": question,
        "reference_answer": reference_answer,
        "rubric": str(raw.get("rubric", "")),
        "task_type": str(raw.get("task_type", "")),
        "difficulty_level": level,
        "difficulty_score": difficulty_score,
        "primary_capabilities": primary,
        "secondary_capabilities": secondary,
        "capability_weights": weights,
        "objective_metric": "llm_judge",
        "input_token_bucket": _token_bucket(question),
        "expected_output_length": str(raw.get("expected_output_length", "medium")),
        "tool_required": bool(raw.get("tool_required", False)),
        "multimodal_required": bool(raw.get("multimodal_required", False)),
        "safety_sensitive": bool(raw.get("safety_sensitive", False)),
        "domain": str(raw.get("domain", "")),
        "language": str(raw.get("language", "en")),
        "answer_format_requirement": str(raw.get("answer_format_requirement", "")),
        "bank_type": "core",
        "metadata_version": "v1.0",
        "extra": {"dataset": sampled.get("dataset"), "index": sampled.get("index")},
    }


def _load_done_ids(path: str) -> Set[str]:
    done = set()
    p = Path(path)
    if not p.exists():
        return done
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(json.loads(line)["item_id"])
    return done


def build_question_bank(config: AdapterConfig) -> str:
    """对全部抽样题目执行标注，输出题库 JSONL（追加写入、断点续跑）。"""
    sampled_items = []
    with open(config.sampled_questions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sampled_items.append(json.loads(line))

    done = _load_done_ids(config.question_bank_path)
    pending = [s for s in sampled_items if s["item_id"] not in done]
    logger.info(
        f"Question bank annotation: {len(done)} already done, {len(pending)} pending"
    )
    if not pending:
        return config.question_bank_path

    if not config.metadata_model:
        raise ValueError("adapter config missing 'metadata_model' for annotation")
    client = AdapterLLMClient(config.metadata_model)

    Path(config.question_bank_path).parent.mkdir(parents=True, exist_ok=True)
    with open(config.question_bank_path, "a", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=config.max_workers) as pool:
            futures = {
                pool.submit(annotate_question, client, s): s for s in pending
            }
            completed = 0
            failed = 0
            for future in as_completed(futures):
                sampled = futures[future]
                try:
                    item = future.result()
                except Exception as e:  # noqa: BLE001
                    # 单题标注永久失败不拖垮整批；未写入的题下次续跑会重试
                    failed += 1
                    completed += 1
                    logger.error(
                        f"Annotation failed permanently for {sampled['item_id']}: {e}"
                    )
                    continue
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                f.flush()
                completed += 1
                if completed % 10 == 0 or completed == len(pending):
                    logger.info(
                        f"Annotation progress: {completed}/{len(pending)} "
                        f"(failed: {failed})"
                    )
            if failed:
                logger.warning(
                    f"Annotation finished with {failed}/{len(pending)} permanent "
                    f"failures; rerun the same command to retry them"
                )

    logger.info(f"Question bank written to {config.question_bank_path}")
    return config.question_bank_path


def main():
    parser = argparse.ArgumentParser(description="Build DemoRouter question bank")
    parser.add_argument(
        "--config",
        default="config/adapter_config.yaml",
        help="Path to adapter config YAML",
    )
    args = parser.parse_args()

    load_dotenv(find_dotenv(usecwd=True))
    config = AdapterConfig.load(args.config)
    build_question_bank(config)


if __name__ == "__main__":
    main()
