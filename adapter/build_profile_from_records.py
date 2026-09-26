"""
数据驱动的模型能力画像构建。

针对"模型能力评估与用户需求表征之间的评分偏差"问题：
不再依赖打分模型的自评式多维打分，而是——
1. 每条 train query 的任务画像给出该题的用户需求得分 w_{i,k}
   （由 profile_train_tasks.py 生成，与线上路由使用同一个任务画像模型，
   保证能力归因与需求表征在同一个评分空间）；
2. 若模型在 query i 上实际答对（records 中的实测得分 s_{m,i}），
   就把该题各需求维度得分按 w_{i,k} 归因累加到模型 m；
3. 加权平均得到能力画像得分：
       A'_{m,k,l} = Σ_i w_{i,k}·s_{m,i} / Σ_i w_{i,k}
   其中 l 为 query 的难度等级。得分取值 0-1，即模型在该 (能力, 难度)
   上的经验正确率。

输出 schema 与 model_profiles.json 完全一致，可直接作为
evaluate_with_cached_tasks.py 的 --model-profiles 输入；
路由主干与性能–成本增强策略（marginal_utility / hardness_aware /
capability_aware / cluster_aware）无需任何改动。

用法：
    python -m adapter.build_profile_from_records \
        --train data/seed42_split0.7/train.jsonl \
        --task-profiles data/seed42_split0.7/train_task_profiles.jsonl \
        --model-pool config/model_pool.yaml \
        --output data/seed42_split0.7/model_profiles_data_driven.json
"""

import argparse
import json
from collections import defaultdict
from typing import Dict

from loguru import logger

from src import (
    CAPABILITY_FIELDS,
    DIFFICULTY_ORDER,
    CapabilityLevelScore,
    ModelPool,
    ModelProfile,
    ProfileBuilder,
    ServiceAttributes,
    TaskProfile,
    save_profiles,
)


def load_train_items(path: str) -> Dict[str, Dict]:
    items = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                items[f"{item['dataset']}||{item['index']}"] = item
    logger.info(f"Loaded {len(items)} train items from {path}")
    return items


def load_task_profiles(path: str) -> Dict[str, TaskProfile]:
    """加载 train query 的任务画像，仅保留成功样本，并归一化能力权重。"""
    profiles: Dict[str, TaskProfile] = {}
    n_failed = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"Skipping malformed line in {path}: {line[:80]}")
                continue
            if rec.get("status") != "success":
                n_failed += 1
                continue
            task = TaskProfile.from_dict(rec)
            total = sum(task.capability_need.values())
            if total <= 0:
                n_failed += 1
                continue
            task.capability_need = {
                k: v / total for k, v in task.capability_need.items() if v > 0
            }
            profiles[f"{rec['dataset']}||{rec['index']}"] = task
    logger.info(
        f"Loaded {len(profiles)} task profiles from {path} "
        f"(skipped {n_failed} failed/empty)"
    )
    return profiles


def build_data_driven_profiles(
    train_items: Dict[str, Dict],
    task_profiles: Dict[str, TaskProfile],
    model_pool: ModelPool,
    difficulty_mode: str = "full",
    shrink_m0: float = 8.0,
) -> Dict[str, ModelProfile]:
    """按 A'_{m,k,l} = Σ w·s / Σ w 构建模型画像，并做全局分档。

    difficulty_mode：难度维度的实现策略（profiler 难度分布偏斜，L4 仅 ~1%，
    细粒度 cell 样本稀疏会导致噪声画像）——
      "full":   每 (能力, 难度) 独立估计（原始实现）；
      "shrink": 经验贝叶斯收缩，A'_l = (Σw·s + m0·μ_cap) / (Σw + m0)，
                μ_cap 为该模型该能力跨难度均值，稀疏 cell 自动向能力级均值收缩；
      "merged": 两级难度（L1+L2 / L3+L4 合并估计，写回对应 level）；
      "none":   不按难度区分，每能力一个估计（写回全部 level）。
    """
    # accum[model][cap][level] = [加权得分和, 权重和, 贡献样本数]
    accum = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0])))

    n_used = 0
    for key, item in train_items.items():
        task = task_profiles.get(key)
        if task is None:
            continue
        records = item.get("records") or {}
        if not records:
            continue
        n_used += 1
        level = task.difficulty_level
        for cap, weight in task.capability_need.items():
            if weight <= 0 or cap not in CAPABILITY_FIELDS:
                continue
            for model_id, score in records.items():
                slot = accum[model_id][cap][level]
                slot[0] += weight * float(score)
                slot[1] += weight
                slot[2] += 1
    logger.info(f"Aggregated over {n_used}/{len(train_items)} train items")

    builder = ProfileBuilder()

    # 组织成 assign_tiers 需要的结构，并补齐全能力 × 全难度
    model_ids = sorted(set(accum.keys()) | set(model_pool.model_ids()))

    def combined_slot(model_id: str, cap: str, levels) -> list:
        ws = sum(accum.get(model_id, {}).get(cap, {}).get(l, [0.0, 0.0, 0])[0] for l in levels)
        w = sum(accum.get(model_id, {}).get(cap, {}).get(l, [0.0, 0.0, 0])[1] for l in levels)
        n = sum(accum.get(model_id, {}).get(cap, {}).get(l, [0.0, 0.0, 0])[2] for l in levels)
        return [ws, w, n]

    scores: Dict[str, Dict[str, Dict[str, CapabilityLevelScore]]] = {}
    for model_id in model_ids:
        scores[model_id] = {}
        for cap in CAPABILITY_FIELDS:
            scores[model_id][cap] = {}
            # 能力级（跨难度）均值，作为 shrink 模式的收缩先验
            ws_all, w_all_sum, _ = combined_slot(model_id, cap, DIFFICULTY_ORDER)
            mu = ws_all / w_all_sum if w_all_sum > 0 else None

            for level in DIFFICULTY_ORDER:
                if difficulty_mode == "none":
                    slot = combined_slot(model_id, cap, DIFFICULTY_ORDER)
                elif difficulty_mode == "merged":
                    grp = ("L1", "L2") if level in ("L1", "L2") else ("L3", "L4")
                    slot = combined_slot(model_id, cap, grp)
                else:
                    slot = accum.get(model_id, {}).get(cap, {}).get(level, [0.0, 0.0, 0])

                ws, w, n = slot
                if w <= 0:
                    entry = CapabilityLevelScore(
                        score=None, effective_sample_count=0, confidence="none"
                    )
                    # shrink 模式下空 cell 用能力级均值填充（纯先验，不置信）
                    if difficulty_mode == "shrink" and mu is not None:
                        entry = CapabilityLevelScore(
                            score=round(mu, 4),
                            effective_sample_count=0,
                            confidence="none",
                        )
                else:
                    val = ws / w
                    if difficulty_mode == "shrink" and mu is not None:
                        val = (ws + shrink_m0 * mu) / (w + shrink_m0)
                    entry = CapabilityLevelScore(
                        score=round(val, 4),
                        effective_sample_count=n,
                        confidence=builder._confidence(n),
                    )
                scores[model_id][cap][level] = entry

    tiers = builder.assign_tiers(scores)

    profiles: Dict[str, ModelProfile] = {}
    for model_id in model_ids:
        capability_scores: Dict[str, Dict] = {}
        specialty_tags = []
        for cap in CAPABILITY_FIELDS:
            tier = tiers[cap].get(model_id, "weak")
            entry: Dict = {level: scores[model_id][cap][level] for level in DIFFICULTY_ORDER}
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
            service_attrs = ServiceAttributes()

        profiles[model_id] = ModelProfile(
            model_id=model_id,
            capability_scores=capability_scores,
            specialty_tags=specialty_tags,
            service_attributes=service_attrs,
        )

    logger.info(f"Built data-driven profiles for {len(profiles)} models")
    return profiles


def print_summary(profiles: Dict[str, ModelProfile]) -> None:
    """打印各模型宏观均分与档位分布，便于 sanity check。"""
    header = f"{'model':<38} {'macro_avg':>9} " + " ".join(
        f"{cap[:12]:>12}" for cap in CAPABILITY_FIELDS
    )
    print(header)
    for model_id, profile in sorted(profiles.items()):
        cap_avgs = {}
        all_scores = []
        for cap in CAPABILITY_FIELDS:
            vals = [
                profile.score_at(cap, level)
                for level in DIFFICULTY_ORDER
                if profile.score_at(cap, level) is not None
            ]
            cap_avgs[cap] = sum(vals) / len(vals) if vals else None
            all_scores.extend(vals)
        macro = sum(all_scores) / len(all_scores) if all_scores else 0.0
        row = f"{model_id:<38} {macro:>9.4f} " + " ".join(
            f"{cap_avgs[cap]:>12.4f}" if cap_avgs[cap] is not None else f"{'-':>12}"
            for cap in CAPABILITY_FIELDS
        )
        print(row)


def main():
    parser = argparse.ArgumentParser(
        description="Build data-driven model profiles from actual correctness records"
    )
    parser.add_argument("--train", required=True, help="Path to train.jsonl")
    parser.add_argument(
        "--task-profiles",
        required=True,
        help="Path to train_task_profiles.jsonl (from profile_train_tasks.py)",
    )
    parser.add_argument(
        "--model-pool",
        default="config/model_pool.yaml",
        help="Path to model_pool.yaml",
    )
    parser.add_argument("--output", required=True, help="Output model_profiles JSON path")
    parser.add_argument(
        "--difficulty-mode",
        choices=["full", "shrink", "merged", "none"],
        default="full",
        help="难度维度处理：full=逐难度独立估计；shrink=向能力级均值经验贝叶斯收缩；merged=L1+L2/L3+L4 合并；none=不分难度",
    )
    parser.add_argument(
        "--shrink-m0",
        type=float,
        default=8.0,
        help="shrink 模式的先验强度（等效权重质量）",
    )
    args = parser.parse_args()

    train_items = load_train_items(args.train)
    task_profiles = load_task_profiles(args.task_profiles)
    model_pool = ModelPool.load(args.model_pool)

    profiles = build_data_driven_profiles(
        train_items,
        task_profiles,
        model_pool,
        difficulty_mode=args.difficulty_mode,
        shrink_m0=args.shrink_m0,
    )
    save_profiles(profiles, args.output)
    print_summary(profiles)


if __name__ == "__main__":
    main()
