"""
测试集路由评估。

对测试集中每道题执行 RouteAlign 线上路由流程（结构化检测 → 任务画像 →
硬过滤 → 档位匹配 → 质量评分 → 选模型），然后离线查表取被选模型在该题上的
既有 score/cost（与其他 baseline 的评估方式一致，不重新调用被路由模型）。

路由记录逐条落盘（routing_records.jsonl），支持断点续跑。

用法：
    python -m adapter.evaluate \
        --config config/adapter_config.yaml
"""

import argparse
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import find_dotenv, load_dotenv
from loguru import logger

from src import (
    DemoRouter,
    DemoRouterConfig,
    LLMClient,
    TaskProfiler,
    detect_request_features,
    load_profiles,
)

from .config import AdapterConfig


def load_test_items(config: AdapterConfig) -> List[Dict]:
    items = []
    with open(config.test_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    logger.info(f"Loaded {len(items)} test items from {config.test_path}")
    return items


def subsample(items: List[Dict], max_per_dataset: Optional[int], seed: int) -> List[Dict]:
    if not max_per_dataset:
        return items
    rng = random.Random(seed)
    by_dataset = defaultdict(list)
    for item in items:
        by_dataset[item["dataset"]].append(item)
    subsampled = []
    for dataset_id in sorted(by_dataset):
        pool = by_dataset[dataset_id]
        n = min(max_per_dataset, len(pool))
        chosen = rng.sample(pool, n)
        chosen.sort(key=lambda x: x["index"])
        subsampled.extend(chosen)
    logger.info(
        f"Subsampled test set: {len(subsampled)}/{len(items)} items "
        f"(max {max_per_dataset} per dataset)"
    )
    return subsampled


def _load_done_records(path: str) -> Dict[str, Dict]:
    done = {}
    p = Path(path)
    if not p.exists():
        return done
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                done[f"{rec['dataset']}||{rec['index']}"] = rec
    return done


def route_test_set(
    adapter_config: AdapterConfig,
    router: DemoRouter,
    profiler: TaskProfiler,
    test_items: List[Dict],
    max_query_chars: int = 20000,
) -> List[Dict]:
    """对测试集执行路由，逐条写入 routing_records.jsonl（断点续跑）。"""
    done = _load_done_records(adapter_config.routing_records_path)
    pending = [
        item for item in test_items
        if f"{item['dataset']}||{item['index']}" not in done
    ]
    logger.info(f"Routing: {len(done)} already done, {len(pending)} pending")

    def run(item: Dict) -> Dict:
        query = item["query"]
        features = detect_request_features(query)
        task = profiler.profile(query, features)
        decision = router.route(task, features)

        selected = decision.selected_model
        score = None
        cost = 0.0
        if selected is not None:
            score = item["records"].get(selected, 0.0)
            usage = (item.get("usages") or {}).get(selected) or {}
            cost = float(usage.get("cost") or 0.0)

        return {
            "dataset": item["dataset"],
            "index": item["index"],
            "selected_model": selected,
            "score": score if score is not None else 0.0,
            "cost": cost,
            "routing_mode": decision.routing_mode,
            "difficulty_level": decision.difficulty_level,
            "primary_capability": decision.primary_capability,
            "target_tier": decision.target_tier,
            "matched_tier": decision.matched_tier,
            "fallback_applied": decision.fallback_applied,
            "quality_scores": decision.quality_scores,
            "candidate_count_after_filter": len(decision.candidates_after_filter),
            "candidate_count_after_tier": len(decision.candidates_after_tier),
            "capability_need": (decision.task_profile or {}).get("capability_need"),
            "task_profile": decision.task_profile,
            "request_features": decision.request_features,
        }

    records = list(done.values())
    if pending:
        Path(adapter_config.routing_records_path).parent.mkdir(parents=True, exist_ok=True)
        with open(adapter_config.routing_records_path, "a", encoding="utf-8") as f:
            with ThreadPoolExecutor(max_workers=adapter_config.max_workers) as pool:
                futures = {pool.submit(run, item): item for item in pending}
                completed = 0
                failed = 0
                for future in as_completed(futures):
                    item = futures[future]
                    try:
                        rec = future.result()
                    except Exception as e:  # noqa: BLE001
                        # 单题路由永久失败不拖垮整批；未写入的题下次续跑会重试
                        failed += 1
                        completed += 1
                        logger.error(
                            f"Routing failed permanently for "
                            f"({item['dataset']}, {item['index']}): {e}"
                        )
                        continue
                    records.append(rec)
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    completed += 1
                    if completed % 20 == 0 or completed == len(pending):
                        logger.info(
                            f"Routing progress: {completed}/{len(pending)} "
                            f"(failed: {failed})"
                        )
                if failed:
                    logger.warning(
                        f"Routing finished with {failed}/{len(pending)} permanent "
                        f"failures; rerun the same command to retry them"
                    )

    # 只保留本次评估范围内的记录
    wanted = {f"{item['dataset']}||{item['index']}" for item in test_items}
    return [r for r in records if f"{r['dataset']}||{r['index']}" in wanted]


def aggregate_results(
    routing_records: List[Dict],
    test_items: List[Dict],
) -> Dict:
    """聚合评估指标：数据集准确率、Sample/Dataset-Avg、成本、模型选择分布、单模型对比。"""
    by_dataset = defaultdict(list)
    for rec in routing_records:
        by_dataset[rec["dataset"]].append(rec)

    dataset_performance = {}
    total_correct = 0.0
    total_cost = 0.0
    for dataset_id in sorted(by_dataset):
        recs = by_dataset[dataset_id]
        correct = sum(r["score"] for r in recs)
        cost = sum(r["cost"] for r in recs)
        dataset_performance[dataset_id] = {
            "correct": round(correct, 4),
            "total": len(recs),
            "accuracy": round(correct / len(recs) * 100, 2) if recs else 0.0,
            "cost": round(cost, 4),
        }
        total_correct += correct
        total_cost += cost

    n = len(routing_records)
    sample_avg = round(total_correct / n * 100, 2) if n else 0.0
    dataset_avg = round(
        sum(d["accuracy"] for d in dataset_performance.values())
        / len(dataset_performance), 2
    ) if dataset_performance else 0.0

    # 模型选择分布
    model_selection = defaultdict(int)
    model_costs = defaultdict(float)
    for rec in routing_records:
        model_selection[rec["selected_model"] or "<none>"] += 1
        model_costs[rec["selected_model"] or "<none>"] += rec["cost"]

    # 难度/档位分布
    difficulty_dist = defaultdict(int)
    tier_dist = defaultdict(int)
    fallback_count = 0
    for rec in routing_records:
        difficulty_dist[rec["difficulty_level"]] += 1
        tier_dist[rec["matched_tier"]] += 1
        if rec.get("fallback_applied"):
            fallback_count += 1

    # 单模型 baseline 对比（同一评估子集上）
    eval_keys = {f"{r['dataset']}||{r['index']}" for r in routing_records}
    single_model = defaultdict(lambda: {"correct": 0.0, "total": 0, "cost": 0.0})
    for item in test_items:
        if f"{item['dataset']}||{item['index']}" not in eval_keys:
            continue
        for model, score in item["records"].items():
            entry = single_model[model]
            entry["correct"] += score
            entry["total"] += 1
            usage = (item.get("usages") or {}).get(model) or {}
            entry["cost"] += float(usage.get("cost") or 0.0)
    baseline_analysis = {
        model: {
            "accuracy": round(v["correct"] / v["total"] * 100, 2) if v["total"] else 0.0,
            "cost": round(v["cost"], 4),
        }
        for model, v in sorted(single_model.items())
    }
    # Oracle 上界：任一模型答对即算对
    oracle_correct = 0.0
    for item in test_items:
        if f"{item['dataset']}||{item['index']}" in eval_keys:
            oracle_correct += max(item["records"].values(), default=0.0)
    oracle_acc = round(oracle_correct / n * 100, 2) if n else 0.0

    return {
        "accuracy": dataset_avg,
        "all_sample_avg": sample_avg,
        "correct_routes": round(total_correct, 4),
        "total_queries": n,
        "dataset_performance": dataset_performance,
        "cost_analysis": {
            "total_cost": round(total_cost, 4),
            "avg_cost_per_query": round(total_cost / n, 6) if n else 0.0,
            "model_costs": {k: round(v, 4) for k, v in sorted(model_costs.items())},
        },
        "model_selection_counts": dict(sorted(model_selection.items())),
        "difficulty_distribution": dict(sorted(difficulty_dist.items())),
        "matched_tier_distribution": dict(sorted(tier_dist.items())),
        "fallback_count": fallback_count,
        "oracle_sample_avg": oracle_acc,
        "baseline_analysis": baseline_analysis,
    }


def run_evaluation(adapter_config: AdapterConfig):
    router_config = DemoRouterConfig.load(adapter_config.demo_router_config)

    profiles = load_profiles(adapter_config.model_profiles_path)
    router = DemoRouter(profiles, policy=router_config.routing_policy())
    profiler = TaskProfiler(
        LLMClient(router_config.task_profile_endpoint()),
        prompt_style=router_config.task_profile_prompt_style(),
    )

    test_items = load_test_items(adapter_config)
    test_items = subsample(
        test_items, adapter_config.eval_max_samples_per_dataset, adapter_config.seed
    )

    routing_records = route_test_set(adapter_config, router, profiler, test_items)
    results = aggregate_results(routing_records, test_items)

    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "adapter_output_dir": adapter_config.output_dir,
            "demo_router_config": adapter_config.demo_router_config,
            "seed": adapter_config.seed,
            "train_ratio": adapter_config.train_ratio,
            "eval_max_samples_per_dataset": adapter_config.eval_max_samples_per_dataset,
            "routing_mode": router_config.routing_mode,
        },
        "results": results,
    }
    Path(adapter_config.eval_result_path).parent.mkdir(parents=True, exist_ok=True)
    with open(adapter_config.eval_result_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"Evaluation complete. Results written to {adapter_config.eval_result_path}")
    logger.info(f"  Dataset-Avg accuracy: {results['accuracy']}%")
    logger.info(f"  Sample-Avg accuracy: {results['all_sample_avg']}%")
    logger.info(f"  Oracle Sample-Avg: {results['oracle_sample_avg']}%")
    logger.info(f"  Total cost: ${results['cost_analysis']['total_cost']}")
    logger.info(f"  Model selection: {results['model_selection_counts']}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Evaluate RouteAlign on the test set")
    parser.add_argument(
        "--config",
        default="config/adapter_config.yaml",
        help="Path to adapter config YAML",
    )
    args = parser.parse_args()

    load_dotenv(find_dotenv(usecwd=True))
    adapter_config = AdapterConfig.load(args.config)
    run_evaluation(adapter_config)


if __name__ == "__main__":
    main()
