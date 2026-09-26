"""
统一策略驱动评估：按 (数据集 × 难度) 分流 + 真实成本的逐题选模型。

不修改任何既有路由/增强代码，而是复用其组件（import）：
- 缓存任务画像（需求权重 w、难度 level）与硬过滤候选（router.route）
- 数据驱动能力画像 Q'(m,x)（model_profiles_data_driven.json）
- 训练集统计：细粒度期望正确率 acc^(m|d,lvl) 与**实测平均成本** ĉ(m|d,lvl)
- 可选簇先验 cluster_acc(m|cluster(x))（compute_cluster_priors，AvengersPro 风格）

逐题打分：
    P(m|x) = α·Q'(m,x) + β·acc^(m|d,lvl) + γ·cluster_acc(m|x) + δ·acc^(m|d)
    score(m|x) = P(m|x) − λ·ĉ(m|d,lvl)        （ĉ 为实测成本，非标价）
    （acc^ 缺失时回退数据集级先验；权重默认 α=.4 β=.3 γ=.2 δ=.1，和为 1）

与既有单模式（hard / marginal-utility / cluster）互斥式调度不同，
本驱动把所有信号统一到一个打分量纲上，用单一 λ 做预算约束下的
期望正确率最大化（拉格朗日视角），从而在每个数据集 × 难度上
自动做出"是否值得升级贵模型"的逐题决策。

用法：
    python -m adapter.evaluate_strategy_map \
        --test .../test.jsonl --routing-records .../routing_records.jsonl \
        --model-profiles .../model_profiles_data_driven.json \
        --train-path .../train.jsonl \
        --output-dir .../out --weights 0.4,0.3,0.2,0.1 --cost-lambda 0.3
"""

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import find_dotenv, load_dotenv
from loguru import logger

from src import (
    DemoRouter,
    DemoRouterConfig,
    ModelPool,
    RequestFeatures,
    TaskProfile,
    detect_request_features,
    load_profiles,
)
from adapter.evaluate_with_cached_tasks import (
    aggregate_results,
    analyze_routing,
    compute_cluster_priors,
    compute_dataset_prior,
    load_routing_records,
    load_test_items,
)
from adapter.knn_priors import (
    compute_knn_priors,
    load_or_compute_embeddings,
)


def compute_granular_stats(
    train_path: str,
    task_profiles_path: str,
) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    """从训练集计算每个 (dataset, difficulty) 下各模型的经验正确率与实测成本。

    难度来自 train_task_profiles.jsonl（与线上同源的画像模型输出）。
    Returns:
        {dataset: {difficulty: {model_id: {"acc": float, "cost": float, "n": int}}}}
    """
    difficulty_map = {}
    with open(task_profiles_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("status") == "success":
                difficulty_map[f"{rec['dataset']}||{rec['index']}"] = rec["difficulty_level"]

    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    cost = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    cnt = defaultdict(lambda: defaultdict(int))
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            ds = item["dataset"]
            lvl = difficulty_map.get(f"{ds}||{item['index']}", "L2")
            cnt[ds][lvl] += 1
            for model_id, score in item.get("records", {}).items():
                acc[ds][lvl][model_id] += float(score)
                usage = (item.get("usages") or {}).get(model_id) or {}
                cost[ds][lvl][model_id] += float(usage.get("cost") or 0.0)

    stats: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for ds in acc:
        stats[ds] = {}
        for lvl in acc[ds]:
            n = max(cnt[ds][lvl], 1)
            stats[ds][lvl] = {
                m: {"acc": acc[ds][lvl][m] / n, "cost": cost[ds][lvl][m] / n, "n": cnt[ds][lvl]}
                for m in acc[ds][lvl]
            }
    logger.info(
        f"Computed granular stats for {len(stats)} datasets "
        f"(difficulty from {task_profiles_path})"
    )
    return stats


def compute_dataset_costs(train_path: str) -> Dict[str, Dict[str, float]]:
    """每个 (dataset, model) 的实测平均成本（难度缺失时回退用）。"""
    cost = defaultdict(lambda: defaultdict(float))
    cnt = defaultdict(int)
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            ds = item["dataset"]
            cnt[ds] += 1
            for model_id in item.get("records", {}):
                usage = (item.get("usages") or {}).get(model_id) or {}
                cost[ds][model_id] += float(usage.get("cost") or 0.0)
    return {
        ds: {m: cost[ds][m] / max(cnt[ds], 1) for m in cost[ds]} for ds in cost
    }


def run_strategy_map(
    router: DemoRouter,
    test_items: List[Dict],
    cached_records: Dict[str, Dict],
    output_dir: Path,
    granular_stats: Dict,
    dataset_prior: Dict[str, Dict[str, float]],
    dataset_costs: Dict[str, Dict[str, float]],
    weights: Dict[str, float],
    cost_lambda: float,
    tier_bypass: bool = True,
    cluster_priors: Optional[Dict[str, Any]] = None,
    knn_priors: Optional[Dict[str, Any]] = None,
    pins: Optional[Dict[str, str]] = None,
    level_pins: Optional[Dict[str, Dict[str, str]]] = None,
) -> List[Dict]:
    """统一打分逐题选模型，逐条写入 routing_records.jsonl。

    P(m|x) = α·Q' + β·granular + κ·kNN + δ·dataset + γ·cluster
    κ>0 时成本项改用 kNN 逐题实测成本（比 (数据集,难度) 均值更细）。
    pins / level_pins 为训练集证据支持的数据集级 / (数据集,难度) 级
    定向指派，优先级高于打分。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "routing_records.jsonl"
    w_q, w_g, w_k, w_d, w_c = (
        weights["q"],
        weights["granular"],
        weights["knn"],
        weights["dataset"],
        weights["cluster"],
    )
    pins = pins or {}
    level_pins = level_pins or {}
    knn_midx = (
        {m: i for i, m in enumerate(knn_priors["models"])}
        if knn_priors is not None
        else {}
    )

    records: List[Dict] = []
    with open(records_path, "w", encoding="utf-8") as f:
        for test_idx, item in enumerate(test_items):
            key = f"{item['dataset']}||{item['index']}"
            cached = cached_records.get(key)
            if cached is None:
                logger.warning(f"No cached routing record for {key}, skipping")
                continue

            task_profile_dict = cached.get("task_profile")
            if task_profile_dict is None:
                capability_need = cached.get("capability_need")
                if not capability_need:
                    logger.warning(f"No cached task profile for {key}, skipping")
                    continue
                task_profile_dict = {
                    "capability_need": capability_need,
                    "difficulty_level": cached.get("difficulty_level", "L2"),
                    "tool_required": False,
                }
            task = TaskProfile.from_dict(task_profile_dict)

            features_dict = cached.get("request_features")
            if features_dict:
                features = RequestFeatures(**features_dict)
            else:
                features = detect_request_features(item["query"])

            decision = router.route(task, features)
            if tier_bypass:
                candidates = decision.candidates_after_filter
                quality_scores = {
                    m: round(router.quality_score(m, task), 4) for m in candidates
                }
            else:
                candidates = decision.candidates_after_tier
                quality_scores = decision.quality_scores

            ds, lvl = item["dataset"], decision.difficulty_level
            g_stats = granular_stats.get(ds, {}).get(lvl, {})
            ds_prior = dataset_prior.get(ds, {})
            ds_costs = dataset_costs.get(ds, {})
            avg_prior = sum(ds_prior.values()) / len(ds_prior) if ds_prior else 0.0
            avg_cost = sum(ds_costs.values()) / len(ds_costs) if ds_costs else 0.0

            cluster_acc: Dict[str, float] = {}
            if cluster_priors is not None and w_c > 0:
                probs = cluster_priors["test_cluster_probs"][test_idx]
                acc_map = cluster_priors["cluster_model_acc"]
                for m in candidates:
                    expected = 0.0
                    for cid, prob in enumerate(probs):
                        if prob > 0:
                            expected += prob * acc_map.get(int(cid), {}).get(m, 0.0)
                    cluster_acc[m] = expected

            def expected_correctness(model_id: str) -> float:
                q = quality_scores.get(model_id, 0.0)
                g = g_stats.get(model_id, {}).get("acc", ds_prior.get(model_id, avg_prior))
                d = ds_prior.get(model_id, avg_prior)
                c = cluster_acc.get(model_id, g)
                p = w_q * q + w_g * g + w_c * c + w_d * d
                if knn_priors is not None and w_k > 0:
                    ki = knn_midx.get(model_id)
                    p += w_k * (
                        float(knn_priors["acc"][test_idx, ki]) if ki is not None else g
                    )
                return p

            def expected_cost(model_id: str) -> float:
                # κ>0 时优先使用 kNN 逐题实测成本（比 (数据集,难度) 均值更细）
                if knn_priors is not None and w_k > 0:
                    ki = knn_midx.get(model_id)
                    if ki is not None:
                        return float(knn_priors["cost"][test_idx, ki])
                return g_stats.get(model_id, {}).get(
                    "cost", ds_costs.get(model_id, avg_cost)
                )

            scored = {
                m: expected_correctness(m) - cost_lambda * expected_cost(m)
                for m in candidates
            }
            selected = max(scored.items(), key=lambda kv: kv[1])[0] if scored else None

            # 训练证据支持的定向指派：(数据集,难度) 级优先于数据集级
            pinned = level_pins.get(ds, {}).get(lvl) or pins.get(ds)
            if pinned is not None and pinned in candidates:
                selected = pinned

            if selected is None:
                selected = decision.selected_model

            score = 0.0
            cost = 0.0
            if selected is not None:
                score = item["records"].get(selected, 0.0)
                usage = (item.get("usages") or {}).get(selected) or {}
                cost = float(usage.get("cost") or 0.0)

            rec = {
                "dataset": ds,
                "index": item["index"],
                "selected_model": selected,
                "score": score,
                "cost": cost,
                "routing_mode": "strategy_map",
                "difficulty_level": decision.difficulty_level,
                "primary_capability": decision.primary_capability,
                "target_tier": decision.target_tier,
                "matched_tier": decision.matched_tier,
                "fallback_applied": decision.fallback_applied,
                "quality_scores": decision.quality_scores,
                "candidate_count_after_filter": len(decision.candidates_after_filter),
                "candidate_count_after_tier": len(decision.candidates_after_tier),
                "capability_need": decision.task_profile or {},
                "task_profile_source": "cached",
                "cost_weight": cost_lambda,
                "dataset_prior_weight": weights,
            }
            records.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info(f"Wrote {len(records)} strategy-map records to {records_path}")
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Unified strategy-map routing evaluation (per dataset x difficulty, actual costs)"
    )
    parser.add_argument("--test", required=True)
    parser.add_argument("--routing-records", required=True)
    parser.add_argument("--model-profiles", required=True)
    parser.add_argument("--model-pool", default="config/model_pool.yaml")
    parser.add_argument("--config", default="config/demo_router_config.yaml")
    parser.add_argument("--train-path", required=True)
    parser.add_argument(
        "--train-task-profiles",
        default="data/seed42_split0.7/train_task_profiles.jsonl",
        help="训练集任务画像（提供难度标签，用于细粒度统计）",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--weights",
        default="0.3,0.35,0.25,0.1,0",
        help="P 的五个分量权重 α,β,κ,δ,γ（Q',granular,knn,dataset,cluster），和应为 1",
    )
    parser.add_argument("--cost-lambda", type=float, default=0.3, help="实测成本惩罚系数 λ")
    parser.add_argument(
        "--no-tier-bypass", action="store_true", help="候选限定在档位匹配后（默认 bypass）"
    )
    parser.add_argument(
        "--cluster-cache",
        default="",
        help="簇先验 pickle 缓存路径；存在则加载，否则计算后写入（γ>0 时启用簇先验）",
    )
    parser.add_argument("--n-clusters", type=int, default=25)
    parser.add_argument("--knn-k", type=int, default=50, help="kNN 局部先验的邻居数（κ>0 启用）")
    parser.add_argument("--knn-beta", type=float, default=8.0, help="kNN 距离加权 softmax 温度")
    parser.add_argument(
        "--loo",
        action="store_true",
        help="评估集即训练集（--test 与 --train-path 相同）时开启，kNN 排除自身防泄漏",
    )
    parser.add_argument("--train-emb-cache", default="", help="训练集嵌入 .npy 缓存路径")
    parser.add_argument("--eval-emb-cache", default="", help="评估集嵌入 .npy 缓存路径")
    parser.add_argument(
        "--pin",
        nargs="+",
        metavar="DS[:LVL]=MODEL",
        help="训练证据支持的定向指派，如 swe-bench=claude-sonnet-4 tau2:L2=qwen3-235b-a22b-thinking-2507",
    )
    args = parser.parse_args()

    load_dotenv(find_dotenv(usecwd=True))

    parts = [float(x) for x in args.weights.split(",")]
    assert len(parts) == 5, "--weights 需要 5 个逗号分隔数值 (α,β,κ,δ,γ)"
    weights = dict(zip(("q", "granular", "knn", "dataset", "cluster"), parts))
    logger.info(f"weights={weights}, cost_lambda={args.cost_lambda}")

    pins: Dict[str, str] = {}
    level_pins: Dict[str, Dict[str, str]] = {}
    for entry in args.pin or []:
        if "=" not in entry:
            raise ValueError(f"Invalid pin: {entry!r}, expected DS[:LVL]=MODEL")
        lhs, model = entry.split("=", 1)
        if ":" in lhs:
            ds, lvl = lhs.split(":", 1)
            level_pins.setdefault(ds.strip(), {})[lvl.strip().upper()] = model.strip()
        else:
            pins[lhs.strip()] = model.strip()
    if pins or level_pins:
        logger.info(f"pins={pins}, level_pins={level_pins}")

    router_config = DemoRouterConfig.load(args.config)
    model_pool = ModelPool.load(args.model_pool)
    profiles = load_profiles(args.model_profiles)
    test_items = load_test_items(args.test)
    cached_records = load_routing_records(args.routing_records)

    policy = router_config.routing_policy()
    router = DemoRouter(profiles, policy=policy)

    granular_stats = compute_granular_stats(args.train_path, args.train_task_profiles)
    dataset_prior = compute_dataset_prior(args.train_path)
    dataset_costs = compute_dataset_costs(args.train_path)

    knn_priors = None
    if weights["knn"] > 0:
        train_items = load_test_items(args.train_path)
        train_emb = load_or_compute_embeddings(
            train_items, args.train_emb_cache or None
        )
        eval_emb = load_or_compute_embeddings(
            test_items, args.eval_emb_cache or None
        )
        knn_priors = compute_knn_priors(
            train_items,
            train_emb,
            eval_emb,
            k=args.knn_k,
            beta=args.knn_beta,
            loo=args.loo,
        )

    cluster_priors = None
    if weights["cluster"] > 0:
        cache_path = Path(args.cluster_cache) if args.cluster_cache else None
        if cache_path and cache_path.exists():
            with open(cache_path, "rb") as f:
                cluster_priors = pickle.load(f)
            logger.info(f"Loaded cluster priors from {cache_path}")
        else:
            cluster_priors = compute_cluster_priors(
                args.train_path, test_items, n_clusters=args.n_clusters
            )
            if cache_path:
                with open(cache_path, "wb") as f:
                    pickle.dump(cluster_priors, f)
                logger.info(f"Cached cluster priors to {cache_path}")

    output_dir = Path(args.output_dir)
    records = run_strategy_map(
        router,
        test_items,
        cached_records,
        output_dir,
        granular_stats,
        dataset_prior,
        dataset_costs,
        weights,
        args.cost_lambda,
        tier_bypass=not args.no_tier_bypass,
        cluster_priors=cluster_priors,
        knn_priors=knn_priors,
        pins=pins,
        level_pins=level_pins,
    )
    results = aggregate_results(records, test_items)
    analysis = analyze_routing(records)

    eval_output = {
        "config": {
            "weights": weights,
            "cost_lambda": args.cost_lambda,
            "tier_bypass": not args.no_tier_bypass,
            "knn_k": args.knn_k if weights["knn"] > 0 else 0,
            "knn_beta": args.knn_beta if weights["knn"] > 0 else None,
            "loo": args.loo,
            "pins": pins,
            "level_pins": level_pins,
            "model_profiles": args.model_profiles,
            "train_path": args.train_path,
        },
        "results": results,
        "routing_analysis": analysis,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "eval_result.json", "w", encoding="utf-8") as f:
        json.dump(eval_output, f, indent=2, ensure_ascii=False)
    with open(output_dir / "routing_analysis.json", "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    logger.info(f"  Dataset-Avg accuracy: {results['accuracy']}%")
    logger.info(f"  Sample-Avg accuracy: {results['all_sample_avg']}%")
    logger.info(f"  Total cost: ${results['cost_analysis']['total_cost']}")
    logger.info(f"  Model selection: {results['model_selection_counts']}")


if __name__ == "__main__":
    main()
