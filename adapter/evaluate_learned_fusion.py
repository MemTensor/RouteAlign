"""
学习式融合路由：用岭回归在训练集上学习"期望正确率"的组合权重。

针对统一打分器的缺陷 D3（手工线性融合无校准、信号可信度不随证据强度调整）：
不再手设 α/β/κ/δ，而是对每对 (train query, model) 提取 7 个特征，
以实测正确率为目标拟合岭回归 P̂(m|x) = w·f(m|x)，路由时按
P̂(m|x) − λ·ĉ(m|x) 逐题选模型。全部特征与权重仅来自训练集
（kNN 特征在训练集上为 LOO，防自匹配泄漏），测试集只做一次性评估。

特征（均为训练集可得的通用信号，无任何数据集级硬编码）：
  f1 = Q'_dd   数据驱动能力画像分（答对归因画像，10 能力 × 难度加权）
  f2 = Q'_jdg  打分模型画像分（gbt-5.4 多维打分画像，独立证据源）
  f3 = g       (数据集, 难度) 细粒度期望正确率
  f4 = knn     kNN 局部期望正确率（逐题）
  f5 = d       数据集级期望正确率
  f6 = knn×disc     kNN × 邻域判别度（局部先验的模型间区分度交互）
  f7 = knn_margin   kNN 邻域边际（与"其他模型中最强者"的差距）
  f8 = is_argmax    是否为该题 kNN 邻域先验的最优模型
  f9 = |knn−g|      kNN 与细粒度先验的分歧度

用法：
    python -m adapter.evaluate_learned_fusion \
        --test .../test.jsonl --routing-records .../routing_records.jsonl \
        --model-profiles .../model_profiles_data_driven.json \
        --judge-profiles .../model_profiles.json \
        --train-path .../train.jsonl --output-dir .../out \
        --knn-k 100 --cost-lambda 0.5 [--loo]
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from dotenv import find_dotenv, load_dotenv
from loguru import logger

from src import (
    CAPABILITY_FIELDS,
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
    compute_dataset_prior,
    load_routing_records,
    load_test_items,
)
from adapter.evaluate_strategy_map import (
    compute_dataset_costs,
    compute_granular_stats,
)
from adapter.knn_priors import (
    compute_knn_priors,
    load_or_compute_embeddings,
)


def build_task_from_cached(cached: Dict) -> Optional[TaskProfile]:
    tp = cached.get("task_profile")
    if tp is None:
        capability_need = cached.get("capability_need")
        if not capability_need:
            return None
        tp = {
            "capability_need": capability_need,
            "difficulty_level": cached.get("difficulty_level", "L2"),
            "tool_required": False,
        }
    return TaskProfile.from_dict(tp)


def extract_features(
    items: List[Dict],
    cached_records: Dict[str, Dict],
    router_dd: DemoRouter,
    router_judge: DemoRouter,
    granular_stats: Dict,
    dataset_prior: Dict[str, Dict[str, float]],
    knn_priors: Dict[str, Any],
) -> "tuple[np.ndarray, List[str], List[Dict]]":
    """对每个 item 的每个候选模型提取特征。

    特征：q_dd / q_judge / granular / knn / dataset / knn×disc /
          knn_margin / is_argmax_knn / |knn−g| / knn_cap（能力条件化局部画像，
          可选，knn_priors 含 cap_acc 时启用）。

    Returns:
        (features (n_pairs, n_feat), model_ids, meta)
        meta[i] = {"item_idx", "model_id", "dataset", "level", "knn_cost", "decision"}
    """
    knn_models = knn_priors["models"]
    knn_midx = {m: i for i, m in enumerate(knn_models)}
    knn_acc = knn_priors["acc"]
    knn_cap_acc = knn_priors.get("cap_acc")  # (n_items, n_caps, n_models) 或 None

    # 邻域证据强度（模型无关，逐题）：有效邻居数 ESS = 1/Σw² 由 kNN 权重隐含，
    # 这里用 acc 矩阵不可还原权重，故以邻域判别度与 ESS 的代理量近似：
    # ess ≈ 1（k 固定时证据强度近似常数）——改用判别度即可。
    # 判别度：该题 kNN 先验在模型间的 max - mean（越大越值得信任 kNN）
    disc = knn_acc.max(axis=1) - knn_acc.mean(axis=1)  # (n_items,)
    disc_mean = float(disc.mean()) or 1.0

    feats: List[List[float]] = []
    model_ids: List[str] = []
    meta: List[Dict] = []

    for item_idx, item in enumerate(items):
        key = f"{item['dataset']}||{item['index']}"
        cached = cached_records.get(key)
        if cached is None:
            continue
        task = build_task_from_cached(cached)
        if task is None:
            continue
        features_dict = cached.get("request_features")
        features = (
            RequestFeatures(**features_dict)
            if features_dict
            else detect_request_features(item["query"])
        )
        decision = router_dd.route(task, features)
        candidates = decision.candidates_after_filter
        if not candidates:
            continue
        q_dd = {m: router_dd.quality_score(m, task) for m in candidates}
        q_judge = {m: router_judge.quality_score(m, task) for m in candidates}

        ds, lvl = item["dataset"], decision.difficulty_level
        g_stats = granular_stats.get(ds, {}).get(lvl, {})
        ds_prior = dataset_prior.get(ds, {})
        avg_prior = sum(ds_prior.values()) / len(ds_prior) if ds_prior else 0.0

        # 该题各候选的 kNN 局部先验及派生量（邻域边际/是否邻域最优/与细粒度先验的分歧）
        knn_vals: Dict[str, float] = {}
        for m in candidates:
            ki = knn_midx.get(m)
            g_m = g_stats.get(m, {}).get("acc", ds_prior.get(m, avg_prior))
            knn_vals[m] = float(knn_acc[item_idx, ki]) if ki is not None else g_m
        top2 = sorted(knn_vals.values(), reverse=True)[:2]
        knn_best = top2[0] if top2 else 0.0
        knn_second = top2[1] if len(top2) > 1 else 0.0
        knn_argmax = (
            max(knn_vals.items(), key=lambda kv: kv[1])[0] if knn_vals else None
        )

        # 能力条件化局部画像：P_cap(m|x) = Σ_c w_{x,c} · A_knn(m, c | x)
        p_cap: Dict[str, float] = {}
        if knn_cap_acc is not None:
            wx = np.zeros(len(CAPABILITY_FIELDS))
            for c, v in task.capability_need.items():
                if c in CAPABILITY_FIELDS:
                    wx[CAPABILITY_FIELDS.index(c)] = v
            wsum = wx.sum()
            if wsum > 0:
                wx = wx / wsum
                p_local = wx @ knn_cap_acc[item_idx]  # (n_models,)
                for m in candidates:
                    ki = knn_midx.get(m)
                    if ki is not None:
                        p_cap[m] = float(p_local[ki])

        for m in candidates:
            g = g_stats.get(m, {}).get("acc", ds_prior.get(m, avg_prior))
            d = ds_prior.get(m, avg_prior)
            knn = knn_vals[m]
            # 邻域边际：该模型的 kNN 先验与"其他模型中最强者"的差距
            margin = knn - (knn_second if m == knn_argmax else knn_best)
            row = [
                q_dd.get(m, 0.0),
                q_judge.get(m, 0.0),
                g,
                knn,
                d,
                knn * (disc[item_idx] / disc_mean),
                margin,
                1.0 if m == knn_argmax else 0.0,
                abs(knn - g),
            ]
            if knn_cap_acc is not None:
                row.append(p_cap.get(m, knn))  # 无覆盖时回退标量局部先验
            feats.append(row)
            model_ids.append(m)
            meta.append({
                "item_idx": item_idx,
                "model_id": m,
                "dataset": ds,
                "level": lvl,
                "decision": decision,
            })

    return np.array(feats, dtype=float), model_ids, meta


def main():
    parser = argparse.ArgumentParser(
        description="Learned-fusion routing: ridge regression on train features"
    )
    parser.add_argument("--test", required=True)
    parser.add_argument("--routing-records", required=True)
    parser.add_argument("--model-profiles", required=True, help="数据驱动画像")
    parser.add_argument(
        "--judge-profiles",
        default="data/seed43_split0.7_gpt5.4/model_profiles.json",
        help="打分模型画像（独立证据源）",
    )
    parser.add_argument("--model-pool", default="config/model_pool.yaml")
    parser.add_argument("--config", default="config/demo_router_config.yaml")
    parser.add_argument("--train-path", required=True)
    parser.add_argument(
        "--train-routing-records",
        default="data/seed42_split0.7/train_task_profiles.jsonl",
        help="训练集任务画像缓存（训练阶段的 cached records）",
    )
    parser.add_argument(
        "--train-task-profiles",
        default="data/seed42_split0.7/train_task_profiles.jsonl",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--knn-k", type=int, default=100)
    parser.add_argument("--knn-beta", type=float, default=8.0)
    parser.add_argument(
        "--knn-cap",
        action="store_true",
        help="启用能力条件化局部 kNN 画像（邻居按各能力维度考察权重加权）",
    )
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--cost-lambda", type=float, default=0.5)
    parser.add_argument(
        "--level-lambda",
        nargs="+",
        metavar="LEVEL=VALUE",
        help="按难度级别指定成本惩罚 λ，例如 L1=0.5 L2=0.4 L3=0.2 L4=0.1；未指定级别回退 --cost-lambda",
    )
    parser.add_argument(
        "--per-difficulty",
        action="store_true",
        help="按难度分别拟合岭回归权重（难度为通用 query 属性）",
    )
    parser.add_argument("--loo", action="store_true", help="评估集即训练集时开启")
    parser.add_argument("--train-emb-cache", default="")
    parser.add_argument("--eval-emb-cache", default="")
    parser.add_argument(
        "--fusion-weights-out",
        default="",
        help="把学到的回归权重写出到该 JSON（可选）",
    )
    args = parser.parse_args()

    level_lambdas: Dict[str, float] = {}
    for entry in args.level_lambda or []:
        if "=" not in entry:
            raise ValueError(f"Invalid level-lambda: {entry!r}")
        lvl, val = entry.split("=", 1)
        level_lambdas[lvl.strip().upper()] = float(val)
    if level_lambdas:
        logger.info(f"level_lambdas={level_lambdas}, fallback lambda={args.cost_lambda}")

    load_dotenv(find_dotenv(usecwd=True))

    router_config = DemoRouterConfig.load(args.config)
    ModelPool.load(args.model_pool)
    profiles_dd = load_profiles(args.model_profiles)
    profiles_judge = load_profiles(args.judge_profiles)
    policy = router_config.routing_policy()
    router_dd = DemoRouter(profiles_dd, policy=policy)
    router_judge = DemoRouter(profiles_judge, policy=policy)

    test_items = load_test_items(args.test)
    cached_records = load_routing_records(args.routing_records)
    train_items = load_test_items(args.train_path)
    train_cached = load_routing_records(args.train_routing_records)

    granular_stats = compute_granular_stats(args.train_path, args.train_task_profiles)
    dataset_prior = compute_dataset_prior(args.train_path)
    dataset_costs = compute_dataset_costs(args.train_path)

    # 训练集 query 的能力需求权重矩阵（与 train_items 行对齐），
    # 用于 kNN 能力条件化局部画像：邻居按各维度考察权重加权
    train_cap_weights = None
    if args.knn_cap:
        cap_idx = {c: i for i, c in enumerate(CAPABILITY_FIELDS)}
        tp_map = {}
        with open(args.train_task_profiles, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("status") == "success":
                    tp_map[f"{rec['dataset']}||{rec['index']}"] = rec.get(
                        "capability_need"
                    ) or {}
        train_cap_weights = np.zeros((len(train_items), len(CAPABILITY_FIELDS)))
        for i, it in enumerate(train_items):
            cn = tp_map.get(f"{it['dataset']}||{it['index']}", {})
            for c, v in cn.items():
                if c in cap_idx:
                    train_cap_weights[i, cap_idx[c]] = float(v)
        logger.info(
            f"train_cap_weights built: {(train_cap_weights.sum(axis=1) > 0).sum()}/{len(train_items)} covered"
        )

    # ---------- kNN 先验（train 用 LOO，eval 不用） ----------
    train_emb = load_or_compute_embeddings(train_items, args.train_emb_cache or None)
    eval_emb = load_or_compute_embeddings(test_items, args.eval_emb_cache or None)
    knn_train = compute_knn_priors(
        train_items, train_emb, train_emb, k=args.knn_k, beta=args.knn_beta,
        loo=True, train_cap_weights=train_cap_weights,
    )
    knn_eval = compute_knn_priors(
        train_items, train_emb, eval_emb, k=args.knn_k, beta=args.knn_beta,
        loo=args.loo, train_cap_weights=train_cap_weights,
    )

    # ---------- 训练阶段：特征 + 岭回归 ----------
    logger.info("Extracting train features...")
    Xtr, tr_models, tr_meta = extract_features(
        train_items, train_cached, router_dd, router_judge,
        granular_stats, dataset_prior, knn_train,
    )
    ytr = np.array([
        float(train_items[m["item_idx"]]["records"].get(m["model_id"], 0.0))
        for m in tr_meta
    ])
    logger.info(f"Train pairs: {Xtr.shape[0]}, features: {Xtr.shape[1]}")

    n_feat = Xtr.shape[1]
    gram = Xtr.T @ Xtr + args.ridge_alpha * np.eye(n_feat)
    w = np.linalg.solve(gram, Xtr.T @ ytr)
    logger.info(f"Learned fusion weights: {np.round(w, 4).tolist()}")

    # 按难度分别拟合（难度是通用 query 属性，非数据集身份）：
    # 让融合权重随难度自适应，如 L1 邻域稠密更信 kNN、L4 更信画像/细粒度
    level_weights: Dict[str, np.ndarray] = {}
    if args.per_difficulty:
        levels = np.array([m["level"] for m in tr_meta])
        for lvl in ("L1", "L2", "L3", "L4"):
            mask = levels == lvl
            if mask.sum() < 500:  # 样本不足回退到全局权重
                level_weights[lvl] = w
                continue
            Xl, yl = Xtr[mask], ytr[mask]
            wl = np.linalg.solve(
                Xl.T @ Xl + args.ridge_alpha * np.eye(n_feat), Xl.T @ yl
            )
            level_weights[lvl] = wl
            logger.info(
                f"  [{lvl}] pairs={mask.sum()}, weights={np.round(wl, 4).tolist()}"
            )

    if args.fusion_weights_out:
        with open(args.fusion_weights_out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "weights": w.tolist(),
                    "level_weights": {k: v.tolist() for k, v in level_weights.items()},
                },
                f,
                indent=2,
            )

    # ---------- 评估阶段：逐题 P̂ − λ·ĉ ----------
    logger.info("Extracting eval features...")
    Xte, te_models, te_meta = extract_features(
        test_items, cached_records, router_dd, router_judge,
        granular_stats, dataset_prior, knn_eval,
    )
    if args.per_difficulty and level_weights:
        pred = np.array([
            Xte[i] @ level_weights.get(te_meta[i]["level"], w)
            for i in range(len(te_meta))
        ])
    else:
        pred = Xte @ w

    knn_models = knn_eval["models"]
    knn_midx = {m: i for i, m in enumerate(knn_models)}
    ds_costs_default = dataset_costs

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # 按 item 分组选择 argmax(P̂ − λ·ĉ)
    by_item: Dict[int, List[int]] = {}
    for i, m in enumerate(te_meta):
        by_item.setdefault(m["item_idx"], []).append(i)

    records: List[Dict] = []
    with open(output_dir / "routing_records.jsonl", "w", encoding="utf-8") as f:
        for item_idx, rows in by_item.items():
            item = test_items[item_idx]
            ds = item["dataset"]
            decision = te_meta[rows[0]]["decision"]
            lvl = decision.difficulty_level
            g_stats = granular_stats.get(ds, {}).get(lvl, {})
            ds_costs = ds_costs_default.get(ds, {})
            avg_cost = sum(ds_costs.values()) / len(ds_costs) if ds_costs else 0.0

            best_m, best_score = None, -float("inf")
            lam = level_lambdas.get(lvl, args.cost_lambda)
            for r in rows:
                m = te_meta[r]["model_id"]
                ki = knn_midx.get(m)
                c = (
                    float(knn_eval["cost"][te_meta[r]["item_idx"], ki])
                    if ki is not None
                    else g_stats.get(m, {}).get("cost", ds_costs.get(m, avg_cost))
                )
                s = pred[r] - lam * c
                if s > best_score:
                    best_score, best_m = s, m

            score = float(item["records"].get(best_m, 0.0)) if best_m else 0.0
            usage = (item.get("usages") or {}).get(best_m) or {}
            cost = float(usage.get("cost") or 0.0)
            rec = {
                "dataset": ds,
                "index": item["index"],
                "selected_model": best_m,
                "score": score,
                "cost": cost,
                "routing_mode": "learned_fusion",
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
                "cost_weight": args.cost_lambda,
                "dataset_prior_weight": 0.0,
            }
            records.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    results = aggregate_results(records, test_items)
    analysis = analyze_routing(records)
    eval_output = {
        "config": {
            "method": "learned_fusion_ridge",
            "features": ["q_dd", "q_judge", "granular", "knn", "dataset", "knn*disc", "knn_margin", "is_argmax_knn", "knn_g_gap", "knn_cap"],
            "fusion_weights": w.tolist(),
            "per_difficulty": args.per_difficulty,
            "ridge_alpha": args.ridge_alpha,
            "cost_lambda": args.cost_lambda,
            "knn_k": args.knn_k,
            "knn_beta": args.knn_beta,
            "model_profiles": args.model_profiles,
            "train_path": args.train_path,
        },
        "results": results,
        "routing_analysis": analysis,
    }
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
