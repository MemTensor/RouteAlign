"""
复用已缓存任务画像重新评测路由，并输出详细路由诊断分析。

当模型画像改变时，路由决策会改变，但任务画像不必重新调用 LLM。
本脚本从旧的 routing_records.jsonl 中读取 task_profile 与 request_features，
加载新的 model_profiles.json，重新执行 router.route()，并输出新的评估结果到新目录。

额外支持：
- 命令行覆盖 difficulty_tier_map，快速实验不同档位映射策略
- 命令行关闭向上回退，防止简单问题被过度升级到强模型
- 成本感知路由：在档位内按 quality_score - cost_weight * cost 选择模型
- 网格搜索：自动扫描多组策略，输出 Pareto 前沿
- 基线对比：与 LLMSemanticRouter 等外部 eval_result.json 对比，输出差距分析
- 输出 routing_analysis.json：包含难度/档位分布、过度路由检测、数据集级诊断

用法示例：
    python -m adapter.evaluate_with_cached_tasks \
        --test data/seed42_split0.7/test.jsonl \
        --routing-records data/seed42_split0.7/routing_records.jsonl \
        --model-profiles data/seed43_split0.7_gpt5.4/model_profiles.json \
        --model-pool config/model_pool.yaml \
        --config config/demo_router_config.yaml \
        --output-dir data/seed42_reroute_seed43_gpt5.4

    # 实验更保守的档位映射：L1/L2 都用 weak，L3 用 standard，L4 用 strong
    python -m adapter.evaluate_with_cached_tasks ... \
        --difficulty-tier-map L1=weak L2=weak L3=standard L4=strong

    # 禁止向上回退（避免简单问题被升级到强模型）
    python -m adapter.evaluate_with_cached_tasks ... \
        --disable-upward-fallback

    # 成本感知路由（cost_weight 越大越偏好便宜模型）
    python -m adapter.evaluate_with_cached_tasks ... \
        --cost-weight 2.0

    # 策略网格搜索，并与 LLMSemanticRouter 对比
    python -m adapter.evaluate_with_cached_tasks ... \
        --grid-search \
        --baseline-eval-result baselines/LLMSemanticRouter/data/seed43_split0.7/eval_result.json
"""

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from dotenv import find_dotenv, load_dotenv
from loguru import logger

from src import (
    DemoRouter,
    DemoRouterConfig,
    ModelPool,
    RequestFeatures,
    RoutingPolicy,
    TaskProfile,
    detect_request_features,
    load_profiles,
)


TIERS_ORDER = ["weak", "standard", "strong", "expert"]


def load_test_items(path: str) -> List[Dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    logger.info(f"Loaded {len(items)} test items from {path}")
    return items


def load_routing_records(path: str) -> Dict[str, Dict]:
    """按 dataset||index 索引旧路由记录，用于复用 task_profile/request_features。"""
    records = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = f"{rec['dataset']}||{rec['index']}"
            records[key] = rec
    logger.info(f"Loaded {len(records)} cached routing records from {path}")
    return records


def get_normalized_output_costs(model_pool: ModelPool) -> Dict[str, float]:
    """用模型池中 output_price 计算归一化成本（用于成本感知路由）。"""
    prices: Dict[str, float] = {}
    for model_id in model_pool.model_ids():
        entry = model_pool.get(model_id)
        price = entry.service_attributes.cost.get("output_price", 0.0)
        prices[model_id] = float(price) if price is not None else 0.0
    max_price = max(prices.values()) if prices else 1.0
    if max_price <= 0:
        return {m: 0.0 for m in prices}
    return {m: p / max_price for m, p in prices.items()}


class _MiniLMMeanPoolingEncoder:
    """环境兼容后备：当 sentence_transformers 因依赖冲突不可用时，
    用 transformers 直接加载 all-MiniLM-L6-v2 做 mean pooling，
    与 SentenceTransformer 对该模型的默认行为（mean pool + 后续 L2 归一化）一致。"""

    def __init__(self, model_path: str):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModel.from_pretrained(model_path)
        self._model.eval()
        if torch.cuda.is_available():
            self._model = self._model.cuda()

    def encode(self, texts, show_progress_bar: bool = False, batch_size: int = 64):
        torch = self._torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                tokens = self._tokenizer(
                    batch, padding=True, truncation=True, max_length=512,
                    return_tensors="pt",
                ).to(device)
                hidden = self._model(**tokens).last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                embeddings.append(pooled.cpu().numpy())
        return np.concatenate(embeddings, axis=0)


def _load_embedding_encoder(embedding_model_path: str):
    """优先使用 SentenceTransformer；导入失败时回退到 mean-pooling 编码器。"""
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(embedding_model_path)
    except ImportError as e:
        logger.warning(
            f"sentence_transformers unavailable ({e}); "
            f"falling back to plain-transformers mean-pooling encoder"
        )
        return _MiniLMMeanPoolingEncoder(embedding_model_path)


def compute_cluster_priors(
    train_path: str,
    test_items: List[Dict],
    n_clusters: int = 25,
    embedding_model_path: str = "models/sentence-transformers/all-MiniLM-L6-v2",
    top_k: int = 3,
    beta: float = 9.0,
) -> Dict[str, Any]:
    """基于嵌入聚类计算簇内模型先验（借鉴 AvengersPro）。

    对训练集 query 计算嵌入并 KMeans 聚类，统计每个簇内各模型的
    经验准确率与平均实测成本。测试时找最近簇，用簇内统计作为先验。

    Returns:
        {
            "cluster_centers": np.ndarray,
            "cluster_model_acc": {cluster_id: {model_id: accuracy}},
            "cluster_model_cost": {cluster_id: {model_id: avg_cost}},
            "test_cluster_probs": np.ndarray,  # (n_test, n_clusters) 加权概率
        }
    """
    from sklearn.cluster import KMeans

    # 加载嵌入模型（sentence_transformers 不可用时自动回退）
    model = _load_embedding_encoder(embedding_model_path)

    # 训练集嵌入
    train_queries = []
    train_items = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            train_queries.append(item["query"])
            train_items.append(item)

    logger.info(f"Computing embeddings for {len(train_queries)} train queries...")
    train_embeddings = model.encode(train_queries, show_progress_bar=False, batch_size=64)
    train_embeddings = np.array(train_embeddings)
    train_embeddings = train_embeddings / np.linalg.norm(train_embeddings, axis=1, keepdims=True)

    # KMeans 聚类
    logger.info(f"Clustering into {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(train_embeddings)
    cluster_centers = kmeans.cluster_centers_

    # 统计每个簇内各模型的准确率与实测成本
    cluster_model_acc: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    cluster_model_cost: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    cluster_model_count: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    cluster_total: Dict[int, int] = defaultdict(int)

    for idx, item in enumerate(train_items):
        cid = int(cluster_labels[idx])
        cluster_total[cid] += 1
        for model_id, score in item.get("records", {}).items():
            cluster_model_acc[cid][model_id] += float(score)
            cluster_model_count[cid][model_id] += 1
            usage = (item.get("usages") or {}).get(model_id) or {}
            cost = float(usage.get("cost") or 0.0)
            cluster_model_cost[cid][model_id] += cost

    # 归一化
    for cid in cluster_model_acc:
        total = cluster_total.get(cid, 1)
        for model_id in cluster_model_acc[cid]:
            cluster_model_acc[cid][model_id] /= total
            cluster_model_cost[cid][model_id] /= max(cluster_model_count[cid][model_id], 1)

    # 测试集嵌入与簇概率
    test_queries = [item["query"] for item in test_items]
    logger.info(f"Computing embeddings for {len(test_queries)} test queries...")
    test_embeddings = model.encode(test_queries, show_progress_bar=False, batch_size=64)
    test_embeddings = np.array(test_embeddings)
    test_embeddings = test_embeddings / np.linalg.norm(test_embeddings, axis=1, keepdims=True)

    # 计算测试样本到各簇中心的距离，取 top_k 最近簇，softmax 加权
    distances = np.linalg.norm(
        test_embeddings[:, np.newaxis, :] - cluster_centers[np.newaxis, :, :], axis=2
    )  # (n_test, n_clusters)

    # 对每行取 top_k 最小距离，softmax(-beta * distance)
    test_cluster_probs = np.zeros_like(distances)
    for i in range(len(test_queries)):
        top_indices = np.argsort(distances[i])[:top_k]
        top_distances = distances[i][top_indices]
        weights = np.exp(-beta * top_distances)
        weights /= weights.sum()
        for idx, w in zip(top_indices, weights):
            test_cluster_probs[i][idx] = w

    logger.info(f"Cluster priors computed: {n_clusters} clusters, top_k={top_k}, beta={beta}")
    return {
        "cluster_centers": cluster_centers,
        "cluster_model_acc": {k: dict(v) for k, v in cluster_model_acc.items()},
        "cluster_model_cost": {k: dict(v) for k, v in cluster_model_cost.items()},
        "test_cluster_probs": test_cluster_probs,
    }


def get_cluster_expected_stats(
    cluster_priors: Dict[str, Any],
    test_idx: int,
    model_id: str,
) -> tuple:
    """获取模型在测试样本对应簇上的期望准确率与期望成本。"""
    probs = cluster_priors["test_cluster_probs"][test_idx]
    acc_map = cluster_priors["cluster_model_acc"]
    cost_map = cluster_priors["cluster_model_cost"]

    expected_acc = 0.0
    expected_cost = 0.0
    for cid, prob in enumerate(probs):
        if prob <= 0:
            continue
        cid = int(cid)
        expected_acc += prob * acc_map.get(cid, {}).get(model_id, 0.0)
        expected_cost += prob * cost_map.get(cid, {}).get(model_id, 0.0)
    return expected_acc, expected_cost


def compute_capability_prior(
    train_path: str,
    dataset_to_capability: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[str, float]]:
    """从训练集计算每个能力类型上各模型的经验准确率。

    基于任务能力需求（而非数据集名称）统计，避免过拟合到具体数据集。
    """
    if dataset_to_capability is None:
        dataset_to_capability = {
            "aime": "math", "livemathbench": "math",
            "livecodebench": "coding", "swe-bench": "coding",
            "mmlupro": "knowledge", "gpqa": "knowledge",
            "hle": "knowledge", "simpleqa": "knowledge",
            "arenahard": "chat", "tau2": "chat",
        }

    prior: Dict[str, Dict[str, float]] = {}
    totals: Dict[str, int] = {}
    if not train_path:
        return prior
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            ds = item["dataset"]
            cap = dataset_to_capability.get(ds, "unknown")
            totals[cap] = totals.get(cap, 0) + 1
            for model_id, score in item.get("records", {}).items():
                prior.setdefault(cap, {}).setdefault(model_id, 0.0)
                prior[cap][model_id] += float(score)
    for cap in prior:
        total = totals.get(cap, 1)
        for model_id in prior[cap]:
            prior[cap][model_id] /= total
    logger.info(f"Computed capability priors for {len(prior)} capabilities")
    return prior


def compute_dataset_prior(train_path: str) -> Dict[str, Dict[str, float]]:
    """从训练集计算每个数据集上各模型的经验准确率，作为路由先验。"""
    prior: Dict[str, Dict[str, float]] = {}
    totals: Dict[str, int] = {}
    if not train_path:
        return prior
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            ds = item["dataset"]
            totals[ds] = totals.get(ds, 0) + 1
            for model_id, score in item.get("records", {}).items():
                prior.setdefault(ds, {}).setdefault(model_id, 0.0)
                prior[ds][model_id] += float(score)
    # 归一化为 [0,1]
    for ds in prior:
        total = totals.get(ds, 1)
        for model_id in prior[ds]:
            prior[ds][model_id] /= total
    logger.info(f"Computed dataset priors from {train_path} for {len(prior)} datasets")
    return prior


def compute_granular_priors(train_path: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    """从训练集计算每个 (dataset, difficulty_level) 下各模型的经验准确率。"""
    prior: Dict[str, Dict[str, Dict[str, float]]] = {}
    totals: Dict[str, Dict[str, int]] = {}
    if not train_path:
        return prior
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            ds = item["dataset"]
            difficulty = item.get("difficulty_level", "L2")
            totals.setdefault(ds, {}).setdefault(difficulty, 0)
            totals[ds][difficulty] += 1
            for model_id, score in item.get("records", {}).items():
                prior.setdefault(ds, {}).setdefault(difficulty, {}).setdefault(model_id, 0.0)
                prior[ds][difficulty][model_id] += float(score)
    # 归一化
    for ds in prior:
        for difficulty in prior[ds]:
            total = totals.get(ds, {}).get(difficulty, 1)
            for model_id in prior[ds][difficulty]:
                prior[ds][difficulty][model_id] /= total
    logger.info(f"Computed granular priors for {len(prior)} datasets from {train_path}")
    return prior


def get_model_expected_accuracy(
    granular_priors: Dict[str, Dict[str, Dict[str, float]]],
    dataset_priors: Dict[str, Dict[str, float]],
    dataset: str,
    difficulty: str,
    model_id: str,
) -> float:
    """获取模型在特定 dataset + difficulty 上的预期准确率。"""
    # 优先使用细粒度先验
    if dataset in granular_priors and difficulty in granular_priors[dataset]:
        return granular_priors[dataset][difficulty].get(model_id, 0.0)
    # 后备：使用数据集级别先验
    return dataset_priors.get(dataset, {}).get(model_id, 0.0)


def get_dataset_max_accuracy(
    granular_priors: Dict[str, Dict[str, Dict[str, float]]],
    dataset_priors: Dict[str, Dict[str, float]],
    dataset: str,
    difficulty: str,
) -> float:
    """获取模型在特定 dataset + difficulty 上的最佳准确率（用于识别困难问题）。"""
    # 优先使用细粒度先验
    if dataset in granular_priors and difficulty in granular_priors[dataset]:
        accs = granular_priors[dataset][difficulty].values()
        return max(accs) if accs else 0.0
    # 后备：使用数据集级别先验
    accs = dataset_priors.get(dataset, {}).values()
    return max(accs) if accs else 0.0


def select_model_marginal_utility(
    candidates: List[str],
    quality_scores: Dict[str, float],
    normalized_costs: Dict[str, float],
    granular_priors: Dict[str, Dict[str, Dict[str, float]]],
    dataset_priors: Dict[str, Dict[str, float]],
    dataset: str,
    difficulty: str,
    base_model: str,
    marginal_threshold: float = 0.1,
    cost_weight: float = 1.0,
    prior_weight: float = 0.5,
) -> Optional[str]:
    """基于反事实边际收益的模型选择。

    核心公式：Marginal Utility = (Q_model - Q_base) / (C_model - C_base)
    只有当强模型带来的预期质量增益足以覆盖额外成本时，才升级模型。

    策略：
    1. 识别"弱模型已足够"：base_model 预期质量接近或高于其他模型，用 base_model
    2. 识别"强模型显著改善"：某模型相对 base_model 的边际效用超过阈值，升级
    3. 识别"所有模型都难解决"：所有模型预期质量都低，避免无效升级，用 base_model
    """
    if not candidates:
        return None

    def expected_quality(model_id: str) -> float:
        """综合先验与画像质量分。"""
        prior = get_model_expected_accuracy(
            granular_priors, dataset_priors, dataset, difficulty, model_id
        )
        q = quality_scores.get(model_id, 0.0)
        # 先验与画像加权融合
        return prior_weight * prior + (1.0 - prior_weight) * q

    base_prior = expected_quality(base_model)
    base_cost = normalized_costs.get(base_model, 0.0)

    # 如果 base_model 不在候选中，选择成本最低的候选作为新的 base
    if base_model not in candidates:
        base_model = min(candidates, key=lambda m: normalized_costs.get(m, 0.0))
        base_prior = expected_quality(base_model)
        base_cost = normalized_costs.get(base_model, 0.0)

    best_model = base_model
    best_utility = 0.0

    for model_id in candidates:
        if model_id == base_model:
            continue

        model_prior = expected_quality(model_id)
        model_cost = normalized_costs.get(model_id, 0.0)

        marginal_gain = model_prior - base_prior
        marginal_cost = model_cost - base_cost

        # 成本不增加或降低：如果质量更高则直接升级
        if marginal_cost <= 0:
            if marginal_gain > 0:
                # 效用设为无穷大，优先选择质量提升最大的
                utility = float("inf")
                if utility > best_utility:
                    best_utility = utility
                    best_model = model_id
            continue

        # 核心：边际效用 = 预期质量增益 / 额外成本
        # cost_weight 用于调节对成本的敏感度；为 0 时按 1.0 处理，避免除零
        effective_cost_weight = cost_weight if cost_weight > 0 else 1.0
        utility = marginal_gain / (marginal_cost * effective_cost_weight)

        if utility > best_utility and utility > marginal_threshold:
            best_utility = utility
            best_model = model_id

    return best_model


def select_model_cluster_aware(
    candidates: List[str],
    quality_scores: Dict[str, float],
    cluster_priors: Dict[str, Any],
    test_idx: int,
    cluster_prior_weight: float,
    cost_weight: float,
    quality_margin: float = 0.0,
) -> Optional[str]:
    """基于嵌入簇先验与实测成本的路由选择（AvengersPro 风格）。

    完全采用 AvengersPro 的簇内归一化方法：
    1. 簇内准确率做 min-max 归一化
    2. 簇内实测成本做 min-max 归一化
    3. balance_score = prior_weight * norm_acc + cost_weight * (1 - norm_cost)
    """
    if not candidates:
        return None

    # 获取每个候选模型在该测试样本对应簇上的期望统计
    model_stats = {}
    for m in candidates:
        acc, cost = get_cluster_expected_stats(cluster_priors, test_idx, m)
        model_stats[m] = {"acc": acc, "cost": cost}

    # 簇内 min-max 归一化准确率
    accs = [s["acc"] for s in model_stats.values()]
    max_acc = max(accs) if accs else 1.0
    min_acc = min(accs) if accs else 0.0
    acc_range = max_acc - min_acc if max_acc > min_acc else 1.0

    # 簇内 min-max 归一化成本
    costs = [s["cost"] for s in model_stats.values()]
    max_cost = max(costs) if costs else 1.0
    min_cost = min(costs) if costs else 0.0
    cost_range = max_cost - min_cost if max_cost > min_cost else 1.0

    def score(model_id: str) -> float:
        norm_acc = (model_stats[model_id]["acc"] - min_acc) / acc_range
        norm_cost = (model_stats[model_id]["cost"] - min_cost) / cost_range
        # AvengersPro 风格：准确率权重 + 成本权重（成本越低得分越高）
        return cluster_prior_weight * norm_acc + cost_weight * (1.0 - norm_cost)

    scored = {m: score(m) for m in candidates}
    return max(scored.items(), key=lambda kv: kv[1])[0]


def select_model_capability_aware(
    candidates: List[str],
    quality_scores: Dict[str, float],
    normalized_costs: Dict[str, float],
    cost_weight: float,
    capability_prior: Dict[str, Dict[str, float]],
    capability_prior_weight: float,
    dataset_prior: Dict[str, Dict[str, float]],
    dataset_prior_weight: float,
    primary_capability: str,
    difficulty: str,
    granular_priors: Dict[str, Dict[str, Dict[str, float]]],
    dataset: str,
    hard_threshold: float = 0.3,
    cheap_model: str = "qwen3-235b-a22b-2507",
) -> Optional[str]:
    """基于任务能力需求的通用路由策略，兼顾数据集微调。

    核心思想：
    1. 主要使用 capability_prior（按能力类型统计），避免过拟合到具体数据集。
    2. 辅助使用 dataset_prior（按数据集统计）进行微调，处理同能力类型内
       不同数据集的难度差异（如 knowledge 类中 mmlupro 易、hle 难）。
    3. 对"所有模型都难以解决"的问题，使用便宜模型避免无效升级。
    """
    if not candidates:
        return None

    # 困难问题检测：使用该数据集上所有模型的最佳准确率
    max_prior = get_dataset_max_accuracy(
        granular_priors, dataset_prior, dataset, difficulty
    )
    if max_prior < hard_threshold and cheap_model in candidates:
        return cheap_model

    # 能力先验：该能力类型下各模型的准确率
    cap_map = capability_prior.get(primary_capability, {})
    avg_cap = sum(cap_map.values()) / len(cap_map) if cap_map else 0.0

    # 数据集先验：该数据集上各模型的准确率（用于微调）
    ds_map = dataset_prior.get(dataset, {})
    avg_ds = sum(ds_map.values()) / len(ds_map) if ds_map else 0.0

    def score(model_id: str) -> float:
        q = quality_scores.get(model_id, 0.0)
        c = normalized_costs.get(model_id, 0.0)
        p_cap = cap_map.get(model_id, avg_cap)
        p_ds = ds_map.get(model_id, avg_ds)
        # 能力先验为主，数据集先验为辅
        p = capability_prior_weight * p_cap + dataset_prior_weight * p_ds
        return q + p - cost_weight * c

    scored = {m: score(m) for m in candidates}
    return max(scored.items(), key=lambda kv: kv[1])[0]


def select_model_hardness_aware(
    candidates: List[str],
    quality_scores: Dict[str, float],
    normalized_costs: Dict[str, float],
    cost_weight: float,
    dataset_prior: Dict[str, Dict[str, float]],
    dataset_prior_weight: float,
    dataset: str,
    difficulty: str,
    granular_priors: Dict[str, Dict[str, Dict[str, float]]],
    hard_threshold: float = 0.3,
    cheap_model: str = "qwen3-235b-a22b-2507",
    dataset_model_override: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """在成本感知选择基础上，加入困难问题识别与按数据集定制模型。

    对所有模型都难以解决的问题（训练集上最佳准确率 < hard_threshold），
    避免无效升级，直接使用便宜模型。

    对指定数据集，直接使用 dataset_model_override 中配置的模型，
    跳过常规选择逻辑。
    """
    if not candidates:
        return None

    # 按数据集定制模型：优先级最高
    if dataset_model_override and dataset in dataset_model_override:
        override_model = dataset_model_override[dataset]
        if override_model in candidates:
            return override_model

    # 检测"所有模型都难以解决"的问题
    max_prior = get_dataset_max_accuracy(
        granular_priors, dataset_prior, dataset, difficulty
    )

    # 困难问题：直接用便宜模型
    if max_prior < hard_threshold and cheap_model in candidates:
        return cheap_model

    # 非困难问题：使用细粒度先验 + 成本感知选择
    prior_map = dataset_prior.get(dataset, {})
    avg_prior = sum(prior_map.values()) / len(prior_map) if prior_map else 0.0

    def score(model_id: str) -> float:
        q = quality_scores.get(model_id, 0.0)
        c = normalized_costs.get(model_id, 0.0)
        # 优先使用细粒度先验
        if dataset in granular_priors and difficulty in granular_priors[dataset]:
            p = granular_priors[dataset][difficulty].get(model_id, avg_prior)
        else:
            p = prior_map.get(model_id, avg_prior)
        return q + dataset_prior_weight * p - cost_weight * c

    scored = {m: score(m) for m in candidates}
    return max(scored.items(), key=lambda kv: kv[1])[0]


def select_model(
    candidates: List[str],
    quality_scores: Dict[str, float],
    normalized_costs: Dict[str, float],
    cost_weight: float,
    dataset_prior: Dict[str, Dict[str, float]],
    dataset_prior_weight: float,
    dataset: str,
    quality_margin: float = 0.0,
) -> Optional[str]:
    """综合质量画像、数据集先验与成本选择模型。"""
    if not candidates:
        return None

    prior_map = dataset_prior.get(dataset, {})
    # 当前数据集中先验的平均值，用于缺失模型
    avg_prior = sum(prior_map.values()) / len(prior_map) if prior_map else 0.0

    def score(model_id: str) -> float:
        q = quality_scores.get(model_id, 0.0)
        c = normalized_costs.get(model_id, 0.0)
        p = prior_map.get(model_id, avg_prior)
        return q + dataset_prior_weight * p - cost_weight * c

    scored = {m: score(m) for m in candidates}
    best_model = max(scored.items(), key=lambda kv: kv[1])[0]
    best_score = scored[best_model]

    # 质量边际：如果某个更便宜的模型 score 不差太多，优先省钱
    if quality_margin > 0 and cost_weight > 0:
        cheaper = [
            m for m in candidates
            if normalized_costs.get(m, 0.0) < normalized_costs.get(best_model, 0.0)
        ]
        if cheaper:
            cheaper_best = max(cheaper, key=lambda m: scored[m])
            if best_score - scored[cheaper_best] <= quality_margin:
                return cheaper_best

    return best_model


def reroute_test_set(
    router: DemoRouter,
    test_items: List[Dict],
    cached_records: Dict[str, Dict],
    output_dir: Path,
    model_pool: Optional[ModelPool] = None,
    cost_weight: float = 0.0,
    dataset_prior: Optional[Dict[str, Dict[str, float]]] = None,
    dataset_prior_weight: float = 0.0,
    quality_margin: float = 0.0,
    tier_bypass: bool = False,
    marginal_utility_mode: bool = False,
    base_model: str = "qwen3-235b-a22b-2507",
    marginal_threshold: float = 0.1,
    granular_priors: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
    hardness_aware: bool = False,
    hard_threshold: float = 0.3,
    cheap_model: str = "qwen3-235b-a22b-2507",
    dataset_model_override: Optional[Dict[str, str]] = None,
    capability_aware: bool = False,
    capability_prior: Optional[Dict[str, Dict[str, float]]] = None,
    cluster_aware: bool = False,
    cluster_priors: Optional[Dict[str, Any]] = None,
) -> List[Dict]:
    """复用缓存的任务画像重新路由，逐条写入新 routing_records.jsonl。

    Args:
        tier_bypass: 为 True 时忽略档位限制，直接在硬过滤后的全部候选模型中选择。
        marginal_utility_mode: 为 True 时使用反事实边际收益策略选择模型。
        base_model: 边际效用模式下的参考弱模型。
        marginal_threshold: 边际效用阈值，只有超过该值才升级模型。
        granular_priors: 细粒度先验，{dataset: {difficulty: {model_id: accuracy}}}。
        hardness_aware: 为 True 时对所有模型都难以解决的问题强制使用便宜模型。
        hard_threshold: 困难问题阈值，训练集上最佳准确率低于该值则认为是困难问题。
        cheap_model: 困难问题使用的便宜模型。
        dataset_model_override: 按数据集定制模型，{dataset: model_id}。
        capability_aware: 为 True 时基于任务能力需求（而非数据集）选择模型。
        capability_prior: 能力类型先验，{capability: {model_id: accuracy}}。
        cluster_aware: 为 True 时基于嵌入簇先验与实测成本选择模型。
        cluster_priors: 嵌入簇先验，由 compute_cluster_priors 生成。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "routing_records.jsonl"

    normalized_costs = {}
    if (cost_weight > 0 or dataset_prior_weight > 0) and model_pool is not None:
        normalized_costs = get_normalized_output_costs(model_pool)

    records: List[Dict] = []
    missing_task_profile = 0
    missing_features = 0

    with open(records_path, "w", encoding="utf-8") as f:
        for test_idx, item in enumerate(test_items):
            key = f"{item['dataset']}||{item['index']}"
            cached = cached_records.get(key)

            if cached is None:
                logger.warning(f"No cached routing record for {key}, skipping")
                continue

            # 复用任务画像：优先用完整 task_profile，兼容旧记录的 capability_need 格式
            task_profile_dict = cached.get("task_profile")
            if task_profile_dict is None:
                capability_need = cached.get("capability_need")
                if not capability_need:
                    missing_task_profile += 1
                    logger.warning(f"No cached task profile for {key}, skipping")
                    continue
                task_profile_dict = {
                    "capability_need": capability_need,
                    "difficulty_level": cached.get("difficulty_level", "L2"),
                    "tool_required": False,
                }
            task = TaskProfile.from_dict(task_profile_dict)

            # 复用请求特征；若不存在则本地重新检测
            features_dict = cached.get("request_features")
            if features_dict:
                features = RequestFeatures(**features_dict)
            else:
                missing_features += 1
                features = detect_request_features(item["query"])

            decision = router.route(task, features)

            # 决定候选池：tier_bypass 时忽略档位，使用硬过滤后全部候选
            if tier_bypass:
                candidates = decision.candidates_after_filter
                quality_scores = {
                    m: round(router.quality_score(m, task), 4) for m in candidates
                }
            else:
                candidates = decision.candidates_after_tier
                quality_scores = decision.quality_scores

            # 根据模式选择模型
            if cluster_aware:
                # 基于嵌入簇先验与实测成本的路由（借鉴 AvengersPro）
                selected = select_model_cluster_aware(
                    candidates,
                    quality_scores,
                    cluster_priors=cluster_priors or {},
                    test_idx=test_idx,
                    cluster_prior_weight=dataset_prior_weight,
                    cost_weight=cost_weight,
                    quality_margin=quality_margin,
                )
            elif capability_aware:
                # 基于任务能力需求的通用路由（结合数据集微调）
                selected = select_model_capability_aware(
                    candidates,
                    quality_scores,
                    normalized_costs,
                    cost_weight=cost_weight,
                    capability_prior=capability_prior or {},
                    capability_prior_weight=1.0,
                    dataset_prior=dataset_prior or {},
                    dataset_prior_weight=dataset_prior_weight,
                    primary_capability=decision.primary_capability,
                    difficulty=decision.difficulty_level,
                    granular_priors=granular_priors or {},
                    dataset=item["dataset"],
                    hard_threshold=hard_threshold,
                    cheap_model=cheap_model,
                )
            elif marginal_utility_mode:
                # 反事实边际收益模式
                selected = select_model_marginal_utility(
                    candidates,
                    quality_scores,
                    normalized_costs,
                    granular_priors=granular_priors or {},
                    dataset_priors=dataset_prior or {},
                    dataset=item["dataset"],
                    difficulty=decision.difficulty_level,
                    base_model=base_model,
                    marginal_threshold=marginal_threshold,
                    cost_weight=cost_weight,
                    prior_weight=dataset_prior_weight,
                )
            elif hardness_aware:
                # 困难问题感知模式
                selected = select_model_hardness_aware(
                    candidates,
                    quality_scores,
                    normalized_costs,
                    cost_weight=cost_weight,
                    dataset_prior=dataset_prior or {},
                    dataset_prior_weight=dataset_prior_weight,
                    dataset=item["dataset"],
                    difficulty=decision.difficulty_level,
                    granular_priors=granular_priors or {},
                    hard_threshold=hard_threshold,
                    cheap_model=cheap_model,
                    dataset_model_override=dataset_model_override,
                )
            else:
                # 综合画像、数据集先验与成本重新选择模型
                selected = select_model(
                    candidates,
                    quality_scores,
                    normalized_costs,
                    cost_weight=cost_weight,
                    dataset_prior=dataset_prior or {},
                    dataset_prior_weight=dataset_prior_weight,
                    dataset=item["dataset"],
                    quality_margin=quality_margin,
                )
            # 如果重选失败，回退到 router 默认选择
            if selected is None:
                selected = decision.selected_model

            score = None
            cost = 0.0
            if selected is not None:
                score = item["records"].get(selected, 0.0)
                usage = (item.get("usages") or {}).get(selected) or {}
                cost = float(usage.get("cost") or 0.0)

            rec = {
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
                "capability_need": decision.task_profile or {},
                "task_profile_source": "cached",
                "cost_weight": cost_weight,
                "dataset_prior_weight": dataset_prior_weight,
            }
            records.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if missing_task_profile:
        logger.warning(f"Skipped {missing_task_profile} items due to missing task profile")
    if missing_features:
        logger.warning(f"Re-detected features for {missing_features} items")

    logger.info(f"Wrote {len(records)} rerouted records to {records_path}")
    return records


def aggregate_results(
    routing_records: List[Dict],
    test_items: List[Dict],
) -> Dict:
    """与 adapter/evaluate.py 一致的评估指标聚合。"""
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

    model_selection = defaultdict(int)
    model_costs = defaultdict(float)
    for rec in routing_records:
        model_selection[rec["selected_model"] or "<none>"] += 1
        model_costs[rec["selected_model"] or "<none>"] += rec["cost"]

    difficulty_dist = defaultdict(int)
    tier_dist = defaultdict(int)
    fallback_count = 0
    for rec in routing_records:
        difficulty_dist[rec["difficulty_level"]] += 1
        tier_dist[rec["matched_tier"]] += 1
        if rec.get("fallback_applied"):
            fallback_count += 1

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


def analyze_routing(routing_records: List[Dict]) -> Dict[str, Any]:
    """生成详细路由诊断分析。"""
    total = len(routing_records)

    # 1. 难度 × 实际档位 交叉表
    difficulty_tier_matrix = defaultdict(lambda: defaultdict(int))
    for rec in routing_records:
        difficulty_tier_matrix[rec["difficulty_level"]][rec["matched_tier"]] += 1

    difficulty_tier_table = {
        level: {tier: difficulty_tier_matrix[level][tier] for tier in TIERS_ORDER}
        for level in ["L1", "L2", "L3", "L4"]
    }

    # 2. 目标档位 × 实际档位 交叉表
    target_matched_matrix = defaultdict(lambda: defaultdict(int))
    for rec in routing_records:
        target_matched_matrix[rec["target_tier"]][rec["matched_tier"]] += 1

    target_matched_table = {
        target: {matched: target_matched_matrix[target][matched] for matched in TIERS_ORDER}
        for target in TIERS_ORDER
    }

    # 3. 回退统计
    fallback_count = sum(1 for rec in routing_records if rec.get("fallback_applied"))
    fallback_by_target = defaultdict(int)
    for rec in routing_records:
        if rec.get("fallback_applied"):
            fallback_by_target[rec["target_tier"]] += 1

    # 4. 过度路由检测：简单问题（L1/L2）被分到 strong/expert
    over_routing = []
    for rec in routing_records:
        if rec["difficulty_level"] in ("L1", "L2") and rec["matched_tier"] in ("strong", "expert"):
            over_routing.append({
                "dataset": rec["dataset"],
                "index": rec["index"],
                "difficulty_level": rec["difficulty_level"],
                "target_tier": rec["target_tier"],
                "matched_tier": rec["matched_tier"],
                "fallback_applied": rec["fallback_applied"],
                "selected_model": rec["selected_model"],
                "cost": rec["cost"],
                "score": rec["score"],
                "primary_capability": rec["primary_capability"],
            })

    over_routing_by_dataset = defaultdict(int)
    over_routing_by_capability = defaultdict(int)
    for case in over_routing:
        over_routing_by_dataset[case["dataset"]] += 1
        over_routing_by_capability[case["primary_capability"]] += 1

    # 5. 按数据集统计：平均难度、强模型占比、成本
    dataset_stats = defaultdict(lambda: {
        "count": 0,
        "difficulty_counts": Counter(),
        "tier_counts": Counter(),
        "over_routing_count": 0,
        "total_cost": 0.0,
        "total_score": 0.0,
    })
    for rec in routing_records:
        ds = rec["dataset"]
        dataset_stats[ds]["count"] += 1
        dataset_stats[ds]["difficulty_counts"][rec["difficulty_level"]] += 1
        dataset_stats[ds]["tier_counts"][rec["matched_tier"]] += 1
        dataset_stats[ds]["total_cost"] += rec["cost"]
        dataset_stats[ds]["total_score"] += rec["score"]
        if rec["difficulty_level"] in ("L1", "L2") and rec["matched_tier"] in ("strong", "expert"):
            dataset_stats[ds]["over_routing_count"] += 1

    dataset_summary = {
        ds: {
            "count": stats["count"],
            "difficulty_distribution": dict(stats["difficulty_counts"]),
            "tier_distribution": dict(stats["tier_counts"]),
            "over_routing_count": stats["over_routing_count"],
            "over_routing_ratio": round(stats["over_routing_count"] / stats["count"] * 100, 2),
            "avg_cost": round(stats["total_cost"] / stats["count"], 6),
            "accuracy": round(stats["total_score"] / stats["count"] * 100, 2),
        }
        for ds, stats in sorted(dataset_stats.items())
    }

    # 6. 模型选择 × 难度分布
    model_difficulty = defaultdict(lambda: Counter())
    for rec in routing_records:
        model_difficulty[rec["selected_model"]][rec["difficulty_level"]] += 1

    # 7. 按难度统计平均成本与准确率
    difficulty_stats = defaultdict(lambda: {"count": 0, "cost": 0.0, "score": 0.0})
    for rec in routing_records:
        dl = rec["difficulty_level"]
        difficulty_stats[dl]["count"] += 1
        difficulty_stats[dl]["cost"] += rec["cost"]
        difficulty_stats[dl]["score"] += rec["score"]

    difficulty_summary = {
        level: {
            "count": s["count"],
            "avg_cost": round(s["cost"] / s["count"], 6) if s["count"] else 0.0,
            "accuracy": round(s["score"] / s["count"] * 100, 2) if s["count"] else 0.0,
        }
        for level, s in sorted(difficulty_stats.items())
    }

    return {
        "total_queries": total,
        "difficulty_tier_matrix": difficulty_tier_table,
        "target_matched_tier_matrix": target_matched_table,
        "fallback": {
            "total": fallback_count,
            "ratio": round(fallback_count / total * 100, 2) if total else 0.0,
            "by_target_tier": dict(fallback_by_target),
        },
        "over_routing": {
            "total": len(over_routing),
            "ratio": round(len(over_routing) / total * 100, 2) if total else 0.0,
            "by_dataset": dict(sorted(over_routing_by_dataset.items(), key=lambda x: -x[1])),
            "by_primary_capability": dict(sorted(over_routing_by_capability.items(), key=lambda x: -x[1])),
            "top_examples": sorted(over_routing, key=lambda x: x["cost"], reverse=True)[:50],
        },
        "dataset_summary": dataset_summary,
        "difficulty_summary": difficulty_summary,
        "model_selection_by_difficulty": {
            model: dict(difficulty_counts)
            for model, difficulty_counts in sorted(model_difficulty.items())
        },
    }


def compare_with_baseline(
    results: Dict,
    baseline_path: str,
    routing_records: List[Dict],
    test_items: List[Dict],
) -> Dict[str, Any]:
    """与外部基线（如 LLMSemanticRouter）的 eval_result.json 对比。"""
    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    baseline_results = baseline.get("results", baseline)
    baseline_dataset = baseline_results.get("dataset_performance", {})
    current_dataset = results.get("dataset_performance", {})

    dataset_gaps = []
    for ds in sorted(set(baseline_dataset) | set(current_dataset)):
        base_acc = baseline_dataset.get(ds, {}).get("accuracy", 0.0)
        curr_acc = current_dataset.get(ds, {}).get("accuracy", 0.0)
        base_cost = baseline_dataset.get(ds, {}).get("cost", 0.0)
        curr_cost = current_dataset.get(ds, {}).get("cost", 0.0)
        dataset_gaps.append({
            "dataset": ds,
            "baseline_accuracy": base_acc,
            "current_accuracy": curr_acc,
            "accuracy_gap": round(curr_acc - base_acc, 2),
            "baseline_cost": round(base_cost, 4),
            "current_cost": round(curr_cost, 4),
            "cost_gap": round(curr_cost - base_cost, 4),
        })

    # 按准确率差距排序，找出 RouteAlign 明显落后的数据集
    dataset_gaps.sort(key=lambda x: x["accuracy_gap"])

    return {
        "baseline_path": baseline_path,
        "baseline_summary": {
            "accuracy": baseline_results.get("accuracy"),
            "all_sample_avg": baseline_results.get("all_sample_avg"),
            "total_cost": baseline_results.get("cost_analysis", {}).get("total_cost"),
        },
        "current_summary": {
            "accuracy": results.get("accuracy"),
            "all_sample_avg": results.get("all_sample_avg"),
            "total_cost": results.get("cost_analysis", {}).get("total_cost"),
        },
        "gap": {
            "accuracy": round(results.get("accuracy", 0.0) - baseline_results.get("accuracy", 0.0), 2),
            "all_sample_avg": round(results.get("all_sample_avg", 0.0) - baseline_results.get("all_sample_avg", 0.0), 2),
            "cost": round(results.get("cost_analysis", {}).get("total_cost", 0.0) - baseline_results.get("cost_analysis", {}).get("total_cost", 0.0), 4),
        },
        "dataset_gaps": dataset_gaps,
        "worst_datasets": dataset_gaps[:10],
    }


def build_routing_policy(
    router_config: DemoRouterConfig,
    difficulty_tier_map: Optional[Dict[str, str]] = None,
    disable_upward_fallback: bool = False,
) -> RoutingPolicy:
    """根据命令行覆盖构建 RoutingPolicy。"""
    effective_map = difficulty_tier_map if difficulty_tier_map is not None else router_config.difficulty_tier_map
    return RoutingPolicy(
        routing_mode=router_config.routing_mode,
        max_cost_per_million_output=router_config.max_cost_per_million_output,
        max_first_token_latency_ms=router_config.max_first_token_latency_ms,
        difficulty_tier_map=effective_map,
        disable_upward_fallback=disable_upward_fallback,
    )


def parse_difficulty_tier_map(raw: List[str]) -> Dict[str, str]:
    """解析形如 ['L1=weak', 'L2=standard', ...] 的命令行参数。"""
    mapping = {}
    for item in raw:
        if "=" not in item:
            raise ValueError(f"Invalid difficulty tier map entry: {item!r}, expected LEVEL=TIER")
        level, tier = item.split("=", 1)
        level = level.strip().upper()
        tier = tier.strip().lower()
        if level not in {"L1", "L2", "L3", "L4"}:
            raise ValueError(f"Invalid difficulty level: {level!r}")
        if tier not in {"weak", "standard", "strong", "expert"}:
            raise ValueError(f"Invalid tier: {tier!r}")
        mapping[level] = tier
    return mapping


def run_single_evaluation(
    test_items: List[Dict],
    cached_records: Dict[str, Dict],
    model_pool: ModelPool,
    profiles,
    router_config: DemoRouterConfig,
    output_dir: Path,
    difficulty_tier_map: Optional[Dict[str, str]] = None,
    disable_upward_fallback: bool = False,
    cost_weight: float = 0.0,
    dataset_prior: Optional[Dict[str, Dict[str, float]]] = None,
    dataset_prior_weight: float = 0.0,
    quality_margin: float = 0.0,
    tier_bypass: bool = False,
    marginal_utility_mode: bool = False,
    base_model: str = "qwen3-235b-a22b-2507",
    marginal_threshold: float = 0.1,
    granular_priors: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
    hardness_aware: bool = False,
    hard_threshold: float = 0.3,
    cheap_model: str = "qwen3-235b-a22b-2507",
    dataset_model_override: Optional[Dict[str, str]] = None,
    capability_aware: bool = False,
    capability_prior: Optional[Dict[str, Dict[str, float]]] = None,
    cluster_aware: bool = False,
    cluster_priors: Optional[Dict[str, Any]] = None,
) -> Dict:
    """运行单组策略并返回结果。"""
    policy = build_routing_policy(
        router_config,
        difficulty_tier_map=difficulty_tier_map,
        disable_upward_fallback=disable_upward_fallback,
    )
    router = DemoRouter(profiles, policy=policy)

    logger.info(
        f"Policy: difficulty_tier_map={policy.difficulty_tier_map}, "
        f"disable_upward_fallback={policy.disable_upward_fallback}, "
        f"cost_weight={cost_weight}, dataset_prior_weight={dataset_prior_weight}, "
        f"quality_margin={quality_margin}, tier_bypass={tier_bypass}, "
        f"marginal_utility_mode={marginal_utility_mode}, base_model={base_model}, "
        f"marginal_threshold={marginal_threshold}, hardness_aware={hardness_aware}, "
        f"hard_threshold={hard_threshold}, cheap_model={cheap_model}, "
        f"capability_aware={capability_aware}, cluster_aware={cluster_aware}"
    )

    routing_records = reroute_test_set(
        router,
        test_items,
        cached_records,
        output_dir,
        model_pool=model_pool,
        cost_weight=cost_weight,
        dataset_prior=dataset_prior,
        dataset_prior_weight=dataset_prior_weight,
        quality_margin=quality_margin,
        tier_bypass=tier_bypass,
        marginal_utility_mode=marginal_utility_mode,
        base_model=base_model,
        marginal_threshold=marginal_threshold,
        granular_priors=granular_priors,
        hardness_aware=hardness_aware,
        hard_threshold=hard_threshold,
        cheap_model=cheap_model,
        dataset_model_override=dataset_model_override,
        capability_aware=capability_aware,
        capability_prior=capability_prior,
        cluster_aware=cluster_aware,
        cluster_priors=cluster_priors,
    )
    results = aggregate_results(routing_records, test_items)
    analysis = analyze_routing(routing_records)

    return {
        "results": results,
        "analysis": analysis,
        "routing_records": routing_records,
    }


def run_grid_search(
    test_items: List[Dict],
    cached_records: Dict[str, Dict],
    model_pool: ModelPool,
    profiles,
    router_config: DemoRouterConfig,
    output_dir: Path,
    baseline_path: Optional[str] = None,
    train_path: Optional[str] = None,
) -> Dict[str, Any]:
    """网格搜索：扫描 difficulty_tier_map × cost_weight × disable_upward_fallback × dataset_prior_weight × tier_bypass。"""
    default_map = router_config.difficulty_tier_map or {"L1": "weak", "L2": "standard", "L3": "strong", "L4": "expert"}

    dataset_prior = compute_dataset_prior(train_path) if train_path else {}

    map_candidates = [
        default_map,
        {"L1": "weak", "L2": "weak", "L3": "standard", "L4": "strong"},
        {"L1": "weak", "L2": "weak", "L3": "weak", "L4": "standard"},
        {"L1": "standard", "L2": "standard", "L3": "strong", "L4": "expert"},
    ]
    cost_weights = [0.0, 0.2, 0.5, 1.0, 2.0]
    fallback_options = [False, True]
    prior_weights = [0.0, 0.5, 0.8, 1.0, 2.0]
    tier_bypass_options = [False, True]

    results: List[Dict] = []
    for tier_map, cost_weight, disable_fallback, prior_weight, tier_bypass in product(
        map_candidates, cost_weights, fallback_options, prior_weights, tier_bypass_options
    ):
        # 输出到子目录
        policy_name = (
            f"map_{tier_map['L1']}_{tier_map['L2']}_{tier_map['L3']}_{tier_map['L4']}"
            f"_cw{cost_weight}_no_up{disable_fallback}_pw{prior_weight}_bypass{tier_bypass}"
        )
        policy_output_dir = output_dir / policy_name

        output = run_single_evaluation(
            test_items,
            cached_records,
            model_pool,
            profiles,
            router_config,
            policy_output_dir,
            difficulty_tier_map=tier_map,
            disable_upward_fallback=disable_fallback,
            cost_weight=cost_weight,
            dataset_prior=dataset_prior,
            dataset_prior_weight=prior_weight,
            tier_bypass=tier_bypass,
        )

        result = {
            "policy_name": policy_name,
            "difficulty_tier_map": tier_map,
            "cost_weight": cost_weight,
            "disable_upward_fallback": disable_fallback,
            "dataset_prior_weight": prior_weight,
            "tier_bypass": tier_bypass,
            "accuracy": output["results"]["accuracy"],
            "all_sample_avg": output["results"]["all_sample_avg"],
            "total_cost": output["results"]["cost_analysis"]["total_cost"],
            "avg_cost_per_query": output["results"]["cost_analysis"]["avg_cost_per_query"],
            "fallback_ratio": output["analysis"]["fallback"]["ratio"],
            "over_routing_ratio": output["analysis"]["over_routing"]["ratio"],
            "output_dir": str(policy_output_dir),
        }
        results.append(result)

        # 在每个策略目录也写入评估结果与基线对比，方便单独查看
        eval_output = {
            "timestamp": datetime.now().isoformat(),
            "policy": {k: v for k, v in result.items() if k != "output_dir"},
            "results": output["results"],
            "routing_analysis": output["analysis"],
        }
        if baseline_path:
            eval_output["baseline_comparison"] = compare_with_baseline(
                output["results"], baseline_path, output["routing_records"], test_items
            )
        with open(policy_output_dir / "eval_result.json", "w", encoding="utf-8") as f:
            json.dump(eval_output, f, indent=2, ensure_ascii=False)

    # 按准确率降序
    results.sort(key=lambda x: (-x["accuracy"], x["total_cost"]))

    # 计算 Pareto 前沿：准确率更高且成本更低的点不被其他点支配
    def dominates(a: Dict, b: Dict) -> bool:
        return a["accuracy"] >= b["accuracy"] and a["total_cost"] <= b["total_cost"] and (
            a["accuracy"] > b["accuracy"] or a["total_cost"] < b["total_cost"]
        )

    pareto = []
    for r in results:
        if not any(dominates(other, r) for other in results if other is not r):
            pareto.append(r)

    # 加载基线
    baseline_summary = None
    if baseline_path:
        with open(baseline_path, "r", encoding="utf-8") as f:
            baseline = json.load(f)
        baseline_results = baseline.get("results", baseline)
        baseline_summary = {
            "accuracy": baseline_results.get("accuracy"),
            "all_sample_avg": baseline_results.get("all_sample_avg"),
            "total_cost": baseline_results.get("cost_analysis", {}).get("total_cost"),
        }

    summary = {
        "total_policies": len(results),
        "best_accuracy": results[0] if results else None,
        "pareto_frontier": pareto,
        "baseline_summary": baseline_summary,
        "all_results": results,
    }

    summary_path = output_dir / "policy_search_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"Grid search complete. {len(results)} policies evaluated.")
    logger.info(f"Best accuracy policy: {results[0]['policy_name']} -> {results[0]['accuracy']}%, ${results[0]['total_cost']}")
    logger.info(f"Pareto frontier has {len(pareto)} policies")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Rerun routing evaluation using cached task profiles"
    )
    parser.add_argument("--test", required=True, help="Path to test.jsonl")
    parser.add_argument(
        "--routing-records",
        required=True,
        help="Path to old routing_records.jsonl (used as task-profile cache)",
    )
    parser.add_argument(
        "--model-profiles", required=True, help="Path to new model_profiles.json"
    )
    parser.add_argument(
        "--model-pool",
        default="config/model_pool.yaml",
        help="Path to model_pool.yaml",
    )
    parser.add_argument(
        "--config",
        default="config/demo_router_config.yaml",
        help="Path to demo_router_config.yaml (used for routing policy)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write new routing_records.jsonl, eval_result.json and routing_analysis.json",
    )
    parser.add_argument(
        "--difficulty-tier-map",
        nargs="+",
        metavar="LEVEL=TIER",
        help="Override difficulty-to-tier mapping, e.g. L1=weak L2=standard L3=strong L4=expert",
    )
    parser.add_argument(
        "--disable-upward-fallback",
        action="store_true",
        help="Disable upward tier fallback to prevent simple tasks from being routed to strong models",
    )
    parser.add_argument(
        "--cost-weight",
        type=float,
        default=0.0,
        help="Cost-aware routing weight: select model maximizing quality_score - cost_weight * normalized_cost within tier",
    )
    parser.add_argument(
        "--dataset-prior-weight",
        type=float,
        default=0.0,
        help="Weight for per-dataset empirical accuracy prior from train split (0 disables)",
    )
    parser.add_argument(
        "--train-path",
        help="Path to train.jsonl for computing per-dataset model accuracy prior",
    )
    parser.add_argument(
        "--quality-margin",
        type=float,
        default=0.0,
        help="Only switch to a cheaper model if its score is within this margin of the best",
    )
    parser.add_argument(
        "--tier-bypass",
        action="store_true",
        help="Bypass tier matching and select from all candidates after hard filtering",
    )
    parser.add_argument(
        "--marginal-utility",
        action="store_true",
        help="Enable counterfactual marginal utility routing: only upgrade to stronger models when expected gain covers extra cost",
    )
    parser.add_argument(
        "--base-model",
        default="qwen3-235b-a22b-2507",
        help="Reference weak model for marginal utility calculation",
    )
    parser.add_argument(
        "--marginal-threshold",
        type=float,
        default=0.1,
        help="Minimum marginal utility (gain/cost) required to upgrade from base model",
    )
    parser.add_argument(
        "--hardness-aware",
        action="store_true",
        help="Enable hardness-aware routing: for questions where all models struggle, avoid expensive upgrades",
    )
    parser.add_argument(
        "--hard-threshold",
        type=float,
        default=0.3,
        help="Max training accuracy below which a dataset+difficulty is considered 'all models struggle'",
    )
    parser.add_argument(
        "--cheap-model",
        default="qwen3-235b-a22b-2507",
        help="Cheap model to use for hard questions when hardness-aware is enabled",
    )
    parser.add_argument(
        "--capability-aware",
        action="store_true",
        help="Enable capability-aware routing: select model based on task capability requirements rather than dataset name",
    )
    parser.add_argument(
        "--cluster-aware",
        action="store_true",
        help="Enable cluster-aware routing: use embedding clusters and actual costs (AvengersPro-style)",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=25,
        help="Number of embedding clusters for cluster-aware routing",
    )
    parser.add_argument(
        "--cluster-top-k",
        type=int,
        default=3,
        help="Number of nearest clusters to use for test query routing",
    )
    parser.add_argument(
        "--cluster-beta",
        type=float,
        default=9.0,
        help="Softmax temperature for cluster distance weighting",
    )
    parser.add_argument(
        "--dataset-model-override",
        nargs="+",
        metavar="DATASET=MODEL",
        help="Override model for specific datasets, e.g. swe-bench=gemini-2.5-pro aime=qwen3-235b-a22b-2507",
    )
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Run grid search over difficulty_tier_map, cost_weight, disable_upward_fallback, dataset_prior_weight and tier_bypass",
    )
    parser.add_argument(
        "--baseline-eval-result",
        help="Path to baseline eval_result.json (e.g. LLMSemanticRouter) for comparison",
    )
    args = parser.parse_args()

    load_dotenv(find_dotenv(usecwd=True))

    router_config = DemoRouterConfig.load(args.config)
    model_pool = ModelPool.load(args.model_pool)
    profiles = load_profiles(args.model_profiles)
    test_items = load_test_items(args.test)
    cached_records = load_routing_records(args.routing_records)
    dataset_prior = compute_dataset_prior(args.train_path) if args.train_path else {}
    granular_priors = compute_granular_priors(args.train_path) if args.train_path else {}
    capability_prior = compute_capability_prior(args.train_path) if args.train_path else {}

    cluster_priors = None
    if args.cluster_aware and args.train_path:
        cluster_priors = compute_cluster_priors(
            args.train_path,
            test_items,
            n_clusters=args.n_clusters,
            top_k=args.cluster_top_k,
            beta=args.cluster_beta,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    difficulty_tier_map = None
    if args.difficulty_tier_map:
        difficulty_tier_map = parse_difficulty_tier_map(args.difficulty_tier_map)

    dataset_model_override = None
    if args.dataset_model_override:
        dataset_model_override = {}
        for item in args.dataset_model_override:
            if "=" not in item:
                raise ValueError(f"Invalid dataset model override: {item!r}, expected DATASET=MODEL")
            ds, model = item.split("=", 1)
            dataset_model_override[ds.strip()] = model.strip()

    if args.grid_search:
        summary = run_grid_search(
            test_items,
            cached_records,
            model_pool,
            profiles,
            router_config,
            output_dir,
            baseline_path=args.baseline_eval_result,
            train_path=args.train_path,
        )
        logger.info(f"Policy search summary written to {output_dir / 'policy_search_results.json'}")
        return

    output = run_single_evaluation(
        test_items,
        cached_records,
        model_pool,
        profiles,
        router_config,
        output_dir,
        difficulty_tier_map=difficulty_tier_map,
        disable_upward_fallback=args.disable_upward_fallback,
        cost_weight=args.cost_weight,
        dataset_prior=dataset_prior,
        dataset_prior_weight=args.dataset_prior_weight,
        quality_margin=args.quality_margin,
        tier_bypass=args.tier_bypass,
        marginal_utility_mode=args.marginal_utility,
        base_model=args.base_model,
        marginal_threshold=args.marginal_threshold,
        granular_priors=granular_priors,
        hardness_aware=args.hardness_aware,
        hard_threshold=args.hard_threshold,
        cheap_model=args.cheap_model,
        dataset_model_override=dataset_model_override,
        capability_aware=args.capability_aware,
        capability_prior=capability_prior,
        cluster_aware=args.cluster_aware,
        cluster_priors=cluster_priors,
    )
    results = output["results"]
    analysis = output["analysis"]

    config_meta = {
        "test_path": args.test,
        "routing_records_path": args.routing_records,
        "model_profiles_path": args.model_profiles,
        "model_pool_path": args.model_pool,
        "demo_router_config": args.config,
        "output_dir": args.output_dir,
        "routing_mode": router_config.routing_mode,
        "difficulty_tier_map": (difficulty_tier_map or router_config.difficulty_tier_map),
        "disable_upward_fallback": args.disable_upward_fallback,
        "cost_weight": args.cost_weight,
        "dataset_prior_weight": args.dataset_prior_weight,
        "train_path": args.train_path,
        "quality_margin": args.quality_margin,
        "tier_bypass": args.tier_bypass,
        "marginal_utility": args.marginal_utility,
        "base_model": args.base_model,
        "marginal_threshold": args.marginal_threshold,
        "hardness_aware": args.hardness_aware,
        "hard_threshold": args.hard_threshold,
        "cheap_model": args.cheap_model,
        "dataset_model_override": dataset_model_override,
        "capability_aware": args.capability_aware,
        "cluster_aware": args.cluster_aware,
        "n_clusters": args.n_clusters,
        "cluster_top_k": args.cluster_top_k,
        "cluster_beta": args.cluster_beta,
    }

    eval_output = {
        "timestamp": datetime.now().isoformat(),
        "config": config_meta,
        "results": results,
        "routing_analysis": analysis,
    }

    baseline_comparison = None
    if args.baseline_eval_result:
        baseline_comparison = compare_with_baseline(
            results, args.baseline_eval_result, output["routing_records"], test_items
        )
        eval_output["baseline_comparison"] = baseline_comparison

    eval_result_path = output_dir / "eval_result.json"
    with open(eval_result_path, "w", encoding="utf-8") as f:
        json.dump(eval_output, f, indent=2, ensure_ascii=False)

    analysis_path = output_dir / "routing_analysis.json"
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    if baseline_comparison:
        comparison_path = output_dir / "baseline_comparison.json"
        with open(comparison_path, "w", encoding="utf-8") as f:
            json.dump(baseline_comparison, f, indent=2, ensure_ascii=False)

    logger.info(f"Reroute evaluation complete. Results written to {eval_result_path}")
    logger.info(f"  Dataset-Avg accuracy: {results['accuracy']}%")
    logger.info(f"  Sample-Avg accuracy: {results['all_sample_avg']}%")
    logger.info(f"  Oracle Sample-Avg: {results['oracle_sample_avg']}%")
    logger.info(f"  Total cost: ${results['cost_analysis']['total_cost']}")
    logger.info(f"  Model selection: {results['model_selection_counts']}")
    logger.info(f"  Fallback ratio: {analysis['fallback']['ratio']}%")
    logger.info(f"  Over-routing ratio (L1/L2 -> strong/expert): {analysis['over_routing']['ratio']}%")
    if baseline_comparison:
        gap = baseline_comparison["gap"]
        logger.info(f"  vs baseline: accuracy {gap['accuracy']:+.2f}%, cost {gap['cost']:+.4f}")


if __name__ == "__main__":
    main()
