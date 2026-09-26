"""
kNN 局部先验：每道 query 的逐题级期望正确率/成本。

针对统一打分器的核心缺陷（D1 信号全是条件均值、D2 簇硬分配损失局部性）：
不再把 query 分进 ~350 题的簇格子共享先验，而是为每道 query 在嵌入空间中
找 k 个最近的训练邻居，按余弦距离 softmax 加权，统计各模型在这些邻居上的
实测正确率与实测成本，得到严格逐题的局部先验 P_knn(m|x)。

- train-LOO：评估集即训练集时排除自身邻居，防止自匹配泄漏（用于训练集上调参）
- 嵌入缓存：.npy 按路径缓存，避免每次运行重算
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from adapter.evaluate_with_cached_tasks import (
    _load_embedding_encoder,
)


def load_or_compute_embeddings(
    items: List[Dict],
    cache_path: Optional[str],
    embedding_model_path: str = "models/sentence-transformers/all-MiniLM-L6-v2",
) -> np.ndarray:
    """加载或计算 L2 归一化嵌入，形状 (n, dim)。"""
    if cache_path and Path(cache_path).exists():
        emb = np.load(cache_path)
        if emb.shape[0] == len(items):
            logger.info(f"Loaded embeddings from {cache_path} ({emb.shape})")
            return emb
        logger.warning(
            f"Embedding cache {cache_path} size mismatch "
            f"({emb.shape[0]} != {len(items)}), recomputing"
        )
    queries = [item["query"] for item in items]
    logger.info(f"Computing embeddings for {len(queries)} queries...")
    encoder = _load_embedding_encoder(embedding_model_path)
    emb = np.array(encoder.encode(queries, show_progress_bar=False, batch_size=64))
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, emb)
        logger.info(f"Cached embeddings to {cache_path}")
    return emb


def compute_knn_priors(
    train_items: List[Dict],
    train_embeddings: np.ndarray,
    eval_embeddings: np.ndarray,
    k: int = 50,
    beta: float = 8.0,
    loo: bool = False,
    train_cap_weights: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """计算每道评估 query 的 kNN 局部先验。

    Args:
        loo: 评估集即训练集时为 True，排除自身作为邻居。
        train_cap_weights: 可选，(n_train, n_caps) 每条训练 query 的能力需求权重
            （来自其任务画像）。提供时额外输出能力条件化局部画像：
            cap_acc[e, c, m] = Σ_j γ_j·w_{j,c}·s_{m,j} / Σ_j γ_j·w_{j,c}，
            即邻居按"考察维度 c 的程度"加权后模型 m 的局部经验正确率；
            无邻居覆盖的维度回退到标量局部先验 acc[e, m]。

    Returns:
        {
            "models": [model_id, ...],
            "acc": np.ndarray (n_eval, n_models),        # 局部期望正确率
            "cost": np.ndarray (n_eval, n_models),       # 局部期望实测成本
            "cap_acc": np.ndarray (n_eval, n_caps, n_models),  # 可选
        }
    """
    models = sorted({m for it in train_items for m in it.get("records", {})})
    midx = {m: i for i, m in enumerate(models)}
    n_train = len(train_items)
    score_mat = np.zeros((n_train, len(models)))
    cost_mat = np.zeros((n_train, len(models)))
    for i, it in enumerate(train_items):
        for m, s in it.get("records", {}).items():
            score_mat[i, midx[m]] = float(s)
            usage = (it.get("usages") or {}).get(m) or {}
            cost_mat[i, midx[m]] = float(usage.get("cost") or 0.0)

    k_eff = min(k, n_train)
    sim = eval_embeddings @ train_embeddings.T  # 已归一化，内积即余弦相似度
    if loo:
        # 自身相似度置为 -inf，保证不会入选邻居（评估集即训练集时防自匹配泄漏）
        np.fill_diagonal(sim, -np.inf)

    idx = np.argpartition(-sim, k_eff, axis=1)[:, :k_eff]
    dist = 1.0 - np.take_along_axis(sim, idx, 1)  # 余弦距离
    weights = np.exp(-beta * dist)
    weights[~np.isfinite(dist)] = 0.0  # 防御：极端情况下剔除无效邻居
    weights /= weights.sum(axis=1, keepdims=True)

    acc = np.zeros((len(eval_embeddings), len(models)))
    cost = np.zeros((len(eval_embeddings), len(models)))
    for j in range(weights.shape[1]):
        nbr = idx[:, j]
        w = weights[:, j : j + 1]
        acc += w * score_mat[nbr]
        cost += w * cost_mat[nbr]

    result: Dict[str, Any] = {"models": models, "acc": acc, "cost": cost}

    # 能力条件化局部画像：邻居按其任务画像中维度 c 的考察权重加权
    if train_cap_weights is not None:
        assert train_cap_weights.shape[0] == n_train
        n_caps = train_cap_weights.shape[1]
        w_nb = train_cap_weights[idx]  # (n_eval, k, n_caps)
        s_nb = score_mat[idx]  # (n_eval, k, n_models)
        cap_acc = np.zeros((len(eval_embeddings), n_caps, len(models)))
        for c in range(n_caps):
            gwc = weights * w_nb[:, :, c]  # (n_eval, k)
            den = gwc.sum(axis=1)  # (n_eval,)
            num = np.einsum("ej,ejm->em", gwc, s_nb)  # (n_eval, n_models)
            ok = den > 1e-6
            cap_acc[:, c, :] = np.where(
                ok[:, None], num / np.maximum(den, 1e-9)[:, None], acc
            )
        result["cap_acc"] = cap_acc
        logger.info(f"kNN capability-conditioned priors added: n_caps={n_caps}")

    logger.info(
        f"kNN priors computed: k={k}, beta={beta}, loo={loo}, "
        f"n_eval={len(eval_embeddings)}, n_train={n_train}"
    )
    return result
