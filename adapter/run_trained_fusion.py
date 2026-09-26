"""
微调任务画像模型的「学习式融合路由」批量评估（DemoRouter3 最佳性能代码路径）。

对每个注册的微调模型依次执行：
1. profiles：用该模型（sglang 服务）为训练集 8710 条 query 生成任务画像
   → task_profiles/train_task_profiles_<tag>.jsonl（断点续跑）
2. build：用训练集实测得分 + 该模型的训练任务画像构建数据驱动模型画像
   → model_profiles_data_driven_<tag>.json
3. fusion：运行 evaluate_learned_fusion（岭回归融合，与 seed43_fusion_final
   同配置：knn_k=150, knn_beta=5.0, cost_lambda=0.4, ridge_alpha=1.0, knn_cap）
   → learned_fusion/<tag>/{eval_result.json, routing_records.jsonl, ...}

嵌入（kNN 用，与任务画像模型无关）只计算一次并缓存为 emb_train.npy / emb_test.npy。

用法（项目根目录）：
    # 全流程（需要 sglang 服务已启动：scripts/serve_trained_models.sh start）
    python -m adapter.run_trained_fusion

    # 只跑指定模型 / 只跑某个阶段
    python -m adapter.run_trained_fusion \
        --models router_21K_Qwen3-0.6B --stage fusion

    # 只汇总已有结果
    python -m adapter.run_trained_fusion --summary-only
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # repository root
PKG_ROOT = Path(__file__).resolve().parents[1]  # repository root
REGISTRY_PATH = PKG_ROOT / "config" / "trained_models.yaml"
DATA_DIR = PKG_ROOT / "data" / "seed42_split0.7"

# 与 DemoRouter3 最佳结果（seed43_fusion_final）一致的融合超参
FUSION_KNN_K = 150
FUSION_KNN_BETA = 5.0
FUSION_COST_LAMBDA = 0.4
FUSION_RIDGE_ALPHA = 1.0

TRAIN_JSONL = DATA_DIR / "train.jsonl"
TEST_JSONL = DATA_DIR / "test.jsonl"
JUDGE_PROFILES = DATA_DIR / "model_profiles.json"
MODEL_POOL = PKG_ROOT / "config" / "model_pool.yaml"
DEMO_ROUTER_CONFIG = PKG_ROOT / "config" / "demo_router_config.yaml"
EMB_TRAIN_CACHE = DATA_DIR / "emb_train.npy"
EMB_TEST_CACHE = DATA_DIR / "emb_test.npy"
PROFILE_WORKERS = 32


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_step(cmd: list, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        return subprocess.run(
            cmd, cwd=PROJECT_ROOT, stdout=f, stderr=subprocess.STDOUT
        ).returncode


def wait_healthy(port: int, timeout_s: int = 120) -> bool:
    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(5)
    return False


def stage_profiles(name: str, port: int, log_dir: Path) -> int:
    """该模型为训练集生成任务画像（断点续跑）。"""
    out = DATA_DIR / "task_profiles" / f"train_task_profiles_{name}.jsonl"
    demo_cfg = PKG_ROOT / "config" / "trained" / f"demo_router_{name}.yaml"
    if not demo_cfg.exists():
        from adapter.run_trained_eval import (
            generate_model_configs,
        )

        registry = yaml.safe_load(open(REGISTRY_PATH, encoding="utf-8"))
        generate_model_configs(name, registry["models"][name], registry, None)
    return run_step(
        [
            sys.executable, "-m",
            "adapter.profile_train_tasks",
            "--train", str(TRAIN_JSONL.relative_to(PROJECT_ROOT)),
            "--output", str(out.relative_to(PROJECT_ROOT)),
            "--config", str(demo_cfg.relative_to(PROJECT_ROOT)),
            "--max-workers", str(PROFILE_WORKERS),
        ],
        log_dir / f"profiles_{name}.log",
    )


def stage_build(name: str, log_dir: Path) -> int:
    """数据驱动模型画像。"""
    out = DATA_DIR / f"model_profiles_data_driven_{name}.json"
    tp = DATA_DIR / "task_profiles" / f"train_task_profiles_{name}.jsonl"
    return run_step(
        [
            sys.executable, "-m",
            "adapter.build_profile_from_records",
            "--train", str(TRAIN_JSONL.relative_to(PROJECT_ROOT)),
            "--task-profiles", str(tp.relative_to(PROJECT_ROOT)),
            "--model-pool", str(MODEL_POOL.relative_to(PROJECT_ROOT)),
            "--output", str(out.relative_to(PROJECT_ROOT)),
        ],
        log_dir / f"build_{name}.log",
    )


def stage_fusion(name: str, log_dir: Path) -> int:
    """岭回归融合评估（seed43_fusion_final 同配置）。"""
    out_dir = DATA_DIR / "learned_fusion" / name
    out_dir.mkdir(parents=True, exist_ok=True)  # fusion_weights.json 先于脚本内 mkdir 写出
    tp = DATA_DIR / "task_profiles" / f"train_task_profiles_{name}.jsonl"
    return run_step(
        [
            sys.executable, "-m",
            "adapter.evaluate_learned_fusion",
            "--test", str(TEST_JSONL.relative_to(PROJECT_ROOT)),
            "--routing-records",
            str((DATA_DIR / f"routing_records_{name}.jsonl").relative_to(PROJECT_ROOT)),
            "--model-profiles",
            str((DATA_DIR / f"model_profiles_data_driven_{name}.json").relative_to(PROJECT_ROOT)),
            "--judge-profiles", str(JUDGE_PROFILES.relative_to(PROJECT_ROOT)),
            "--model-pool", str(MODEL_POOL.relative_to(PROJECT_ROOT)),
            "--config", str(DEMO_ROUTER_CONFIG.relative_to(PROJECT_ROOT)),
            "--train-path", str(TRAIN_JSONL.relative_to(PROJECT_ROOT)),
            "--train-routing-records", str(tp.relative_to(PROJECT_ROOT)),
            "--train-task-profiles", str(tp.relative_to(PROJECT_ROOT)),
            "--output-dir", str(out_dir.relative_to(PROJECT_ROOT)),
            "--knn-k", str(FUSION_KNN_K),
            "--knn-beta", str(FUSION_KNN_BETA),
            "--knn-cap",
            "--ridge-alpha", str(FUSION_RIDGE_ALPHA),
            "--cost-lambda", str(FUSION_COST_LAMBDA),
            "--fusion-weights-out",
            str((out_dir / "fusion_weights.json").relative_to(PROJECT_ROOT)),
            "--train-emb-cache", str(EMB_TRAIN_CACHE.relative_to(PROJECT_ROOT)),
            "--eval-emb-cache", str(EMB_TEST_CACHE.relative_to(PROJECT_ROOT)),
        ],
        log_dir / f"fusion_{name}.log",
    )


def ensure_embeddings() -> None:
    """kNN 嵌入与任务画像模型无关，只算一次并缓存。"""
    from adapter.evaluate_with_cached_tasks import (
        load_test_items,
    )
    from adapter.knn_priors import (
        load_or_compute_embeddings,
    )

    if not (EMB_TRAIN_CACHE.exists() and EMB_TEST_CACHE.exists()):
        log("计算 kNN 嵌入（仅首次，后续走缓存）...")
        train_items = load_test_items(str(TRAIN_JSONL))
        test_items = load_test_items(str(TEST_JSONL))
        load_or_compute_embeddings(train_items, str(EMB_TRAIN_CACHE))
        load_or_compute_embeddings(test_items, str(EMB_TEST_CACHE))
    log(f"嵌入缓存就绪: {EMB_TRAIN_CACHE.name}, {EMB_TEST_CACHE.name}")


def summarize(models: list) -> None:
    rows = []
    for name in models:
        path = DATA_DIR / "learned_fusion" / name / "eval_result.json"
        if not path.exists():
            rows.append((name, None))
            continue
        with open(path, "r", encoding="utf-8") as f:
            rows.append((name, json.load(f)["results"]))
    header = f"{'model':<26} {'DatasetAvg':>10} {'SampleAvg':>10} {'Oracle':>8} {'Cost($)':>9} {'#queries':>9}"
    print("\n" + header)
    print("-" * len(header))
    for name, r in rows:
        if r is None:
            print(f"{name:<26} {'(无结果)':>10}")
            continue
        print(
            f"{name:<26} {r['accuracy']:>10.2f} {r['all_sample_avg']:>10.2f} "
            f"{r['oracle_sample_avg']:>8.2f} "
            f"{r['cost_analysis']['total_cost']:>9.2f} {r['total_queries']:>9d}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Learned-fusion routing evaluation for trained task-profile models"
    )
    parser.add_argument("--models", nargs="*", default=None, help="默认注册表全部")
    parser.add_argument(
        "--stage",
        choices=["all", "profiles", "build", "fusion"],
        default="all",
        help="只跑某个阶段（默认 all = profiles→build→fusion）",
    )
    parser.add_argument(
        "--summary-only", action="store_true", help="只汇总已有结果"
    )
    args = parser.parse_args()

    registry = yaml.safe_load(open(REGISTRY_PATH, encoding="utf-8"))
    all_models = registry["models"]
    selected = args.models or list(all_models)
    for name in selected:
        if name not in all_models:
            parser.error(f"未知模型: {name}（可选：{list(all_models)}）")

    if args.summary_only:
        summarize(selected)
        return

    log_dir = DATA_DIR / "trained_eval_logs"
    stages = ["profiles", "build", "fusion"] if args.stage == "all" else [args.stage]

    # profiles 阶段需要 sglang 服务
    if "profiles" in stages:
        alive = []
        for name in selected:
            port = all_models[name]["port"]
            if wait_healthy(port, timeout_s=1800):
                alive.append(name)
                log(f"[health] {name} (:{port}) 就绪")
            else:
                log(f"[error] {name} 服务未就绪，跳过")
        selected = alive
        if not selected:
            sys.exit(1)

    # fusion 阶段需要嵌入缓存（模型无关，只算一次）
    if "fusion" in stages:
        ensure_embeddings()

    def run_model(name: str):
        port = all_models[name]["port"]
        for st in stages:
            t0 = time.time()
            if st == "profiles":
                rc = stage_profiles(name, port, log_dir)
            elif st == "build":
                rc = stage_build(name, log_dir)
            else:
                rc = stage_fusion(name, log_dir)
            dt = time.time() - t0
            log(f"[{st}] {name}: {'OK' if rc == 0 else f'FAILED(rc={rc})'} ({dt:.0f}s)")
            if rc != 0:
                return name, st, rc
        return name, None, 0

    with ThreadPoolExecutor(max_workers=len(selected)) as pool:
        results = list(pool.map(run_model, selected))

    failed = [(n, st) for n, st, rc in results if rc != 0]
    if failed:
        log(f"失败: {failed}（日志见 {log_dir}）")

    summarize(selected)


if __name__ == "__main__":
    main()
