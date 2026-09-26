"""
微调任务画像模型（router_21K / grpo_router 系列）的测试集路由批量评估。

流程：
1. 按 config/trained_models.yaml 注册表，为每个模型生成
   config/trained/demo_router_<name>.yaml 与 config/trained/adapter_<name>.yaml
   （仅改任务画像端点、prompt_style 与 eval_tag，其余继承基础配置）。
2. 健康检查各模型的 sglang 服务（由 scripts/serve_trained_models.sh 启动）。
3. 并行运行 adapter.evaluate（每模型一个子进程，断点续跑由各模型自己的
   routing_records_<tag>.jsonl 保证），日志写入
   data/seed42_split0.7/trained_eval_logs/<name>.log。
4. 全部结束后汇总各模型 eval_result_<tag>.json，打印对比表。

用法（项目根目录，conda 基础环境即可）：
    # 冒烟：每数据集抽 20 题，只跑 0.6B 两个模型
    python -m adapter.run_trained_eval \
        --models router_21K_Qwen3-0.6B grpo_router_Qwen3-0.6B \
        --max-samples-per-dataset 20

    # 全量：6 个模型 × 全量测试集
    python -m adapter.run_trained_eval

    # 只汇总已有结果
    python -m adapter.run_trained_eval --summary-only
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # repository root
PKG_ROOT = Path(__file__).resolve().parents[1]  # repository root
REGISTRY_PATH = PKG_ROOT / "config" / "trained_models.yaml"
BASE_DEMO_ROUTER_CONFIG = PKG_ROOT / "config" / "demo_router_config.yaml"
BASE_ADAPTER_CONFIG = PKG_ROOT / "config" / "adapter_config.yaml"
TRAINED_CONFIG_DIR = PKG_ROOT / "config" / "trained"


def load_registry() -> dict:
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_model_configs(
    name: str,
    model_cfg: dict,
    registry: dict,
    max_samples_per_dataset,
) -> Path:
    """基于基础配置生成某模型的 adapter 配置（demo_router 端点内联其中）。

    返回 adapter 配置路径。demo_router 配置同样落盘，便于人工核查与单独复跑。
    """
    TRAINED_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    profile = registry.get("profile") or {}

    demo_router = yaml.safe_load(open(BASE_DEMO_ROUTER_CONFIG, encoding="utf-8"))
    demo_router["demo_router"]["task_profile_model"] = {
        "model": name,
        "base_url": f"http://127.0.0.1:{model_cfg['port']}/v1",
        "api_key": "none",
        "temperature": float(profile.get("temperature", 0.0)),
        "timeout": int(profile.get("timeout", 600)),
        "max_tokens": int(profile.get("max_tokens", 1024)),
        "prompt_style": profile.get("prompt_style", "sft_v3"),
    }
    demo_router_path = TRAINED_CONFIG_DIR / f"demo_router_{name}.yaml"
    with open(demo_router_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(demo_router, f, allow_unicode=True, sort_keys=False)

    adapter = yaml.safe_load(open(BASE_ADAPTER_CONFIG, encoding="utf-8"))
    adapter["adapter"]["demo_router_config"] = str(
        demo_router_path.relative_to(PROJECT_ROOT)
    )
    adapter["adapter"].setdefault("evaluation", {})["eval_tag"] = name
    if max_samples_per_dataset:
        adapter["adapter"]["evaluation"]["max_samples_per_dataset"] = int(
            max_samples_per_dataset
        )
    adapter_path = TRAINED_CONFIG_DIR / f"adapter_{name}.yaml"
    with open(adapter_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(adapter, f, allow_unicode=True, sort_keys=False)

    return adapter_path


def wait_healthy(port: int, timeout_s: int) -> bool:
    """轮询 /v1/models 直到服务就绪或超时。"""
    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except Exception:  # noqa: BLE001 - 服务未就绪，继续等
            pass
        time.sleep(10)
    return False


def run_one(model_name: str, adapter_config: Path, log_dir: Path) -> tuple:
    """运行单个模型的评估子进程，返回 (model_name, returncode)。"""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{model_name}.log"
    with open(log_path, "w", encoding="utf-8") as log_f:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "adapter.evaluate",
                "--config",
                str(adapter_config.relative_to(PROJECT_ROOT)),
            ],
            cwd=PROJECT_ROOT,
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )
        returncode = proc.wait()
    return model_name, returncode


def summarize(models: list, output_dir: Path) -> None:
    """读取各模型 eval_result_<tag>.json，打印对比表。"""
    rows = []
    for name in models:
        path = output_dir / f"eval_result_{name}.json"
        if not path.exists():
            rows.append((name, None))
            continue
        with open(path, "r", encoding="utf-8") as f:
            r = json.load(f)["results"]
        rows.append((name, r))

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
        description="Batch-evaluate trained task-profile models on the test set"
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="只跑指定模型（默认注册表中全部）",
    )
    parser.add_argument(
        "--max-samples-per-dataset",
        type=int,
        default=None,
        help="每数据集抽样评估题数（冒烟用；默认全量）",
    )
    parser.add_argument(
        "--health-timeout",
        type=int,
        default=1800,
        help="等待 sglang 服务就绪的秒数（默认 1800）",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="串行运行各模型评估（默认并行）",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="不跑评估，只汇总已有 eval_result_<tag>.json",
    )
    args = parser.parse_args()

    registry = load_registry()
    all_models = registry["models"]
    selected = args.models or list(all_models)
    for name in selected:
        if name not in all_models:
            parser.error(f"未知模型: {name}（可选：{list(all_models)}）")

    base_adapter = yaml.safe_load(open(BASE_ADAPTER_CONFIG, encoding="utf-8"))
    output_dir = PROJECT_ROOT / base_adapter["adapter"]["output_dir"]
    log_dir = output_dir / "trained_eval_logs"

    if args.summary_only:
        summarize(selected, output_dir)
        return

    # 1. 生成配置
    TRAINED_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    adapter_configs = {}
    for name in selected:
        adapter_configs[name] = generate_model_configs(
            name, all_models[name], registry, args.max_samples_per_dataset
        )
    print(f"[config] 已生成 {len(selected)} 套配置到 {TRAINED_CONFIG_DIR}")

    # 2. 健康检查
    for name in selected:
        port = all_models[name]["port"]
        print(f"[health] 等待 {name} (:{port}) 就绪 ...", flush=True)
        if not wait_healthy(port, args.health_timeout):
            print(f"[error] {name} 服务在 {args.health_timeout}s 内未就绪，跳过", flush=True)
            selected.remove(name)
    if not selected:
        print("[error] 没有可用服务，退出", flush=True)
        sys.exit(1)

    # 3. 并行评估
    print(f"[eval] 开始评估 {len(selected)} 个模型，日志: {log_dir}", flush=True)
    if args.sequential:
        results = [run_one(n, adapter_configs[n], log_dir) for n in selected]
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=len(selected)) as pool:
            results = list(
                pool.map(lambda n: run_one(n, adapter_configs[n], log_dir), selected)
            )
    for name, rc in results:
        status = "OK" if rc == 0 else f"FAILED(rc={rc})"
        print(f"[done] {name}: {status}（日志 {log_dir / (name + '.log')}）", flush=True)

    # 4. 汇总
    summarize(selected, output_dir)


if __name__ == "__main__":
    main()
