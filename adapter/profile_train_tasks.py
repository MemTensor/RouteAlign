"""
批量生成训练集 query 的任务画像（能力需求权重 + 难度等级），带断点续跑。

train.jsonl 中只有 query 与各模型实测得分（records/usages），没有任务画像。
本脚本调用任务画像生成模型（与线上路由同一个 profiler，保证训练/测试的
用户需求表征一致），为每条 train query 生成：
    capability_need / difficulty_level / tool_required

输出 JSONL 每行一条：
    {"dataset", "index", "status", "capability_need", "difficulty_level",
     "tool_required", "error"}
续跑时跳过 status=success 的样本，失败样本自动重试。

用法：
    python -m adapter.profile_train_tasks \
        --train data/seed42_split0.7/train.jsonl \
        --output data/seed42_split0.7/train_task_profiles.jsonl \
        --config config/demo_router_config.yaml \
        --max-workers 15
"""

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

from dotenv import find_dotenv, load_dotenv
from loguru import logger

from src import (
    DemoRouterConfig,
    LLMClient,
    RequestFeatures,
    TaskProfiler,
    detect_request_features,
)


def load_train_items(path: str) -> List[Dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    logger.info(f"Loaded {len(items)} train items from {path}")
    return items


def load_done_keys(output_path: Path) -> Dict[str, Dict]:
    """读取已有输出，返回 {dataset||index: record}（仅 status=success）。"""
    done = {}
    if not output_path.exists():
        return done
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("status") == "success":
                done[f"{rec['dataset']}||{rec['index']}"] = rec
    logger.info(f"Resuming: {len(done)} successful profiles already in {output_path}")
    return done


def profile_one(profiler: TaskProfiler, item: Dict) -> Dict:
    """对单条 query 生成任务画像，返回输出记录。"""
    key = f"{item['dataset']}||{item['index']}"
    rec = {
        "dataset": item["dataset"],
        "index": item["index"],
        "status": "failed",
        "capability_need": None,
        "difficulty_level": None,
        "tool_required": None,
        "error": None,
    }
    try:
        features = detect_request_features(item["query"])
        task = profiler.profile(item["query"], features)
        rec.update(
            status="success",
            capability_need=task.capability_need,
            difficulty_level=task.difficulty_level,
            tool_required=task.tool_required,
        )
    except Exception as e:  # noqa: BLE001 - 单条失败不中断整体
        rec["error"] = str(e)[:500]
        logger.warning(f"Profiling failed for {key}: {e}")
    return rec


def main():
    parser = argparse.ArgumentParser(
        description="Batch-generate task profiles for train queries (with resume)"
    )
    parser.add_argument("--train", required=True, help="Path to train.jsonl")
    parser.add_argument("--output", required=True, help="Path to output JSONL")
    parser.add_argument(
        "--config",
        default="config/demo_router_config.yaml",
        help="Path to demo_router_config.yaml (task_profile_model endpoint)",
    )
    parser.add_argument("--max-workers", type=int, default=15, help="并发数")
    parser.add_argument(
        "--limit", type=int, default=0, help="只处理前 N 条待处理样本（0 = 全部，调试用）"
    )
    args = parser.parse_args()

    load_dotenv(find_dotenv(usecwd=True))

    config = DemoRouterConfig.load(args.config)
    profiler = TaskProfiler(
        LLMClient(config.task_profile_endpoint()),
        prompt_style=config.task_profile_prompt_style(),
    )

    items = load_train_items(args.train)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_keys(output_path)

    pending = [
        it for it in items if f"{it['dataset']}||{it['index']}" not in done
    ]
    if args.limit > 0:
        pending = pending[: args.limit]
    logger.info(f"Pending: {len(pending)} / {len(items)} items")

    if not pending:
        logger.info("Nothing to do.")
        return

    lock = threading.Lock()
    n_ok = 0
    n_fail = 0
    with open(output_path, "a", encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futures = {
                pool.submit(profile_one, profiler, item): item for item in pending
            }
            for i, fut in enumerate(as_completed(futures), 1):
                rec = fut.result()
                with lock:
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out.flush()
                    if rec["status"] == "success":
                        n_ok += 1
                    else:
                        n_fail += 1
                    if i % 100 == 0 or i == len(pending):
                        logger.info(
                            f"Progress: {i}/{len(pending)} "
                            f"(success={n_ok}, failed={n_fail})"
                        )

    logger.info(f"Done. success={n_ok}, failed={n_fail}. Output: {output_path}")


if __name__ == "__main__":
    main()
