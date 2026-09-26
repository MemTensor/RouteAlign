"""
离线建模流水线（方案 §2.1）：模型入池评测 → 多维打分 → 模型画像。

模型在题库上的回答有两种来源：
1. 外部提供的回答文件（answers JSONL：{"item_id", "model_id", "answer"}），
   例如适配模块从既有基准测试结果中导出的回答；
2. 通过模型池中配置的 endpoint 真正调用模型生成（generate_answers）。

打分结果会持久化为 JSONL，支持断点续跑（已打分的 (item_id, model_id) 跳过）。
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger

from .grading import AnswerGrader
from .llm_client import LLMClient, LLMEndpoint
from .model_pool import ModelPool
from .profiling import ProfileBuilder, save_profiles
from .question_bank import QuestionBank
from .schemas import GradingResult, ModelProfile


# ---------------------------------------------------------------------------
# 回答的读取与生成
# ---------------------------------------------------------------------------

def load_answers(path: str) -> Dict[Tuple[str, str], str]:
    """
    加载外部提供的回答文件。

    JSONL 格式，每行：{"item_id": str, "model_id": str, "answer": str}

    Returns:
        {(item_id, model_id): answer}
    """
    answers: Dict[Tuple[str, str], str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            answers[(data["item_id"], data["model_id"])] = data.get("answer", "")
    logger.info(f"Loaded {len(answers)} answers from {path}")
    return answers


def generate_answers(
    bank: QuestionBank,
    model_pool: ModelPool,
    model_ids: Optional[List[str]] = None,
    max_workers: int = 4,
) -> Dict[Tuple[str, str], str]:
    """
    调用模型池中各模型的 endpoint，在题库上生成回答。

    要求模型池配置中对应模型包含 endpoint 字段。
    """
    model_ids = model_ids or model_pool.model_ids()
    answers: Dict[Tuple[str, str], str] = {}

    tasks = []
    clients: Dict[str, LLMClient] = {}
    for model_id in model_ids:
        entry = model_pool.get(model_id)
        if not entry.endpoint:
            raise ValueError(
                f"Model {model_id} has no endpoint config; cannot generate answers"
            )
        clients[model_id] = LLMClient(LLMEndpoint.from_config(entry.endpoint))
        for item in bank:
            tasks.append((item.item_id, model_id, item.question))

    logger.info(f"Generating {len(tasks)} answers ({len(model_ids)} models x {len(bank)} items)")

    def run(task):
        item_id, model_id, question = task
        content = clients[model_id].chat([{"role": "user", "content": question}])
        return item_id, model_id, content

    failed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(run, t): t for t in tasks}
        for future in as_completed(futures):
            item_id, model_id, _ = futures[future]
            try:
                _, _, content = future.result()
            except Exception as e:  # noqa: BLE001
                failed += 1
                logger.error(
                    f"Answer generation failed permanently for "
                    f"({item_id}, {model_id}): {e}"
                )
                continue
            answers[(item_id, model_id)] = content

    if failed:
        logger.warning(f"Answer generation finished with {failed}/{len(tasks)} failures")
    return answers


# ---------------------------------------------------------------------------
# 批量打分（支持断点续跑）
# ---------------------------------------------------------------------------

def load_grading_results(path: str) -> List[GradingResult]:
    results = []
    p = Path(path)
    if not p.exists():
        return results
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            results.append(GradingResult.from_dict(json.loads(line)))
    return results


def grade_all(
    bank: QuestionBank,
    answers: Dict[Tuple[str, str], str],
    grader: AnswerGrader,
    output_path: str,
    max_workers: int = 8,
) -> List[GradingResult]:
    """
    对所有 (题目, 模型) 回答进行多维打分，结果追加写入 output_path（JSONL）。

    已存在于 output_path 中的 (item_id, model_id) 会被跳过（断点续跑）。
    """
    existing = load_grading_results(output_path)
    done = {(r.item_id, r.model_id) for r in existing}
    pending = [
        (item_id, model_id, answer)
        for (item_id, model_id), answer in answers.items()
        if (item_id, model_id) not in done
    ]
    logger.info(
        f"Grading: {len(done)} already done, {len(pending)} pending "
        f"(output: {output_path})"
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    results = list(existing)

    def run(task):
        item_id, model_id, answer = task
        item = bank.get(item_id)
        return grader.grade(item, model_id, answer)

    if pending:
        with open(output_path, "a", encoding="utf-8") as f:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(run, t): t for t in pending}
                completed = 0
                failed = 0
                for future in as_completed(futures):
                    item_id, model_id, _ = futures[future]
                    try:
                        result = future.result()
                    except Exception as e:  # noqa: BLE001
                        # 单任务永久失败（如重试耗尽）不拖垮整批任务；
                        # 失败项不写入文件，下次断点续跑会自动重试。
                        failed += 1
                        completed += 1
                        logger.error(
                            f"Grading failed permanently for "
                            f"({item_id}, {model_id}): {e}"
                        )
                        continue
                    results.append(result)
                    f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
                    f.flush()
                    completed += 1
                    if completed % 20 == 0 or completed == len(pending):
                        logger.info(
                            f"Grading progress: {completed}/{len(pending)} "
                            f"(failed: {failed})"
                        )
                if failed:
                    logger.warning(
                        f"Grading finished with {failed}/{len(pending)} permanent "
                        f"failures; rerun the same command to retry them"
                    )

    return results


# ---------------------------------------------------------------------------
# 完整离线流水线
# ---------------------------------------------------------------------------

def run_offline_pipeline(
    bank: QuestionBank,
    model_pool: ModelPool,
    grader: AnswerGrader,
    profile_builder: ProfileBuilder,
    grading_output_path: str,
    profiles_output_path: str,
    answers: Optional[Dict[Tuple[str, str], str]] = None,
    answers_path: Optional[str] = None,
    max_workers: int = 8,
) -> Dict[str, ModelProfile]:
    """
    执行完整离线建模：回答 → 多维打分 → 画像聚合与分档 → 持久化。

    answers 与 answers_path 二选一；均未提供时通过模型池 endpoint 生成回答。
    """
    if answers is None:
        if answers_path:
            answers = load_answers(answers_path)
        else:
            answers = generate_answers(bank, model_pool, max_workers=max_workers)

    grading_results = grade_all(
        bank, answers, grader, grading_output_path, max_workers=max_workers
    )
    profiles = profile_builder.build_profiles(bank, grading_results, model_pool)
    save_profiles(profiles, profiles_output_path)
    return profiles
