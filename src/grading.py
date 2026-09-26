"""
多维能力打分机制（方案 §5）。

打分模型对「模型回答」在 10 个能力维度上分别输出相关性和表现得分。
"""

from typing import Optional

from loguru import logger

from .capabilities import normalize_capability_key
from .llm_client import LLMClient
from .prompts import build_grading_messages
from .schemas import CapabilityGrade, GradingResult, QuestionItem


class AnswerGrader:
    """使用打分模型对模型回答进行多维能力打分。"""

    def __init__(self, client: LLMClient, max_answer_chars: int = 30000):
        self.client = client
        self.max_answer_chars = max_answer_chars

    def grade(
        self,
        item: QuestionItem,
        model_id: str,
        model_answer: Optional[str],
    ) -> GradingResult:
        """
        对模型 model_id 在题目 item 上的回答打分。

        空回答直接判定为 invalid，不调用打分模型。
        """
        if model_answer is None or not str(model_answer).strip():
            return GradingResult(
                item_id=item.item_id,
                model_id=model_id,
                validity="invalid",
                capability_list={},
                brief_rationale="空回答，判定为无效。",
            )

        answer_text = str(model_answer)
        if len(answer_text) > self.max_answer_chars:
            answer_text = (
                answer_text[: self.max_answer_chars]
                + "\n...（回答过长，已截断）"
            )

        messages = build_grading_messages(
            question=item.question,
            reference_answer=item.reference_answer,
            rubric=item.rubric,
            model_answer=answer_text,
        )
        raw = self.client.chat_json(messages)

        capability_list = {}
        for raw_key, entry in (raw.get("capability_list") or {}).items():
            field = normalize_capability_key(raw_key)
            if not field or not isinstance(entry, dict):
                continue
            try:
                relevance = float(entry.get("relevance", 0.0))
                score = float(entry.get("scores", entry.get("score", -1)))
            except (TypeError, ValueError):
                continue
            relevance = max(0.0, min(1.0, relevance))
            capability_list[field] = CapabilityGrade(relevance=relevance, score=score)

        validity = str(raw.get("validity", "invalid")).strip().lower()
        if validity not in ("valid", "invalid"):
            validity = "invalid"

        result = GradingResult(
            item_id=item.item_id,
            model_id=model_id,
            validity=validity,
            capability_list=capability_list,
            brief_rationale=str(raw.get("brief_rationale", "")),
        )
        logger.debug(
            f"Graded {model_id} on {item.item_id}: validity={validity}, "
            f"{len(capability_list)} capability grades"
        )
        return result
