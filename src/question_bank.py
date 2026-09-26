"""
动态评测题库的读取与使用。

题库采用「输入题库」即插即用设计：Demo Router 不负责生成或处理题库，
只读取外部（例如适配模块）已经准备好的、元数据与能力权重完整的题库文件。

题库文件格式为 JSONL，每行一个 QuestionItem（见 schemas.QuestionItem）。
"""

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from loguru import logger

from .schemas import QuestionItem


class QuestionBank:
    """评测题库：只读加载，按难度/能力/题库类型索引。"""

    def __init__(self, items: List[QuestionItem]):
        self.items = items
        self._by_id: Dict[str, QuestionItem] = {}
        for item in items:
            if item.item_id in self._by_id:
                raise ValueError(f"Duplicate item_id in question bank: {item.item_id}")
            self._by_id[item.item_id] = item

    @classmethod
    def load(cls, path: str, bank_types: Optional[Iterable[str]] = None) -> "QuestionBank":
        """
        从 JSONL 文件加载题库。

        Args:
            path: 题库 JSONL 文件路径
            bank_types: 可选过滤，只加载指定题库类型（core/dynamic/hidden）
        """
        bank_path = Path(path)
        if not bank_path.exists():
            raise FileNotFoundError(f"Question bank file not found: {path}")

        allowed = set(bank_types) if bank_types else None
        items: List[QuestionItem] = []
        with open(bank_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    item = QuestionItem.from_dict(data)
                except Exception as e:
                    raise ValueError(
                        f"Invalid question bank line {line_no} in {path}: {e}"
                    ) from e
                if allowed and item.bank_type not in allowed:
                    continue
                items.append(item)

        bank = cls(items)
        logger.info(
            f"Loaded question bank from {path}: {len(items)} items "
            f"(difficulty: {bank.count_by_difficulty()})"
        )
        return bank

    def get(self, item_id: str) -> QuestionItem:
        return self._by_id[item_id]

    def by_difficulty(self, level: str) -> List[QuestionItem]:
        return [it for it in self.items if it.difficulty_level == level]

    def count_by_difficulty(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for it in self.items:
            counts[it.difficulty_level] = counts.get(it.difficulty_level, 0) + 1
        return dict(sorted(counts.items()))

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)
