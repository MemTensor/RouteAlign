"""
线上任务画像生成（方案 §7）。

任务画像生成模型预测：能力需求权重分布、难度等级、工具需求。
"""

import json

from loguru import logger

from .llm_client import LLMClient
from .prompts import build_task_profile_messages, build_task_profile_messages_sft
from .schemas import RequestFeatures, TaskProfile


class TaskProfiler:
    """调用任务画像生成模型，输出任务需求画像。

    prompt_style：
    - "default"：通用提示词（含系统自动检测信息，输出含 tool_required）。
    - "sft_v3"：微调任务画像模型（router_21K / grpo_router 系列）专用，
      与训练数据 task-profile-sft-v3 逐字一致的提示词，不附带检测信息，
      输出仅含 capability_need 与 difficulty_level（tool_required 缺省 False）。
    """

    def __init__(
        self,
        client: LLMClient,
        max_query_chars: int = 20000,
        prompt_style: str = "default",
    ):
        if prompt_style not in ("default", "sft_v3"):
            raise ValueError(f"Unknown task profile prompt_style: {prompt_style}")
        self.client = client
        self.max_query_chars = max_query_chars
        self.prompt_style = prompt_style

    def profile(self, query: str, features: RequestFeatures) -> TaskProfile:
        query_text = query
        if len(query_text) > self.max_query_chars:
            query_text = (
                query_text[: self.max_query_chars]
                + "\n...（输入过长，已截断，完整长度见自动检测信息）"
            )

        if self.prompt_style == "sft_v3":
            messages = build_task_profile_messages_sft(query=query_text)
        else:
            messages = build_task_profile_messages(
                query=query_text,
                auto_features_json=json.dumps(
                    features.to_dict(), ensure_ascii=False, indent=2
                ),
            )
        raw = self.client.chat_json(messages)
        profile = TaskProfile.from_dict(raw)

        # 归一化能力需求权重
        total = sum(profile.capability_need.values())
        if total > 0:
            profile.capability_need = {
                k: v / total for k, v in profile.capability_need.items() if v > 0
            }
        else:
            logger.warning("Task profile has empty capability_need; defaulting to instruction_following")
            profile.capability_need = {"instruction_following": 1.0}

        return profile
