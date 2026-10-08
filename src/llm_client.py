"""
RouteAlign 统一 LLM 调用模块。

所有 RouteAlign 内部对大模型的调用（打分模型、任务画像模型、最终回答模型）
均通过本模块进行，内置指数退避重试机制（最多 10 次）。

注意：本模块仅供 RouteAlign 使用，外部适配模块有自己独立的 LLM 调用实现。
"""

import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loguru import logger
from openai import OpenAI

MAX_RETRIES = 10
BASE_DELAY_SECONDS = 1.0
MAX_DELAY_SECONDS = 60.0


@dataclass
class LLMEndpoint:
    """一个 OpenAI 兼容的 LLM 服务端点配置。"""

    model: str
    base_url: str
    api_key: str
    temperature: float = 0.2
    max_tokens: Optional[int] = None
    timeout: float = 600.0
    extra_body: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "LLMEndpoint":
        """
        从配置 dict 构建端点。api_key / base_url 支持环境变量引用：
        若值为全大写且含下划线的字符串（如 "GRADER_API_KEY"），则从环境变量读取。
        """
        def resolve(value: str) -> str:
            if isinstance(value, str) and re.fullmatch(r"[A-Z][A-Z0-9_]+", value):
                resolved = os.environ.get(value)
                if resolved:
                    return resolved
            return value

        return cls(
            model=cfg["model"],
            base_url=resolve(cfg["base_url"]),
            api_key=resolve(cfg.get("api_key", "none")),
            temperature=float(cfg.get("temperature", 0.2)),
            max_tokens=cfg.get("max_tokens"),
            timeout=float(cfg.get("timeout", 600.0)),
            extra_body=dict(cfg.get("extra_body") or {}),
        )


class LLMClient:
    """带指数退避重试的 LLM 客户端。"""

    def __init__(self, endpoint: LLMEndpoint, max_retries: Optional[int] = None):
        self.endpoint = endpoint
        # 默认沿用离线批处理的全局重试策略；在线服务可传入更小值以快速失败兜底
        self._max_retries = max_retries if max_retries is not None else MAX_RETRIES
        self._client = OpenAI(
            base_url=endpoint.base_url,
            api_key=endpoint.api_key,
            timeout=endpoint.timeout,
            max_retries=0,  # 重试由本模块统一控制
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """发起 chat completion，指数退避重试最多 MAX_RETRIES 次。"""
        last_error: Optional[Exception] = None
        for attempt in range(self._max_retries):
            try:
                kwargs: Dict[str, Any] = {
                    "model": self.endpoint.model,
                    "messages": messages,
                    "temperature": (
                        temperature if temperature is not None
                        else self.endpoint.temperature
                    ),
                }
                mt = max_tokens if max_tokens is not None else self.endpoint.max_tokens
                if mt is not None:
                    kwargs["max_tokens"] = mt
                if self.endpoint.extra_body:
                    kwargs["extra_body"] = self.endpoint.extra_body

                response = self._client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                if content is None or not content.strip():
                    raise ValueError("LLM returned empty content")
                return content
            except Exception as e:  # noqa: BLE001 - 任何异常均触发重试
                last_error = e
                delay = min(
                    BASE_DELAY_SECONDS * (2 ** attempt), MAX_DELAY_SECONDS
                ) * (1 + random.random() * 0.1)
                logger.warning(
                    f"LLM call failed (model={self.endpoint.model}, "
                    f"attempt {attempt + 1}/{self._max_retries}): {e}. "
                    f"Retrying in {delay:.1f}s"
                )
                if attempt < self._max_retries - 1:
                    time.sleep(delay)
        raise RuntimeError(
            f"LLM call failed after {self._max_retries} attempts "
            f"(model={self.endpoint.model}): {last_error}"
        )

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        发起 chat completion 并解析 JSON 输出。
        JSON 解析失败同样计入重试（整体最多 MAX_RETRIES 次）。
        """
        last_error: Optional[Exception] = None
        for attempt in range(self._max_retries):
            try:
                content = self.chat(messages, temperature=temperature, max_tokens=max_tokens)
                return extract_json(content)
            except RuntimeError:
                # chat() 内部已经耗尽重试，直接向上抛出
                raise
            except Exception as e:  # JSON 解析失败，重试
                last_error = e
                delay = min(
                    BASE_DELAY_SECONDS * (2 ** attempt), MAX_DELAY_SECONDS
                ) * (1 + random.random() * 0.1)
                logger.warning(
                    f"LLM JSON parse failed (model={self.endpoint.model}, "
                    f"attempt {attempt + 1}/{self._max_retries}): {e}. "
                    f"Retrying in {delay:.1f}s"
                )
                if attempt < self._max_retries - 1:
                    time.sleep(delay)
        raise RuntimeError(
            f"LLM JSON call failed after {self._max_retries} attempts "
            f"(model={self.endpoint.model}): {last_error}"
        )


def _strip_json_comments(text: str) -> str:
    """去掉字符串外的 // 与 # 行内注释（部分模型会往 JSON 里塞注释）。"""
    out = []
    in_string = False
    escape = False
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if in_string:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue
        if ch == "#" or (ch == "/" and i + 1 < n and text[i + 1] == "/"):
            # 跳到行尾
            while i < n and text[i] != "\n":
                i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _repair_json(candidate: str) -> str:
    """修复常见的非标准 JSON 输出：行内注释、尾逗号。"""
    repaired = _strip_json_comments(candidate)
    # 去掉 } 或 ] 前的尾逗号
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    return repaired


def extract_json(text: str) -> Dict[str, Any]:
    """
    从模型输出中提取 JSON 对象。

    容忍 ```json 代码块、思维链前后缀、行内注释、尾逗号等常见格式问题；
    先尝试原文解析，失败后再做修复解析。
    """
    text = text.strip()
    # 去掉 <think> 思维链
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    candidates = []
    # 优先 ```json ... ``` 代码块
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence:
        candidates.append(fence.group(1))
    # 全文
    candidates.append(text)
    # 第一个 { 到最后一个 } 的最大范围
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])

    last_error: Optional[Exception] = None
    for candidate in candidates:
        for attempt_text in (candidate, _repair_json(candidate)):
            try:
                result = json.loads(attempt_text)
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError as e:
                last_error = e

    raise ValueError(
        f"No JSON object found in LLM output (last error: {last_error}): {text[:200]}"
    )
