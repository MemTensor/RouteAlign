"""
适配模块独立的 LLM 调用客户端。

用于适配模块自身的 LLM 调用（题库元数据与能力权重标注等），
内置指数退避重试机制（最多 10 次）。

注意：本模块与 src.llm_client 相互独立，不共用实现。
"""

import json
import os
import random
import re
import time
from typing import Any, Dict, List, Optional

from loguru import logger
from openai import OpenAI

ADAPTER_MAX_RETRIES = 10
ADAPTER_BASE_DELAY = 1.0
ADAPTER_MAX_DELAY = 60.0


def _resolve_env(value: str) -> str:
    """值为全大写环境变量名形式（如 GRADER_API_KEY）时从环境变量读取。"""
    if isinstance(value, str) and re.fullmatch(r"[A-Z][A-Z0-9_]+", value):
        resolved = os.environ.get(value)
        if resolved:
            return resolved
    return value


class AdapterLLMClient:
    """适配模块专用的、带指数退避重试的 OpenAI 兼容客户端。"""

    def __init__(self, model_config: Dict[str, Any]):
        """
        Args:
            model_config: {"model", "base_url", "api_key", "temperature",
                           "max_tokens", "timeout"}
                          api_key / base_url 支持环境变量名引用。
        """
        self.model = model_config["model"]
        self.temperature = float(model_config.get("temperature", 0.2))
        self.max_tokens = model_config.get("max_tokens")
        self._client = OpenAI(
            base_url=_resolve_env(model_config["base_url"]),
            api_key=_resolve_env(model_config.get("api_key", "none")),
            timeout=float(model_config.get("timeout", 600.0)),
            max_retries=0,
        )

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """chat completion 调用，指数退避重试最多 ADAPTER_MAX_RETRIES 次。"""
        last_error: Optional[Exception] = None
        for attempt in range(ADAPTER_MAX_RETRIES):
            try:
                kwargs: Dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                }
                if self.max_tokens is not None:
                    kwargs["max_tokens"] = self.max_tokens
                response = self._client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                if content is None or not content.strip():
                    raise ValueError("Adapter LLM returned empty content")
                return content
            except Exception as e:  # noqa: BLE001
                last_error = e
                delay = min(
                    ADAPTER_BASE_DELAY * (2 ** attempt), ADAPTER_MAX_DELAY
                ) * (1 + random.random() * 0.1)
                logger.warning(
                    f"Adapter LLM call failed (model={self.model}, "
                    f"attempt {attempt + 1}/{ADAPTER_MAX_RETRIES}): {e}. "
                    f"Retrying in {delay:.1f}s"
                )
                if attempt < ADAPTER_MAX_RETRIES - 1:
                    time.sleep(delay)
        raise RuntimeError(
            f"Adapter LLM call failed after {ADAPTER_MAX_RETRIES} attempts "
            f"(model={self.model}): {last_error}"
        )

    def chat_json(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """调用并解析 JSON 输出，解析失败同样计入重试。"""
        last_error: Optional[Exception] = None
        for attempt in range(ADAPTER_MAX_RETRIES):
            try:
                content = self.chat(messages)
                return _extract_json(content)
            except RuntimeError:
                raise
            except Exception as e:
                last_error = e
                delay = min(
                    ADAPTER_BASE_DELAY * (2 ** attempt), ADAPTER_MAX_DELAY
                ) * (1 + random.random() * 0.1)
                logger.warning(
                    f"Adapter LLM JSON parse failed (model={self.model}, "
                    f"attempt {attempt + 1}/{ADAPTER_MAX_RETRIES}): {e}. "
                    f"Retrying in {delay:.1f}s"
                )
                if attempt < ADAPTER_MAX_RETRIES - 1:
                    time.sleep(delay)
        raise RuntimeError(
            f"Adapter LLM JSON call failed after {ADAPTER_MAX_RETRIES} attempts "
            f"(model={self.model}): {last_error}"
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
            while i < n and text[i] != "\n":
                i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _repair_json(candidate: str) -> str:
    """修复常见的非标准 JSON 输出：行内注释、尾逗号。"""
    repaired = _strip_json_comments(candidate)
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    return repaired


def _extract_json(text: str) -> Dict[str, Any]:
    """
    从适配模块 LLM 输出中提取 JSON 对象。

    容忍 ```json 代码块、思维链前后缀、行内注释、尾逗号等常见格式问题；
    先尝试原文解析，失败后再做修复解析。
    """
    text = re.sub(r"<think>.*?</think>", "", text.strip(), flags=re.DOTALL).strip()

    candidates = []
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence:
        candidates.append(fence.group(1))
    candidates.append(text)
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
        f"No JSON object found in adapter LLM output "
        f"(last error: {last_error}): {text[:200]}"
    )
