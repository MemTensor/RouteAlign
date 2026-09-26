"""
请求结构化信息的自动检测（方案 §7，代码计算，不经过 LLM）。

包括输入 token 数、上下文 token 数、附件文件与多模态检测。
敏感数据检测暂不实现，contains_sensitive_data 恒为 False。
"""

from typing import Any, Dict, List, Optional

from .schemas import RequestFeatures

IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "gif", "bmp", "webp", "svg", "tiff"}
AUDIO_EXTENSIONS = {"mp3", "wav", "flac", "aac", "ogg", "m4a"}
VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm", "flv"}

_tokenizer = None


def _count_tokens(text: str) -> int:
    """使用 tiktoken 计算 token 数，不可用时按字符数近似（约 3 字符/token）。"""
    global _tokenizer
    if not text:
        return 0
    if _tokenizer is None:
        try:
            import tiktoken
            _tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _tokenizer = False
    if _tokenizer:
        return len(_tokenizer.encode(text, disallowed_special=()))
    return max(1, len(text) // 3)


def detect_request_features(
    query: str,
    context: str = "",
    attachments: Optional[List[Dict[str, Any]]] = None,
    high_risk: bool = False,
) -> RequestFeatures:
    """
    检测请求的结构化信息。

    Args:
        query: 用户当前输入文本
        context: 历史上下文文本
        attachments: 附件列表，每个附件为 {"filename": str, "mime_type": str（可选）}
        high_risk: 是否为高风险任务（高风险任务路由时不允许降档）
    """
    attachments = attachments or []
    file_types: List[str] = []
    contains_image = contains_audio = contains_video = False

    for att in attachments:
        filename = str(att.get("filename", ""))
        mime = str(att.get("mime_type", "")).lower()
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        file_types.append(ext or mime or "unknown")

        if mime.startswith("image/") or ext in IMAGE_EXTENSIONS:
            contains_image = True
        elif mime.startswith("audio/") or ext in AUDIO_EXTENSIONS:
            contains_audio = True
        elif mime.startswith("video/") or ext in VIDEO_EXTENSIONS:
            contains_video = True

    return RequestFeatures(
        input_tokens=_count_tokens(query),
        context_tokens=_count_tokens(context),
        file_count=len(attachments),
        file_types=file_types,
        contains_image=contains_image,
        contains_audio=contains_audio,
        contains_video=contains_video,
        contains_sensitive_data=False,  # 敏感数据检测暂不实现
        high_risk=high_risk,
    )
