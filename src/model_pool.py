"""
模型池配置的加载。

待路由的模型服务属性标签由用户在单独的模型池配置文件（JSON/YAML）中自行配置，
包含价格、延迟、上下文窗口、工具支持、多模态支持、敏感数据许可、可用状态，
以及可选的调用端点信息（用于线上真正调用最终选定的模型）。

数据隐私/敏感数据检测暂不实现，sensitive_data_allowed 统一按 true 处理。
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from loguru import logger

from .schemas import ServiceAttributes


class ModelPoolEntry:
    """模型池中的一个模型：服务属性 + 可选的调用端点配置。"""

    def __init__(
        self,
        model_id: str,
        service_attributes: ServiceAttributes,
        endpoint: Optional[Dict[str, Any]] = None,
    ):
        self.model_id = model_id
        self.service_attributes = service_attributes
        self.endpoint = endpoint  # {"model", "base_url", "api_key", ...}，可为 None


class ModelPool:
    """模型池：所有待路由模型的服务属性集合。"""

    def __init__(self, entries: List[ModelPoolEntry]):
        self.entries = {e.model_id: e for e in entries}

    @classmethod
    def load(cls, path: str) -> "ModelPool":
        """从 JSON 或 YAML 模型池配置文件加载。"""
        pool_path = Path(path)
        if not pool_path.exists():
            raise FileNotFoundError(f"Model pool config not found: {path}")

        with open(pool_path, "r", encoding="utf-8") as f:
            if pool_path.suffix in (".yaml", ".yml"):
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        models_cfg = data.get("models") if isinstance(data, dict) else data
        if not isinstance(models_cfg, list):
            raise ValueError(
                f"Model pool config must contain a 'models' list: {path}"
            )

        entries = []
        for cfg in models_cfg:
            model_id = cfg.get("model_id") or cfg.get("name")
            if not model_id:
                raise ValueError(f"Model pool entry missing model_id: {cfg}")
            attrs = ServiceAttributes.from_dict(cfg.get("service_attributes") or {})
            # 敏感数据检测暂不实现，统一允许
            attrs.sensitive_data_allowed = True
            entries.append(ModelPoolEntry(
                model_id=model_id,
                service_attributes=attrs,
                endpoint=cfg.get("endpoint"),
            ))

        logger.info(f"Loaded model pool from {path}: {len(entries)} models")
        return cls(entries)

    def model_ids(self) -> List[str]:
        return list(self.entries.keys())

    def get(self, model_id: str) -> ModelPoolEntry:
        return self.entries[model_id]

    def service_attributes(self, model_id: str) -> ServiceAttributes:
        return self.entries[model_id].service_attributes

    def __contains__(self, model_id: str) -> bool:
        return model_id in self.entries

    def __len__(self) -> int:
        return len(self.entries)
