from dataclasses import dataclass
import inspect
from typing import Any, Dict


@dataclass
class BaseModelConfig:
    device: str = "cpu"

    @classmethod
    def from_dict(cls, kwargs: Dict[str, Any]) -> "BaseModelConfig":
        expected_params = inspect.signature(cls).parameters
        return cls(**{k: kwargs[k] for k in expected_params if k in kwargs})
