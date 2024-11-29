from abc import ABC
from typing import Any, Dict, Tuple
from xfold.protein.structure import ProteinStructure


class BaseModel(ABC):
    def predict(seq: str) -> Tuple[ProteinStructure, Dict[str, Any]]:
        raise NotImplementedError()
