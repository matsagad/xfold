from abc import ABC
from typing import Tuple
from xfold.protein.structure import ProteinStructure


class BaseModel(ABC):
    def predict(seq: str) -> Tuple[ProteinStructure, float]:
        raise NotImplementedError()
