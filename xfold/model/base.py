from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
from xfold.protein.structure import ProteinStructure


class BaseFoldingModel(ABC):
    """Abstract base class for the folding models."""
    @abstractmethod
    def fold(self, seq: str) -> Tuple[ProteinStructure, Dict[str, Any]]:
        """Predict structure and auxiliary measures (e.g. uncertainty) from sequence."""
        raise NotImplementedError()
