import pathlib
from typing import Callable, Dict, Tuple
from xfold.utils.path import import_all_files_and_submodules_in_directory
from xfold.configs.base import BaseModelConfig
from xfold.model.base import BaseModel

FOLDING_MODEL_REGISTRY: Dict[str, Tuple[BaseModel, BaseModelConfig]] = {}


def register_folding_model(
    name: str, config: BaseModelConfig
) -> Callable[[BaseModel], BaseModel]:

    def register(model: BaseModel) -> BaseModel:
        if name in FOLDING_MODEL_REGISTRY:
            raise Exception(f"Folding model '{name}' already registered!")

        FOLDING_MODEL_REGISTRY[name] = (model, config)

        return model

    return register


# Load all folding models to populate registry
import_all_files_and_submodules_in_directory(
    pathlib.Path(__file__).parent.resolve(), "xfold.model"
)
