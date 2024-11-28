import hydra
from omegaconf import DictConfig, OmegaConf
from dataclasses import asdict
from xfold.model import FOLDING_MODEL_REGISTRY


@hydra.main(version_base=None, config_path="../configs", config_name="predict")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)

    model_cfg = cfg.model
    model_cls, model_cfg_resolver = FOLDING_MODEL_REGISTRY.get(
        model_cfg.name, (None, None)
    )
    if model_cls is None:
        available_models = ", ".join(FOLDING_MODEL_REGISTRY.keys())
        raise Exception(
            f"Model '{model_cfg.name}' is not registered. Choose from: {available_models}"
        )
    model_kwargs = asdict(model_cfg_resolver.from_dict(model_cfg))
    model = model_cls(**model_kwargs)

    struct, plddt = model.predict(cfg.experiment.seq)


if __name__ == "__main__":
    main()
