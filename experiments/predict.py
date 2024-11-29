from dataclasses import asdict
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import random
import torch
from xfold.utils.path import out_dir
from xfold.model import FOLDING_MODEL_REGISTRY


@hydra.main(version_base=None, config_path="../configs", config_name="predict")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)

    # Ensure all experiment configs are set
    exp_cfg = cfg.experiment
    missing_exp_cfg_keys = OmegaConf.missing_keys(exp_cfg)
    if missing_exp_cfg_keys:
        raise Exception(f"Missing mandatory values: {', '.join(missing_exp_cfg_keys)}")

    if exp_cfg.seed is not None:
        set_seed(exp_cfg.seed)

    # Initialise model
    model_cfg = cfg.model
    model_cls, model_cfg_resolver = FOLDING_MODEL_REGISTRY.get(
        model_cfg.name, (None, None)
    )
    if model_cls is None:
        available_models = ", ".join(FOLDING_MODEL_REGISTRY.keys())
        raise Exception(
            f"Model '{model_cfg.name}' is not registered. Choose from: {available_models}"
        )
    ## If no weights given, test on randomly initialised model
    has_weights = not OmegaConf.is_missing(model_cfg, "weights")

    model_kwargs = asdict(model_cfg_resolver.from_dict(model_cfg))
    model = model_cls(**model_kwargs)
    if has_weights:
        model.load_state_dict(torch.load(model_cfg.weights, weights_only=True))

    # Predict structure
    model.eval()
    with torch.no_grad():
        struct, aux_preds = model.predict(cfg.experiment.seq)

    out = out_dir()
    f_out = os.path.join(out, f"pred_struct__{exp_cfg.seq}.pdb")
    struct.save_as_pdb(f_out)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    main()
