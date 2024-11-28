# xFold

A project of implementing AlphaFold2 from scratch.

## Directory Structure

```
.
├── configs/                        # Hydra configuration directory
│   ├── config.yaml                     # Base configuration
│   ├── predict.yaml                    # Prediction configuration
│   ├── model/                          # Model-specific configuration
│   │   └── alphafold2.yaml
│   └── experiment/                     # Experiment-specific configuration
│       └── predict.yaml
├── experiments/                    # Executable experiment scripts
│   └── predict.py                      # Inference script
└── xfold/
    ├── configs/                        # Configuration files for different models
    │   ├── base.py                         # Base configuration classes
    │   └── alphafold2.py                   # AlphaFold2 specific configurations
    ├── model/                          # Core model implementations
    │   ├── __init__.py                     # Model registration
    │   ├── base.py                         # Base abstract model class
    │   ├── common/                         # Shared model components
    │   │   ├── aux_heads.py                    # Auxiliary model heads
    │   │   ├── misc.py                         # Miscellaneous components
    │   │   ├── structure.py                    # Structure module
    │   │   └── triangle_ops.py                 # Triangle operations
    │   └── alphafold2/                     # AlphaFold2 specific implementation
    │       ├── embedders.py                    # Embedding layers
    │       ├── evoformer.py                    # Evoformer stack
    │       └── model.py                        # Main AlphaFold2 model
    ├─── protein/                       # Protein classes and functions
    │   ├── constants.py                    # Amino-acid/atomic constants
    │   ├── sequence.py                     # MSA and sequence
    │   └── structure.py                    # Rigid frames and structure
    └─── utils/                         # Utility functions
        └── path.py                         # for handling paths
```

## Checklist

- [ ] Data pipeline
- [x] Main inference loop
    - [x] MSA/Template featurisation
    - [x] Evolutionary information inclusion
        - [x] Input embedder
        - [x] Recycling embedder
        - [x] Template embedder
        - [x] Extra MSA embedder
    - [x] Evoformer stack
    - [x] Structure module
- [ ] Auxiliary loss heads 

## Resources

- AlphaFold2 [main paper](https://www.nature.com/articles/s41586-021-03819-2) and [supplementary information](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-021-03819-2/MediaObjects/41586_2021_3819_MOESM1_ESM.pdf)
