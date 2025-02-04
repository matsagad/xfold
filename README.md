# xFold

A collection of folding models implemented from scratch.

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
    │   ├── alphafold2.py                   # AlphaFold2 specific configurations
    │   └── esmfold.py                      # ESMFold specific configurations
    ├── model/                          # Core model implementations
    │   ├── __init__.py                     # Model registration
    │   ├── base.py                         # Base abstract model class
    │   ├── common/                         # Shared model components
    │   │   ├── aux_heads.py                    # Auxiliary model heads
    │   │   ├── misc.py                         # Miscellaneous components
    │   │   ├── structure.py                    # Structure module
    │   │   └── triangle_ops.py                 # Triangle operations
    │   ├── alphafold2/                     # AlphaFold2 specific implementation
    │   │   ├── embedders.py                    # Embedding layers
    │   │   ├── evoformer.py                    # Evoformer stack
    │   │   └── model.py                        # Main AlphaFold2 model
    │   └── esmfold/                        # ESMFold specific implementation
    │       ├── constants.py                    # ESM-specific constants
    │       ├── lm.py                           # BERT / ESM language model
    │       └── model.py                        # Main ESMFold model
    ├─── protein/                       # Protein classes and functions
    │   ├── constants.py                    # Amino-acid/atomic constants
    │   ├── sequence.py                     # MSA and sequence
    │   └── structure.py                    # Rigid frames and structure
    └─── utils/                         # Utility functions
        └── path.py                         # for handling paths
```

## Checklist

### AlphaFold2
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
- [x] Auxiliary loss heads 

### ESMFold
- [x] Main inference loop
    - [x] ESM language model
    - [x] Folding trunk

## Resources

- AlphaFold2 [main paper](https://www.nature.com/articles/s41586-021-03819-2) and [supplementary information](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-021-03819-2/MediaObjects/41586_2021_3819_MOESM1_ESM.pdf)
- ESMFold [biorxiv paper](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v1.full.pdf) and [supplementary information](https://www.science.org/doi/suppl/10.1126/science.ade2574/suppl_file/science.ade2574_sm.pdf)