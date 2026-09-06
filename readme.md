# FinShield

Research code for **FinShield**, an offline LLM-assisted feature-engineering
pipeline for graph-based financial fraud detection. The LLM is used during
development to assist with feature-engineering code; the released downstream
pipeline extracts graph features and trains a CatBoost classifier.

## Repository layout

```text
.
├── code_cross/             # Main feature, training, and evaluation pipeline
├── code_cross_tfinance/    # T-Finance-specific feature pipeline
├── code_cross_all/         # Dataset-specific cross-feature experiments
├── baselines/              # Baseline implementations and their local scripts
└── reproducibility/        # Prompt material and frozen DGraph feature order
```

### Main implementation

- `code_cross/` supports the DGraph, Amazon, YelpChi, and T-Finance data
  layouts used by the shared feature and CatBoost pipeline.
- `code_cross_tfinance/` contains the dedicated multi-stage T-Finance feature
  pipeline.
- `code_cross_all/` contains standalone cross-feature implementations for the
  supported datasets.

### Baselines

`baselines/` contains separated implementations for AMNet, DGA-GNN, DSGAD,
GHRN, AntiFraud, and WWW25-Grad. These methods have heterogeneous dependencies
and entry points; consult the README or runner script within each baseline
directory before executing it.

## Requirements

The repository does not currently provide a pinned root environment file. The
main pipelines import the following packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas scipy scikit-learn catboost torch dgl \
  tqdm joblib click networkx psutil lightgbm matplotlib
```

Install PyTorch and DGL builds compatible with your operating system and
accelerator configuration. The DGraph feature and CatBoost training path runs
with `task_type='CPU'` in the released training code.

## Data

Datasets and generated artifacts are intentionally excluded from version
control. Obtain each dataset from its original distribution and comply with its
terms of use. For DGraph background and access information, see the
[DGraph paper](https://arxiv.org/abs/2207.03579).

The DGraph preparation script expects the following local input:

```text
data/
└── dgraphfin.npz
```

The input archive is expected to contain `x`, `y`, `edge_index`, `edge_type`,
`edge_timestamp`, `train_mask`, `valid_mask`, and `test_mask`. Generated split
files are written under `data_split/`; generated features are written under
`feature_split/`; trained models and predictions are written under
`models_ours/`. These directories are ignored by Git.

## DGraph workflow

Run commands from `code_cross/` so that the relative paths used by the scripts
resolve correctly.

```bash
cd code_cross

# Create data_split/dgraphfin_{train,valid,test}.npz from data/dgraphfin.npz.
mkdir -p ../data_split
python data_split_dgraphfin.py

# Build training features and train the CatBoost ensemble.
bash run_fe_train.sh dgraphfin train

# Build test features and evaluate the trained ensemble.
bash run_fe_test.sh dgraphfin test
```

The runner scripts create intermediate `.pkl` feature files and model outputs
in the ignored directories described above. They also contain optional runtime
and memory measurements. Review the scripts before changing paths, dataset
names, or resource-related settings.

## T-Finance workflow

The T-Finance pipeline is kept separate because it uses a different staged
feature workflow. Its primary entry points are:

```text
code_cross_tfinance/run_fe_train.sh
code_cross_tfinance/run_fe_test.sh
code_cross_tfinance/run_train.sh
```

Each runner accepts or defines paths relative to `code_cross_tfinance/`. Check
the script header before execution and provide the expected
`data_split/tfinance_{train,test}.npz` inputs.

## Reproducibility materials

[`reproducibility/README.md`](reproducibility/README.md) documents the
available prompt material, the fixed DGraph cross-feature definitions, and the
released downstream training configuration.
[`reproducibility/dgraphfin_final_features.json`](reproducibility/dgraphfin_final_features.json)
freezes the ordered list of 130 DGraph base features used by the released
training code.
