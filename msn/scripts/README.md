# Automation Scripts - HSI-MSN

This directory contains scripts to automate the complete experiment pipeline.

## 📁 Structure

```
scripts/
├── extract_embeddings.py   # Extract and save model embeddings
├── visualize.py            # Generate visualizations (t-SNE, UMAP, histograms)
├── regression.py           # Train and evaluate XGBoost models
├── compare_models.py       # Compare results between models
└── run_experiments.sh      # Orchestrate the entire pipeline
```

## 🚀 Quick Start

### Process a specific model:

```bash
# 1. Extract embeddings
python scripts/extract_embeddings.py --model proto11_embed32_encoder_hsi

# 2. Generate visualizations
python scripts/visualize.py --model proto11_embed32_encoder_hsi

# 3. Run regressions
python scripts/regression.py --model proto11_embed32_encoder_hsi
```

### Process all models:

```bash
python scripts/extract_embeddings.py --all
python scripts/visualize.py --all
python scripts/regression.py --all
```

### Automated complete pipeline:

```bash
chmod +x scripts/run_experiments.sh
./scripts/run_experiments.sh
```

## 📋 Script Details

### 1. `extract_embeddings.py`

Extracts embeddings from the encoder and saves in NumPy format.

**Output** (`embeddings/[model]/`):
- `embeddings.npy` - Embeddings (N × D)
- `scores.npy` - Prototype scores (N × K)
- `prototypes.npy` - Prototypes (D × K)
- `targets.csv` - Ground truth
- `metadata.json` - Model information

**Arguments**:
- `--model NAME` - Model name
- `--all` - Process all models
- `--checkpoint NAME` - Checkpoint name (default: min_loss_checkpoint.pth)
- `--output DIR` - Output directory (default: ./embeddings)

### 2. `visualize.py`

Generates embedding visualizations.

**Output** (`downstream/[model]/`):
- `tsne_grid.png` - 3×2 grid with t-SNE colored by element
- `umap_grid.png` - 3×2 grid with UMAP colored by element
- `histograms/` - Histograms per cluster for each element
- `tsne_embeddings.npy` - Saved t-SNE projections
- `umap_embeddings.npy` - Saved UMAP projections

**Arguments**:
- `--model NAME` - Model name
- `--all` - Process all models
- `--embeddings DIR` - Embeddings directory (default: ./embeddings)
- `--output DIR` - Output directory (default: ./downstream)

### 3. `regression.py`

Trains XGBoost models to predict each element.

**Output** (`downstream/[model]/`):
- `xgboost_results.csv` - Metrics (R², MAE, RMSE)
- `xgboost_scatter_plots.png` - Scatter plots True vs Predicted

**Arguments**:
- `--model NAME` - Model name
- `--all` - Process all models
- `--embeddings DIR` - Embeddings directory (default: ./embeddings)
- `--output DIR` - Output directory (default: ./downstream)

### 4. `compare_models.py`

Compares results from multiple models.

**Output** (`comparison/`):
- `model_comparison.csv` - Model ranking
- `model_comparison_full.csv` - Complete data
- `r2_heatmap.png` - R² heatmap by model and element
- `r2_boxplot.png` - R² distribution by element
- `avg_r2_by_model.png` - Average R² by model

**Arguments**:
- `--downstream DIR` - Directory with results (default: ./downstream)
- `--output FILE` - Output file (default: ./comparison/model_comparison.csv)

### 5. `run_experiments.sh`

Orchestrates the entire pipeline automatically.

**Configuration** (edit the file):
```bash
DO_TRAINING=0        # Train new models
DO_EMBEDDINGS=1      # Extract embeddings
DO_VISUALIZATION=1   # Generate visualizations
DO_REGRESSION=1      # Run regressions
SPECIFIC_MODEL=""    # Specific model (empty = all)
```

## 📊 Complete Workflow Example

```bash
# 1. Train a new model
python train.py --config configs/config_vanilla.yaml

# 2. Process the model
python scripts/extract_embeddings.py --model proto11_embed32_encoder_vanilla
python scripts/visualize.py --model proto11_embed32_encoder_vanilla
python scripts/regression.py --model proto11_embed32_encoder_vanilla

# 3. Compare with other models
python scripts/compare_models.py

# 4. View results
cat downstream/proto11_embed32_encoder_vanilla/xgboost_results.csv
```

## 🔧 Requirements

Make sure you have all dependencies installed:

```bash
pip install torch numpy pandas matplotlib scikit-learn xgboost umap-learn tqdm pyyaml
```

## 📝 Notes

- Scripts assume you are running from the `msn/` folder
- The `train_gt.csv` file must be at `../train_gt.csv`
- Models must be in `output/[model_name]/`
- Checkpoints must be named `min_loss_checkpoint.pth` (or specify another)

## 🆘 Troubleshooting

**Error: "Embeddings not found"**
→ Run `extract_embeddings.py` first

**Error: "Config not found"**
→ Check if the model exists in `output/`

**Error: "Targets not found"**
→ Check if `train_gt.csv` exists at `../`

## 📧 Support

For issues or suggestions, consult the main project documentation.
