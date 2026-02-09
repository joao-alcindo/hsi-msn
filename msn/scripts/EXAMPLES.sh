#!/bin/bash

# ============================================================================
# USAGE EXAMPLES - Automation Scripts
# ============================================================================
# This file shows different ways to use the scripts
# ============================================================================

echo "╭────────────────────────────────────────────────────────────╮"
echo "│         USAGE EXAMPLES - HSI-MSN Automation              │"
echo "╰────────────────────────────────────────────────────────────╯"
echo ""

# ----------------------------------------------------------------------------
# EXAMPLE 1: Process a specific model (complete pipeline)
# ----------------------------------------------------------------------------
echo "📌 EXAMPLE 1: Complete pipeline for a model"
echo ""
echo "python scripts/extract_embeddings.py --model proto11_embed32_encoder_hsi"
echo "python scripts/visualize.py --model proto11_embed32_encoder_hsi"
echo "python scripts/regression.py --model proto11_embed32_encoder_hsi"
echo ""

# ----------------------------------------------------------------------------
# EXAMPLE 2: Process all models
# ----------------------------------------------------------------------------
echo "📌 EXAMPLE 2: Process all existing models"
echo ""
echo "python scripts/extract_embeddings.py --all"
echo "python scripts/visualize.py --all"
echo "python scripts/regression.py --all"
echo "python scripts/compare_models.py"
echo ""

# ----------------------------------------------------------------------------
# EXAMPLE 3: Use the automated bash script
# ----------------------------------------------------------------------------
echo "📌 EXAMPLE 3: Automated pipeline with bash"
echo ""
echo "# Edit run_experiments.sh to configure:"
echo "#   DO_TRAINING=0        (do not train new models)"
echo "#   DO_EMBEDDINGS=1      (extract embeddings)"
echo "#   DO_VISUALIZATION=1   (generate plots)"
echo "#   DO_REGRESSION=1      (run XGBoost)"
echo "#   SPECIFIC_MODEL=''    (empty = all models)"
echo ""
echo "./scripts/run_experiments.sh"
echo ""

# ----------------------------------------------------------------------------
# EXAMPLE 4: Train and process a new model
# ----------------------------------------------------------------------------
echo "📌 EXAMPLE 4: Train and process a new model"
echo ""
echo "# 1. Train"
echo "python train.py --config configs/config_vanilla.yaml"
echo ""
echo "# 2. Extract embeddings (replace NAME with actual name)"
echo "python scripts/extract_embeddings.py --model MODEL_NAME"
echo ""
echo "# 3. Process"
echo "python scripts/visualize.py --model MODEL_NAME"
echo "python scripts/regression.py --model MODEL_NAME"
echo ""

# ----------------------------------------------------------------------------
# EXAMPLE 5: Compare existing models
# ----------------------------------------------------------------------------
echo "📌 EXAMPLE 5: Compare results from multiple models"
echo ""
echo "python scripts/compare_models.py"
echo ""
echo "# View ranking"
echo "cat comparison/model_comparison.csv"
echo ""
echo "# View plots"
echo "ls comparison/*.png"
echo ""

# ----------------------------------------------------------------------------
# EXAMPLE 6: Process only embeddings (without visualizations)
# ----------------------------------------------------------------------------
echo "📌 EXAMPLE 6: Custom embedding extraction"
echo ""
echo "# Use specific checkpoint"
echo "python scripts/extract_embeddings.py \\"
echo "    --model proto11_embed32_encoder_hsi \\"
echo "    --checkpoint ckpt_epoch_100.pth \\"
echo "    --output ./embeddings_epoch100"
echo ""

# ----------------------------------------------------------------------------
# EXAMPLE 7: Production workflow (batch of experiments)
# ----------------------------------------------------------------------------
echo "📌 EXAMPLE 7: Batch of experiments"
echo ""
echo "# Train multiple models"
echo "for config in configs/*.yaml; do"
echo "    echo \"Training with \$config\""
echo "    python train.py --config \"\$config\""
echo "done"
echo ""
echo "# Process all"
echo "python scripts/extract_embeddings.py --all"
echo "python scripts/visualize.py --all"
echo "python scripts/regression.py --all"
echo "python scripts/compare_models.py"
echo ""

# ----------------------------------------------------------------------------
# EXAMPLE 8: Debug a specific model
# ----------------------------------------------------------------------------
echo "📌 EXAMPLE 8: Detailed debug"
echo ""
echo "MODEL='proto11_embed32_encoder_hsi'"
echo ""
echo "# Check if files exist"
echo "ls -lh output/\$MODEL/"
echo "ls -lh embeddings/\$MODEL/ 2>/dev/null || echo 'Embeddings not extracted'"
echo "ls -lh downstream/\$MODEL/ 2>/dev/null || echo 'Downstream not processed'"
echo ""
echo "# Process"
echo "python scripts/extract_embeddings.py --model \$MODEL"
echo "python scripts/visualize.py --model \$MODEL"
echo "python scripts/regression.py --model \$MODEL"
echo ""

# ----------------------------------------------------------------------------
# OUTPUT STRUCTURE
# ----------------------------------------------------------------------------
echo ""
echo "╭────────────────────────────────────────────────────────────╮"
echo "│                  OUTPUT STRUCTURE                        │"
echo "╰────────────────────────────────────────────────────────────╯"
echo ""
echo "embeddings/[modelo]/"
echo "  ├── embeddings.npy          # (N, embed_dim)"
echo "  ├── scores.npy              # (N, num_prototypes)"
echo "  ├── prototypes.npy          # (embed_dim, num_prototypes)"
echo "  ├── targets.csv             # ground truth"
echo "  └── metadata.json           # info do modelo"
echo ""
echo "downstream/[modelo]/"
echo "  ├── tsne_grid.png           # visualização t-SNE"
echo "  ├── umap_grid.png           # visualização UMAP"
echo "  ├── xgboost_results.csv     # métricas de regressão"
echo "  ├── xgboost_scatter_plots.png"
echo "  └── histograms/             # histogramas por cluster"
echo "      ├── histograms_B.png"
echo "      ├── histograms_Cu.png"
echo "      └── ..."
echo ""
echo "comparison/"
echo "  ├── model_comparison.csv        # model ranking"
echo "  ├── model_comparison_full.csv   # complete data"
echo "  ├── r2_heatmap.png             # R² heatmap by model"
echo "  ├── r2_boxplot.png             # R² distribution"
echo "  └── avg_r2_by_model.png        # average R²"
echo ""

# ----------------------------------------------------------------------------
# TIPS
# ----------------------------------------------------------------------------
echo "╭────────────────────────────────────────────────────────────╮"
echo "│                         TIPS                              │"
echo "╰────────────────────────────────────────────────────────────╯"
echo ""
echo "✓ Always run from the msn/ folder"
echo "✓ Make sure train_gt.csv exists at ../train_gt.csv"
echo "✓ Use --all to process all models at once"
echo "✓ Extract embeddings first (they are reused by other scripts)"
echo "✓ Use compare_models.py to see ranking and best configurations"
echo ""
echo "For more information, see scripts/README.md"
echo ""
