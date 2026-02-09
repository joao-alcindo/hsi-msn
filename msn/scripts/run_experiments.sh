#!/bin/bash

# ============================================================================
# Experiment Automation Script - HSI-MSN
# ============================================================================
# This script automates the complete pipeline:
# 1. Train models
# 2. Extract embeddings
# 3. Generate visualizations
# 4. Run regressions
# ============================================================================

set -e  # Stop on first error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# CONFIGURATION
# ============================================================================

# List of configurations to train
CONFIGS=(
    "configs/config_vanilla.yaml"
    "configs/config.yaml"
)

# Options (0 = do not run, 1 = run)
DO_TRAINING=0        # Train new models
DO_EMBEDDINGS=1      # Extract embeddings
DO_VISUALIZATION=1   # Generate visualizations
DO_REGRESSION=1      # Run regressions

# Specific model (leave empty to process all)
SPECIFIC_MODEL="proto20_embed64_encoder_hsi_mr_50_depth12_nH16"    # Ex: "proto11_embed32_encoder_hsi"

# ============================================================================
# FUNCTIONS
# ============================================================================

print_header() {
    echo -e "${BLUE}"
    echo "============================================================================"
    echo "$1"
    echo "============================================================================"
    echo -e "${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# ============================================================================
# STAGE 1: TRAINING
# ============================================================================

if [ $DO_TRAINING -eq 1 ]; then
    print_header "STAGE 1: MODEL TRAINING"
    
    for config in "${CONFIGS[@]}"; do
        if [ -f "$config" ]; then
            echo -e "\n${YELLOW}Training with: $config${NC}"
            python train.py --config "$config"
            print_success "Training completed: $config"
        else
            print_warning "Config not found: $config"
        fi
    done
    
    print_success "All training completed!"
else
    print_warning "Training disabled (DO_TRAINING=0)"
fi

# ============================================================================
# STAGE 2: EMBEDDING EXTRACTION
# ============================================================================

if [ $DO_EMBEDDINGS -eq 1 ]; then
    print_header "STAGE 2: EMBEDDING EXTRACTION"
    
    if [ -n "$SPECIFIC_MODEL" ]; then
        echo "Extracting embeddings from model: $SPECIFIC_MODEL"
        python scripts/extract_embeddings.py --model "$SPECIFIC_MODEL"
    else
        echo "Extracting embeddings from all models..."
        python scripts/extract_embeddings.py --all
    fi
    
    print_success "Embeddings extracted!"
else
    print_warning "Embedding extraction disabled (DO_EMBEDDINGS=0)"
fi

# ============================================================================
# STAGE 3: VISUALIZATIONS
# ============================================================================

if [ $DO_VISUALIZATION -eq 1 ]; then
    print_header "STAGE 3: VISUALIZATION GENERATION"
    
    if [ -n "$SPECIFIC_MODEL" ]; then
        echo "Generating visualizations for: $SPECIFIC_MODEL"
        python scripts/visualize.py --model "$SPECIFIC_MODEL"
    else
        echo "Generating visualizations for all models..."
        python scripts/visualize.py --all
    fi
    
    print_success "Visualizations generated!"
else
    print_warning "Visualizations disabled (DO_VISUALIZATION=0)"
fi

# ============================================================================
# STAGE 4: REGRESSIONS
# ============================================================================

if [ $DO_REGRESSION -eq 1 ]; then
    print_header "STAGE 4: REGRESSION EXPERIMENTS"
    
    if [ -n "$SPECIFIC_MODEL" ]; then
        echo "Running regressions for: $SPECIFIC_MODEL"
        python scripts/regression.py --model "$SPECIFIC_MODEL"
    else
        echo "Running regressions for all models..."
        python scripts/regression.py --all
    fi
    
    print_success "Regressions completed!"
else
    print_warning "Regressions disabled (DO_REGRESSION=0)"
fi

# ============================================================================
# COMPLETION
# ============================================================================

print_header "PIPELINE COMPLETE!"

echo ""
echo "Summary of executed stages:"
[ $DO_TRAINING -eq 1 ] && echo -e "${GREEN}✓${NC} Training" || echo -e "${YELLOW}⊘${NC} Training"
[ $DO_EMBEDDINGS -eq 1 ] && echo -e "${GREEN}✓${NC} Embeddings" || echo -e "${YELLOW}⊘${NC} Embeddings"
[ $DO_VISUALIZATION -eq 1 ] && echo -e "${GREEN}✓${NC} Visualizations" || echo -e "${YELLOW}⊘${NC} Visualizations"
[ $DO_REGRESSION -eq 1 ] && echo -e "${GREEN}✓${NC} Regressions" || echo -e "${YELLOW}⊘${NC} Regressions"

echo ""
print_success "All experiments completed successfully!"
echo ""
echo "Results available at:"
echo "  - Embeddings: ./embeddings/"
echo "  - Visualizations: ./downstream/"
echo "  - Regressions: ./downstream/*/xgboost_results.csv"
echo ""
