"""
Script to train regression models on embeddings.

Usage:
    python scripts/regression.py --model proto11_embed32_encoder_hsi
    python scripts/regression.py --all
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def load_embeddings_and_targets(model_name, embeddings_base='./embeddings'):
    """Load embeddings and targets."""
    model_dir = os.path.join(embeddings_base, model_name)
    
    if not os.path.exists(model_dir):
        print(f"❌ Embeddings não encontrados: {model_name}")
        return None
    
    embeddings = np.load(os.path.join(model_dir, 'embeddings.npy'))
    targets_path = os.path.join(model_dir, 'targets.csv')
    
    if not os.path.exists(targets_path):
        print(f"❌ Targets não encontrados: {targets_path}")
        return None
    
    targets = pd.read_csv(targets_path)
    
    return embeddings, targets


def train_and_evaluate_xgboost(embeddings, targets, elementos=['B', 'Cu', 'Zn', 'Fe', 'S', 'Mn'],
                               test_size=0.2, random_seed=42):
    """
    Train XGBoost for each element and return results.
    
    Returns:
        results_df: DataFrame with metrics
        predictions_dict: Dictionary with predictions for each element
    """
    # Split train/val based on sample_index
    if 'sample_index' in targets.columns:
        sample_idx = targets['sample_index'].nunique()
        np.random.seed(random_seed)
        train_indices = np.random.choice(sample_idx, 
                                        size=int((1-test_size) * sample_idx), 
                                        replace=False)
        val_indices = np.setdiff1d(np.arange(sample_idx), train_indices)
        
        X_train_idx = targets[targets['sample_index'].isin(train_indices)].index
        X_val_idx = targets[targets['sample_index'].isin(val_indices)].index
    else:
        # Simple split if no sample_index
        n_samples = len(embeddings)
        indices = np.arange(n_samples)
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        
        split_point = int((1-test_size) * n_samples)
        X_train_idx = indices[:split_point]
        X_val_idx = indices[split_point:]
    
    E_train = embeddings[X_train_idx]
    E_val = embeddings[X_val_idx]
    
    results = []
    predictions_dict = {}
    
    for elemento in elementos:
        if elemento not in targets.columns:
            print(f"⚠️  Elemento {elemento} não encontrado!")
            continue
        
        print(f"  Treinando {elemento}...", end=' ')
        
        y_train = targets.loc[X_train_idx, elemento].values
        y_val = targets.loc[X_val_idx, elemento].values
        
        # XGBoost model
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            early_stopping_rounds=50,
            n_jobs=-1,
            random_state=random_seed
        )
        
        model.fit(
            E_train, y_train,
            eval_set=[(E_val, y_val)],
            verbose=False
        )
        
        preds = model.predict(E_val)
        
        # Metrics
        r2 = r2_score(y_val, preds)
        mae = mean_absolute_error(y_val, preds)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        
        results.append([elemento, r2, mae, rmse])
        predictions_dict[elemento] = {
            'y_true': y_val,
            'y_pred': preds
        }
        
        print(f"R²={r2:.4f}")
    
    results_df = pd.DataFrame(results, columns=['Elemento', 'R2', 'MAE', 'RMSE'])
    return results_df, predictions_dict


def plot_regression_results(predictions_dict, output_path, elementos=['B', 'Cu', 'Zn', 'Fe', 'S', 'Mn']):
    """Create grid of scatter plots for regression."""
    available_elements = [e for e in elementos if e in predictions_dict]
    
    if not available_elements:
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()
    
    for i, elemento in enumerate(available_elements):
        if i >= len(axes):
            break
        
        ax = axes[i]
        data = predictions_dict[elemento]
        
        y_true = data['y_true']
        y_pred = data['y_pred']
        
        ax.scatter(y_true, y_pred, alpha=0.5, s=20)
        
        lim_min = min(y_true.min(), y_pred.min())
        lim_max = max(y_true.max(), y_pred.max())
        ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', lw=2)
        
        ax.set_title(elemento, fontsize=14, weight='bold')
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predictions')
        ax.grid(True, alpha=0.3)
    
    # Disable extra axes
    for j in range(len(available_elements), len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def run_regression_experiments(model_name, embeddings_base='./embeddings',
                               output_base='./downstream'):
    """Run regression experiments for a model."""
    print(f"\n{'='*60}")
    print(f"Regressão: {model_name}")
    print(f"{'='*60}")
    
    # Load data
    data = load_embeddings_and_targets(model_name, embeddings_base)
    if data is None:
        return False
    
    embeddings, targets = data
    
    # Create output folder
    output_dir = os.path.join(output_base, model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Train models
    print("\nTraining XGBoost models:")
    results_df, predictions = train_and_evaluate_xgboost(embeddings, targets)
    
    # Save results
    results_path = os.path.join(output_dir, 'xgboost_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Results saved: {results_path}")
    
    # Show results
    print("\n" + "="*60)
    print(results_df.to_string(index=False))
    print("="*60)
    
    # Plot results
    plot_path = os.path.join(output_dir, 'xgboost_scatter_plots.png')
    plot_regression_results(predictions, plot_path)
    print(f"✓ Plots saved: {plot_path}")
    
    print(f"\n✅ Regressão concluída: {model_name}\n")
    return True


def main():
    parser = argparse.ArgumentParser(description='Regression experiments')
    parser.add_argument('--model', type=str, help='Model name')
    parser.add_argument('--all', action='store_true', help='Process all models')
    parser.add_argument('--embeddings', type=str, default='./embeddings',
                       help='Embeddings directory')
    parser.add_argument('--output', type=str, default='./downstream',
                       help='Output directory')
    
    args = parser.parse_args()
    
    if not args.model and not args.all:
        parser.error("Specify --model NAME or --all")
    
    if args.all:
        if not os.path.exists(args.embeddings):
            print(f"❌ Pasta {args.embeddings} não encontrada!")
            return
        
        models = [d for d in os.listdir(args.embeddings)
                 if os.path.isdir(os.path.join(args.embeddings, d))]
        
        print(f"Encontrados {len(models)} modelos")
        
        success = 0
        for model in models:
            if run_regression_experiments(model, args.embeddings, args.output):
                success += 1
        
        print(f"\n{'='*60}")
        print(f"Regression complete: {success}/{len(models)}")
        print(f"{'='*60}")
    else:
        run_regression_experiments(args.model, args.embeddings, args.output)


if __name__ == '__main__':
    main()
