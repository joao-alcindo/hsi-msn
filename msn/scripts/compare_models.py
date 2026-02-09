"""
Script to compare regression results between models.

Usage:
    python scripts/compare_models.py
    python scripts/compare_models.py --output comparison_report.csv
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


def collect_results(downstream_dir='./downstream'):
    """Collect results from all models."""
    all_results = []
    
    if not os.path.exists(downstream_dir):
        print(f"❌ Diretório não encontrado: {downstream_dir}")
        return None
    
    models = [d for d in os.listdir(downstream_dir)
             if os.path.isdir(os.path.join(downstream_dir, d))]
    
    for model in models:
        results_file = os.path.join(downstream_dir, model, 'xgboost_results.csv')
        
        if os.path.exists(results_file):
            df = pd.read_csv(results_file)
            df['Model'] = model
            all_results.append(df)
    
    if not all_results:
        print("❌ Nenhum resultado encontrado!")
        return None
    
    combined = pd.concat(all_results, ignore_index=True)
    return combined


def create_comparison_plots(df, output_dir='./comparison'):
    """Create comparative plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Heatmap of R² by model and element
    pivot_r2 = df.pivot(index='Model', columns='Elemento', values='R2')
    
    plt.figure(figsize=(10, len(pivot_r2) * 0.5 + 2))
    sns.heatmap(pivot_r2, annot=True, fmt='.3f', cmap='RdYlGn', 
                vmin=0, vmax=1, cbar_kws={'label': 'R²'})
    plt.title('R² Score por Modelo e Elemento', fontsize=14, weight='bold')
    plt.xlabel('Elemento')
    plt.ylabel('Modelo')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'r2_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Heatmap R² salvo")
    
    # 2. Boxplot of R² by element
    plt.figure(figsize=(12, 6))
    df_sorted = df.sort_values('Elemento')
    sns.boxplot(data=df_sorted, x='Elemento', y='R2', palette='Set2')
    plt.title('Distribuição de R² por Elemento (todos os modelos)', fontsize=14, weight='bold')
    plt.ylabel('R² Score')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'r2_boxplot.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Boxplot R² salvo")
    
    # 3. Bar graph by model (average R²)
    avg_r2 = df.groupby('Model')['R2'].mean().sort_values(ascending=False)
    
    plt.figure(figsize=(10, max(6, len(avg_r2) * 0.3)))
    avg_r2.plot(kind='barh', color='steelblue')
    plt.title('R² Médio por Modelo', fontsize=14, weight='bold')
    plt.xlabel('R² Médio')
    plt.ylabel('Modelo')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'avg_r2_by_model.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Gráfico de barras salvo")
    
    # 4. Ranking table
    ranking = df.groupby('Model').agg({
        'R2': ['mean', 'std', 'min', 'max'],
        'MAE': 'mean',
        'RMSE': 'mean'
    }).round(4)
    
    ranking.columns = ['R2_mean', 'R2_std', 'R2_min', 'R2_max', 'MAE_mean', 'RMSE_mean']
    ranking = ranking.sort_values('R2_mean', ascending=False)
    
    return ranking


def main():
    parser = argparse.ArgumentParser(description='Compare results from multiple models')
    parser.add_argument('--downstream', type=str, default='./downstream',
                       help='Directory with results')
    parser.add_argument('--output', type=str, default='./comparison/model_comparison.csv',
                       help='Output file')
    parser.add_argument('--plots-dir', type=str, default='./comparison',
                       help='Directory for plots')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60 + "\n")
    
    # Collect results
    print("Collecting results...")
    df = collect_results(args.downstream)
    
    if df is None:
        return
    
    print(f"✓ Encontrados {df['Model'].nunique()} modelos")
    print(f"✓ Total de {len(df)} resultados\n")
    
    # Create plots
    print("Generating comparative visualizations...")
    ranking = create_comparison_plots(df, args.plots_dir)
    
    # Save full comparison
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output.replace('.csv', '_full.csv'), index=False)
    ranking.to_csv(args.output)
    
    print(f"\n✓ Resultados completos: {args.output.replace('.csv', '_full.csv')}")
    print(f"✓ Ranking: {args.output}")
    
    # Show ranking
    print("\n" + "="*60)
    print("MODEL RANKING (by average R²)")
    print("="*60)
    print(ranking.to_string())
    print("="*60)
    
    # Best model by element
    print("\n" + "="*60)
    print("BEST MODEL BY ELEMENT")
    print("="*60)
    best_by_element = df.loc[df.groupby('Elemento')['R2'].idxmax()]
    print(best_by_element[['Elemento', 'Model', 'R2', 'MAE']].to_string(index=False))
    print("="*60)
    
    print(f"\n✅ Comparação concluída! Resultados em: {args.plots_dir}")


if __name__ == '__main__':
    main()
