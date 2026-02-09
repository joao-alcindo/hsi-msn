"""
Script to generate visualizations of embeddings.

Usage:
    python scripts/visualize.py --model proto11_embed32_encoder_hsi
    python scripts/visualize.py --all
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from sklearn.manifold import TSNE
import umap


def load_embeddings(model_name, embeddings_base='./embeddings'):
    """Load embeddings from a model."""
    model_dir = os.path.join(embeddings_base, model_name)
    
    if not os.path.exists(model_dir):
        print(f"❌ Embeddings não encontrados para: {model_name}")
        print(f"   Execute primeiro: python scripts/extract_embeddings.py --model {model_name}")
        return None
    
    data = {
        'embeddings': np.load(os.path.join(model_dir, 'embeddings.npy')),
        'scores': np.load(os.path.join(model_dir, 'scores.npy')),
        'prototypes': np.load(os.path.join(model_dir, 'prototypes.npy')),
    }
    
    targets_path = os.path.join(model_dir, 'targets.csv')
    if os.path.exists(targets_path):
        data['targets'] = pd.read_csv(targets_path)
    
    return data


def compute_tsne_umap(embeddings, prototypes):
    """Compute t-SNE and UMAP."""
    print("Computing t-SNE...")
    data_all = np.vstack([embeddings, prototypes.T])
    
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(data_all)
    tsne_E = tsne_results[:len(embeddings)]
    tsne_prot = tsne_results[len(embeddings):]
    
    print("Computing UMAP...")
    reducer = umap.UMAP(n_components=2, random_state=42)
    umap_results = reducer.fit_transform(data_all)
    umap_E = umap_results[:len(embeddings)]
    umap_prot = umap_results[len(embeddings):]
    
    return {
        'tsne_E': tsne_E, 'tsne_prot': tsne_prot,
        'umap_E': umap_E, 'umap_prot': umap_prot
    }


def plot_element_grid(projection_E, projection_prot, targets, output_path, 
                     projection_name='t-SNE', elements=['B', 'Cu', 'Zn', 'Fe', 'S', 'Mn']):
    """Create 3x2 grid of plots colored by element."""
    fig, axes = plt.subplots(3, 2, figsize=(12, 18))
    axes = axes.flatten()
    
    for ax, elemento in zip(axes, elements):
        if elemento not in targets.columns:
            continue
        
        values = targets[elemento].values
        lower = np.percentile(values, 1)
        upper = np.percentile(values, 99)
        values = np.clip(values, lower, upper)
        
        scatter = ax.scatter(
            projection_E[:, 0], projection_E[:, 1],
            c=values, cmap='hsv', alpha=0.3, s=10
        )
        
        cbar = fig.colorbar(scatter, ax=ax, orientation='vertical')
        cbar.set_label(elemento, rotation=270, labelpad=15)
        
        for i, (x, y) in enumerate(projection_prot):
            ax.text(x, y, i, color='black', fontsize=12, 
                   ha='center', va='center', weight='bold')
        
        ax.set_title(f"{projection_name} - {elemento}")
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Salvo: {output_path}")


def plot_cluster_histograms(targets, scores, output_dir, 
                            elements=['B', 'Cu', 'Zn', 'Fe', 'S', 'Mn']):
    """Create histograms by cluster for each element."""
    os.makedirs(output_dir, exist_ok=True)
    
    cluster_labels = np.argmax(scores, axis=1)
    targets_copy = targets.copy()
    targets_copy['cluster_label'] = cluster_labels
    
    for elemento in elements:
        if elemento not in targets.columns:
            continue
        
        all_clusters = sorted(targets_copy['cluster_label'].unique())
        num_clusters = len(all_clusters)
        
        ncols = 5
        nrows = int(np.ceil(num_clusters / ncols))
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 4))
        axes_flat = axes.flatten() if num_clusters > 1 else [axes]
        
        cluster_limits = {
            'x_min': targets_copy[elemento].min(),
            'x_max': targets_copy[elemento].max()
        }
        
        for i, cluster in enumerate(all_clusters):
            ax = axes_flat[i]
            cluster_data = targets_copy[targets_copy['cluster_label'] == cluster]
            
            ax.hist(cluster_data[elemento], bins=30, color='blue', alpha=0.7)
            cluster_mean = cluster_data[elemento].mean()
            ax.axvline(cluster_mean, color='red', linestyle='dotted', linewidth=2)
            
            ax.set_title(f'Cluster {cluster}')
            ax.set_xlabel(f'{elemento} Value', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.grid(True)
            ax.set_xlim(cluster_limits['x_min'], cluster_limits['x_max'])
        
        for j in range(num_clusters, nrows * ncols):
            axes_flat[j].axis('off')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'histograms_{elemento}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"✓ Histogramas salvos em: {output_dir}")


def visualize_model(model_name, embeddings_base='./embeddings', 
                   output_base='./downstream'):
    """Generate all visualizations for a model."""
    print(f"\n{'='*60}")
    print(f"Visualizando: {model_name}")
    print(f"{'='*60}")
    
    # Load data
    data = load_embeddings(model_name, embeddings_base)
    if data is None:
        return False
    
    # Create output folder
    output_dir = os.path.join(output_base, model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute projections
    projections = compute_tsne_umap(data['embeddings'], data['prototypes'])
    
    # Save projections
    np.save(os.path.join(output_dir, 'tsne_embeddings.npy'), projections['tsne_E'])
    np.save(os.path.join(output_dir, 'umap_embeddings.npy'), projections['umap_E'])
    
    if 'targets' in data:
        # Grid plots para t-SNE
        plot_element_grid(
            projections['tsne_E'], projections['tsne_prot'],
            data['targets'], 
            os.path.join(output_dir, 'tsne_grid.png'),
            projection_name='t-SNE'
        )
        
        # Grid plots for UMAP
        plot_element_grid(
            projections['umap_E'], projections['umap_prot'],
            data['targets'],
            os.path.join(output_dir, 'umap_grid.png'),
            projection_name='UMAP'
        )
        
        # Histograms by cluster
        hist_dir = os.path.join(output_dir, 'histograms')
        plot_cluster_histograms(data['targets'], data['scores'], hist_dir)
    
    print(f"✅ Visualizações concluídas para: {model_name}\n")
    return True


def main():
    parser = argparse.ArgumentParser(description='Generate embeddings visualizations')
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
        
        print(f"Found {len(models)} models to visualize")
        
        success = 0
        for model in models:
            if visualize_model(model, args.embeddings, args.output):
                success += 1
        
        print(f"\n{'='*60}")
        print(f"Visualizations complete: {success}/{len(models)}")
        print(f"{'='*60}")
    else:
        visualize_model(args.model, args.embeddings, args.output)


if __name__ == '__main__':
    main()
