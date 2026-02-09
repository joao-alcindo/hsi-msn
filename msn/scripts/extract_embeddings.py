"""
Script to extract and save embeddings from trained models.

Usage:
    python scripts/extract_embeddings.py --model proto11_embed32_encoder_hsi
    python scripts/extract_embeddings.py --all  # Process all models
"""

import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
import argparse
import json
from tqdm import tqdm
from types import SimpleNamespace

# Add root directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hsi_msn import MSNModel
from src.dataset import init_data


def extract_embeddings_from_model(model_name, output_base='./embeddings', checkpoint_name='min_loss_checkpoint.pth'):
    """
    Extract embeddings from a specific model.
    
    Args:
        model_name: Name of the model folder in output/
        output_base: Base directory to save embeddings
        checkpoint_name: Name of the checkpoint to load
    """
    print(f"\n{'='*60}")
    print(f"Processing model: {model_name}")
    print(f"{'='*60}")
    
    # Paths
    model_folder = os.path.join('output', model_name)
    config_path = os.path.join(model_folder, 'config.yaml')
    checkpoint_path = os.path.join(model_folder, checkpoint_name)
    
    # Check if they exist
    if not os.path.exists(config_path):
        print(f"⚠️  Config not found: {config_path}")
        return False
    if not os.path.exists(checkpoint_path):
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
        return False
    
    # Load configuration
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
        config = SimpleNamespace(**config_dict)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = MSNModel(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load dataset
    print("Loading dataset...")
    config_dict['shuffle'] = False
    dataset = init_data(config_dict)
    
    # Extract embeddings
    print("Extracting embeddings...")
    embeddings_list = []
    scores_list = []
    prototypes_t = None
    
    with torch.no_grad():
        for views in tqdm(dataset, desc="Processing batches"):
            views = views[0].to(device)
            
            # Normalized embeddings
            z_views = model.target_encoder(views, mask_ratio=0.0)
            z_views = torch.nn.functional.normalize(z_views, dim=1)
            
            if prototypes_t is None:
                prototypes_t = model.prototypes.detach().to(device).float()
                if prototypes_t.shape[0] == z_views.shape[1]:
                    pass
                elif prototypes_t.shape[1] == z_views.shape[1]:
                    prototypes_t = prototypes_t.t()
                else:
                    raise ValueError(
                        "Prototype shape does not match embedding dimension: "
                        f"prototypes={tuple(prototypes_t.shape)}, embed_dim={z_views.shape[1]}"
                    )
                prototypes_t = torch.nn.functional.normalize(prototypes_t, dim=0)

            # Scores (projection onto prototypes)
            scores = z_views @ prototypes_t
            
            embeddings_list.append(z_views.cpu().numpy())
            scores_list.append(scores.cpu().numpy())
    
    # Concatenate everything
    embeddings = np.vstack(embeddings_list)
    scores = np.vstack(scores_list)
    
    print(f"✓ Embeddings extracted: {embeddings.shape}")
    print(f"✓ Scores extracted: {scores.shape}")
    prototypes = prototypes_t.detach().cpu().numpy()
    print(f"✓ Prototypes: {prototypes.shape}")
    
    # Create output folder
    output_dir = os.path.join(output_base, model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save files
    np.save(os.path.join(output_dir, 'embeddings.npy'), embeddings)
    np.save(os.path.join(output_dir, 'scores.npy'), scores)
    np.save(os.path.join(output_dir, 'prototypes.npy'), prototypes)
    
    # Save metadata
    metadata = {
        'model_name': model_name,
        'checkpoint': checkpoint_name,
        'epoch': checkpoint.get('epoch', 'unknown'),
        'num_samples': embeddings.shape[0],
        'embed_dim': embeddings.shape[1],
        'num_prototypes': prototypes.shape[1],
        'config': config_dict
    }
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Load and save ground truth (if it exists)
    gt_path = './../train_gt.csv'
    if os.path.exists(gt_path):
        train_gt = pd.read_csv(gt_path)
        train_gt = train_gt.iloc[:len(embeddings)]
        train_gt.to_csv(os.path.join(output_dir, 'targets.csv'), index=False)
        print(f"✓ Ground truth saved: {len(train_gt)} samples")
    
    print(f"\n✅ Embeddings saved to: {output_dir}\n")
    return True


def main():
    parser = argparse.ArgumentParser(description='Extract embeddings from trained models')
    parser.add_argument('--model', type=str, help='Model name (folder in output/)')
    parser.add_argument('--all', action='store_true', help='Process all models')
    parser.add_argument('--checkpoint', type=str, default='min_loss_checkpoint.pth',
                       help='Checkpoint name (default: min_loss_checkpoint.pth)')
    parser.add_argument('--output', type=str, default='./embeddings',
                       help='Directory to save embeddings')
    
    args = parser.parse_args()
    
    # Check arguments
    if not args.model and not args.all:
        parser.error("Specify --model NAME or --all")
    
    # Process models
    if args.all:
        output_dir = './output'
        if not os.path.exists(output_dir):
            print(f"❌ Folder {output_dir} not found!")
            return
        
        models = [d for d in os.listdir(output_dir) 
                 if os.path.isdir(os.path.join(output_dir, d)) and 
                 not d.startswith('.')]
        
        print(f"Found {len(models)} models to process")
        
        success_count = 0
        for model in models:
            if extract_embeddings_from_model(model, args.output, args.checkpoint):
                success_count += 1
        
        print(f"\n{'='*60}")
        print(f"Processing complete: {success_count}/{len(models)} models")
        print(f"{'='*60}")
    else:
        extract_embeddings_from_model(args.model, args.output, args.checkpoint)


if __name__ == '__main__':
    main()
