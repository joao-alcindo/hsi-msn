"""
Script to list models and check processing status.

Usage:
    python scripts/list_models.py
"""

import os
import json
from datetime import datetime


def format_size(size_bytes):
    """Format size in bytes to readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def check_model_status(model_name):
    """Check processing status of a model."""
    status = {
        'trained': False,
        'embeddings': False,
        'visualized': False,
        'regression': False,
        'checkpoint': None,
        'epoch': None,
        'embed_dim': None,
        'num_prototypes': None
    }
    
    # Check training
    model_path = os.path.join('output', model_name)
    if os.path.exists(model_path):
        status['trained'] = True
        
        # Look for checkpoint
        for ckpt in ['min_loss_checkpoint.pth', 'last_checkpoint.pth']:
            if os.path.exists(os.path.join(model_path, ckpt)):
                status['checkpoint'] = ckpt
                break
    
    # Check embeddings
    emb_path = os.path.join('embeddings', model_name)
    if os.path.exists(emb_path):
        status['embeddings'] = True
        
        # Read metadata
        metadata_path = os.path.join(emb_path, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                meta = json.load(f)
                status['epoch'] = meta.get('epoch', 'N/A')
                status['embed_dim'] = meta.get('embed_dim', 'N/A')
                status['num_prototypes'] = meta.get('num_prototypes', 'N/A')
    
    # Check visualizations
    vis_path = os.path.join('downstream', model_name)
    if os.path.exists(vis_path):
        if os.path.exists(os.path.join(vis_path, 'tsne_grid.png')):
            status['visualized'] = True
        
        if os.path.exists(os.path.join(vis_path, 'xgboost_results.csv')):
            status['regression'] = True
    
    return status


def get_checkpoint_info(model_name):
    """Return information about available checkpoints."""
    model_path = os.path.join('output', model_name)
    if not os.path.exists(model_path):
        return []
    
    checkpoints = []
    for file in os.listdir(model_path):
        if file.endswith('.pth'):
            full_path = os.path.join(model_path, file)
            size = os.path.getsize(full_path)
            mtime = os.path.getmtime(full_path)
            checkpoints.append({
                'name': file,
                'size': format_size(size),
                'modified': datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
            })
    
    return sorted(checkpoints, key=lambda x: x['name'])


def main():
    print("\n" + "="*80)
    print("AVAILABLE MODELS - Processing Status")
    print("="*80 + "\n")
    
    # List models
    output_dir = './output'
    if not os.path.exists(output_dir):
        print("❌ Diretório 'output/' não encontrado!")
        return
    
    models = sorted([d for d in os.listdir(output_dir)
                    if os.path.isdir(os.path.join(output_dir, d)) and not d.startswith('.')])
    
    if not models:
        print("❌ Nenhum modelo encontrado em 'output/'")
        return
    
    print(f"Total models: {len(models)}\n")
    
    # Status table
    for i, model in enumerate(models, 1):
        status = check_model_status(model)
        
        print(f"{i}. {model}")
        print(f"   {'─'*70}")
        
        # Basic status
        trained = "✓" if status['trained'] else "✗"
        embeddings = "✓" if status['embeddings'] else "✗"
        visualized = "✓" if status['visualized'] else "✗"
        regression = "✓" if status['regression'] else "✗"
        
        print(f"   Status: Trained {trained} | Embeddings {embeddings} | Visualized {visualized} | Regression {regression}")
        
        # Additional info
        if status['embed_dim']:
            print(f"   Info: Dim={status['embed_dim']}, Prototypes={status['num_prototypes']}, Epoch={status['epoch']}")
        
        if status['checkpoint']:
            print(f"   Checkpoint: {status['checkpoint']}")
        
        # Available checkpoints
        checkpoints = get_checkpoint_info(model)
        if checkpoints:
            print(f"   Checkpoints ({len(checkpoints)}):")
            for ckpt in checkpoints[:3]:  # Mostra até 3
                print(f"      • {ckpt['name']} ({ckpt['size']}) - {ckpt['modified']}")
            if len(checkpoints) > 3:
                print(f"      ... e mais {len(checkpoints) - 3}")
        
        print()
    
    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    
    total = len(models)
    trained = sum(1 for m in models if check_model_status(m)['trained'])
    with_embeddings = sum(1 for m in models if check_model_status(m)['embeddings'])
    visualized = sum(1 for m in models if check_model_status(m)['visualized'])
    regressed = sum(1 for m in models if check_model_status(m)['regression'])
    
    print(f"Total models: {total}")
    print(f"  Trained: {trained}/{total}")
    print(f"  With embeddings: {with_embeddings}/{total}")
    print(f"  Visualized: {visualized}/{total}")
    print(f"  With regression: {regressed}/{total}")
    
    # Models needing processing
    need_processing = [m for m in models if not check_model_status(m)['regression']]
    if need_processing:
        print(f"\n⚠️  {len(need_processing)} models need processing:")
        for model in need_processing:
            status = check_model_status(model)
            missing = []
            if not status['embeddings']:
                missing.append('embeddings')
            if not status['visualized']:
                missing.append('visualization')
            if not status['regression']:
                missing.append('regression')
            
            print(f"  • {model} (missing: {', '.join(missing)})")
        
        print("\nTo process all:")
        print("  python scripts/extract_embeddings.py --all")
        print("  python scripts/visualize.py --all")
        print("  python scripts/regression.py --all")
    else:
        print("\n✅ All models have been processed!")
    
    print()


if __name__ == '__main__':
    main()
