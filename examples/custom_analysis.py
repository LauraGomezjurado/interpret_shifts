#!/usr/bin/env python3
"""
Custom Analysis Example: Advanced Distribution Shift Experiments

This script demonstrates how to:
1. Create custom attribution methods
2. Analyze specific model behaviors
3. Generate custom visualizations
4. Compare multiple models

Usage:
    python examples/custom_analysis.py --models vit resnet --ood_datasets cifar100 svhn
"""

import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.utils.attribution_methods import AttributionAnalyzer
from src.utils.datasets import load_ood_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Custom Distribution Shift Analysis')
    parser.add_argument('--models', nargs='+', default=['vit', 'resnet'],
                       help='Models to analyze')
    parser.add_argument('--ood_datasets', nargs='+', default=['cifar100', 'svhn'],
                       help='OOD datasets to test')
    parser.add_argument('--attribution_methods', nargs='+', 
                       default=['saliency', 'integrated_gradients'],
                       help='Attribution methods to use')
    parser.add_argument('--output_dir', type=str, default='custom_analysis_results',
                       help='Output directory for results')
    return parser.parse_args()


def analyze_attribution_consistency(model, ood_dataset, attribution_methods):
    """
    Analyze how consistent attribution methods are across different samples
    """
    print(f"🔍 Analyzing attribution consistency for {model} on {ood_dataset}")
    
    # Load OOD dataset
    ood_loader = load_ood_dataset(ood_dataset, batch_size=32)
    
    # Initialize attribution analyzer
    analyzer = AttributionAnalyzer(model)
    
    consistency_scores = {}
    
    for method in attribution_methods:
        print(f"  📊 Computing {method} attributions...")
        
        # Compute attributions for multiple samples
        attributions = []
        for batch_idx, (data, _) in enumerate(ood_loader):
            if batch_idx >= 5:  # Limit to first 5 batches for demo
                break
                
            attr = analyzer.compute_attribution(data, method=method)
            attributions.append(attr)
        
        # Compute consistency (example: correlation between attributions)
        if len(attributions) > 1:
            # Flatten attributions for correlation analysis
            flat_attrs = [attr.flatten() for attr in attributions]
            correlations = np.corrcoef(flat_attrs)
            consistency_scores[method] = np.mean(correlations[np.triu_indices_from(correlations, k=1)])
        else:
            consistency_scores[method] = 0.0
    
    return consistency_scores


def create_custom_visualization(results, output_dir):
    """
    Create custom visualizations comparing models and datasets
    """
    print(f"📊 Creating custom visualizations in {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Example: Model comparison heatmap
    models = list(results.keys())
    datasets = list(results[models[0]].keys())
    
    # Create a simple comparison plot
    fig, axes = plt.subplots(1, len(datasets), figsize=(5*len(datasets), 4))
    if len(datasets) == 1:
        axes = [axes]
    
    for i, dataset in enumerate(datasets):
        model_scores = [results[model][dataset]['consistency'] for model in models]
        axes[i].bar(models, model_scores)
        axes[i].set_title(f'Attribution Consistency on {dataset.upper()}')
        axes[i].set_ylabel('Consistency Score')
        axes[i].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualizations saved to {output_dir}")


def main():
    args = parse_args()
    
    print("🔬 Custom Distribution Shift Analysis")
    print(f"Models: {args.models}")
    print(f"OOD Datasets: {args.ood_datasets}")
    print(f"Attribution Methods: {args.attribution_methods}")
    print("-" * 60)
    
    # Initialize results structure
    results = {}
    
    # Analyze each model-dataset combination
    for model in args.models:
        print(f"\n🤖 Analyzing {model.upper()} model...")
        results[model] = {}
        
        for dataset in args.ood_datasets:
            print(f"  📊 Dataset: {dataset}")
            
            # Simulate analysis (in real implementation, load actual model)
            consistency_scores = analyze_attribution_consistency(
                model, dataset, args.attribution_methods
            )
            
            results[model][dataset] = {
                'consistency': np.mean(list(consistency_scores.values())),
                'method_scores': consistency_scores
            }
            
            print(f"    ✅ Average consistency: {results[model][dataset]['consistency']:.3f}")
    
    # Generate custom visualizations
    create_custom_visualization(results, args.output_dir)
    
    # Print summary
    print("\n📋 Analysis Summary:")
    print("-" * 40)
    for model in args.models:
        print(f"\n{model.upper()}:")
        for dataset in args.ood_datasets:
            score = results[model][dataset]['consistency']
            print(f"  {dataset}: {score:.3f}")
    
    print(f"\n🎉 Custom analysis completed!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
