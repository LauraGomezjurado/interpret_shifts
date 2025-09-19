#!/usr/bin/env python3
"""
Quick Start Example: Interpreting Distribution Shifts

This script demonstrates the basic workflow for:
1. Training a model on CIFAR-10
2. Evaluating on out-of-distribution data
3. Analyzing attribution drift

Usage:
    python examples/quick_start.py --model vit --epochs 10
    python examples/quick_start.py --model resnet --epochs 20
"""

import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.training.main import main as train_model
from experiments.ood_analysis.run_ood_experiments import main as run_ood_analysis
from experiments.visualization.visualize_ood_results import main as visualize_results


def parse_args():
    parser = argparse.ArgumentParser(description='Quick Start: Distribution Shift Analysis')
    parser.add_argument('--model', type=str, default='vit-hf-scratch-small',
                       choices=['vit-hf-scratch-small', 'vit-hf-scratch', 'resnet'],
                       help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--ood_dataset', type=str, default='cifar100',
                       choices=['cifar100', 'svhn'],
                       help='Out-of-distribution dataset for evaluation')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training and use existing model')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("🚀 Starting Distribution Shift Analysis Demo")
    print(f"Model: {args.model}")
    print(f"OOD Dataset: {args.ood_dataset}")
    print(f"Epochs: {args.epochs}")
    print("-" * 50)
    
    # Step 1: Train model (if not skipping)
    if not args.skip_training:
        print("📚 Step 1: Training model on CIFAR-10...")
        train_args = [
            '--model', args.model,
            '--epochs', str(args.epochs),
            '--batch_size', str(args.batch_size),
            '--lr', str(args.lr)
        ]
        
        # Simulate training by calling the training script
        print(f"Training command: python experiments/training/main.py {' '.join(train_args)}")
        print("✅ Training completed (simulated)")
    else:
        print("⏭️  Skipping training (using existing model)")
    
    # Step 2: Run OOD analysis
    print("\n🔍 Step 2: Analyzing out-of-distribution performance...")
    ood_args = [
        '--model', args.model.replace('-hf-scratch-small', ''),
        '--ood_dataset', args.ood_dataset
    ]
    
    print(f"OOD Analysis command: python experiments/ood_analysis/run_ood_experiments.py {' '.join(ood_args)}")
    print("✅ OOD analysis completed (simulated)")
    
    # Step 3: Generate visualizations
    print("\n📊 Step 3: Generating visualizations...")
    print("Visualization command: python experiments/visualization/visualize_ood_results.py")
    print("✅ Visualizations generated (simulated)")
    
    print("\n🎉 Demo completed!")
    print("\nNext steps:")
    print("1. Check results/consolidated/ for generated files")
    print("2. View visualizations in the results directory")
    print("3. Read the analysis reports in docs/reports/")
    print("4. Try different models and OOD datasets")


if __name__ == '__main__':
    main()
