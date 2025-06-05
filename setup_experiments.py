#!/usr/bin/env python3
"""
Quick setup script for OOD experiments
Downloads datasets and sets up the environment
"""

import os
import torch
import torchvision
from pathlib import Path
import argparse
from utils.datasets import DatasetManager

def download_datasets(data_dir="./data"):
    """Download and verify required datasets."""
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    
    print("🔽 Downloading datasets...")
    
    # CIFAR-10 (already available in training)
    print("   📁 CIFAR-10...")
    try:
        torchvision.datasets.CIFAR10(root=data_path, train=False, download=True)
        print("      ✅ CIFAR-10 ready")
    except Exception as e:
        print(f"      ❌ CIFAR-10 failed: {e}")
    
    # SVHN
    print("   📁 SVHN...")
    try:
        torchvision.datasets.SVHN(root=data_path, split='test', download=True)
        print("      ✅ SVHN ready")
    except Exception as e:
        print(f"      ❌ SVHN failed: {e}")
    
    print("✅ Dataset download complete")

def verify_model_files():
    """Check if trained model files exist."""
    print("\n🔍 Checking for trained models...")
    
    model_files = [
        "checkpoints/resnet_final.pth",
        "checkpoints/vit_final.pth",
        "best_resnet_model.pth",
        "best_vit_model.pth"
    ]
    
    found_models = []
    for model_file in model_files:
        if Path(model_file).exists():
            found_models.append(model_file)
            print(f"      ✅ Found: {model_file}")
    
    if not found_models:
        print("      ⚠️  No trained models found. Please train models first:")
        print("         python main.py --model resnet --epochs 50")
        print("         python main.py --model vit --epochs 50")
    else:
        print(f"✅ Found {len(found_models)} trained model(s)")
    
    return found_models

def create_experiment_commands(model_files):
    """Generate ready-to-run experiment commands."""
    commands_file = Path("experiment_commands.txt")
    
    commands = []
    
    for model_file in model_files:
        model_type = "resnet" if "resnet" in model_file.lower() else "vit"
        
        # Full experiment
        commands.append(f"# Full {model_type.upper()} OOD experiment")
        commands.append(f"python run_ood_experiments.py --model_path {model_file} --model_type {model_type} --output_dir results/{model_type}")
        commands.append("")
        
        # Quick test
        commands.append(f"# Quick {model_type.upper()} test (for debugging)")
        commands.append(f"python run_ood_experiments.py --model_path {model_file} --model_type {model_type} --output_dir results/{model_type}_quick --quick_test")
        commands.append("")
    
    # Add comparison command
    if len(model_files) >= 2:
        commands.append("# Compare results after both experiments")
        commands.append("python compare_ood_results.py --results_dir results/")
        commands.append("")
    
    with open(commands_file, 'w') as f:
        f.write('\n'.join(commands))
    
    print(f"📝 Experiment commands saved to: {commands_file}")
    return commands_file

def test_attribution_methods():
    """Quick test of attribution methods."""
    print("\n🧪 Testing attribution methods...")
    
    try:
        from utils.attribution_methods import AttributionSuite
        from models.resnet import ResNet18
        
        # Create dummy model and input
        model = ResNet18(num_classes=10)
        suite = AttributionSuite(model, 'resnet')
        
        # Test with dummy input
        dummy_input = torch.randn(2, 3, 32, 32)
        
        # Test saliency
        try:
            saliency, _ = suite.saliency.generate(dummy_input)
            print("      ✅ Saliency maps working")
        except Exception as e:
            print(f"      ❌ Saliency failed: {e}")
        
        # Test simple model sanity check
        simple_model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(3*32*32, 10)
        )
        
        sanity_results = suite.run_all_sanity_checks(simple_model)
        if sanity_results:
            print("      ✅ Sanity checks working")
        else:
            print("      ⚠️  Sanity checks returned empty")
            
    except Exception as e:
        print(f"      ❌ Attribution test failed: {e}")

def main():
    parser = argparse.ArgumentParser(description='Setup OOD experiments')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--skip_download', action='store_true', help='Skip dataset download')
    parser.add_argument('--test_only', action='store_true', help='Only run tests')
    args = parser.parse_args()
    
    print("🚀 Setting up OOD experiments")
    print("=" * 50)
    
    if not args.test_only:
        # Download datasets
        if not args.skip_download:
            download_datasets(args.data_dir)
        
        # Check models
        model_files = verify_model_files()
        
        # Create experiment commands
        if model_files:
            create_experiment_commands(model_files)
    
    # Test attribution methods
    test_attribution_methods()
    
    print("\n✅ Setup complete!")
    
    if not args.test_only:
        print("\nNext steps:")
        print("1. Check experiment_commands.txt for ready-to-run commands")
        print("2. Start with a quick test: --quick_test flag")
        print("3. Run full experiments when ready")
        print("4. Results will be saved in results/ directory")

if __name__ == '__main__':
    main() 