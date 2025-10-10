#!/usr/bin/env python3
"""
Efficient OOD Experiment Runner
Focuses on the most critical experiments with SVHN as primary OOD dataset.
"""

import torch
import torch.nn as nn
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime

# Import our modules
from models.resnet import ResNet18
from models.vit import create_small_vit_for_cifar10
from utils.datasets import DatasetManager, evaluate_model_on_dataset
from utils.attribution_methods import AttributionSuite


def get_best_device(preferred_device='auto'):
    """Get the best available device for experiments."""
    if preferred_device != 'auto':
        device = torch.device(preferred_device)
        print(f"🎯 Using user-specified device: {device}")
        return device
    
    # Auto-select best device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"🚀 Using Apple Metal Performance Shaders (MPS): {device}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 Using CUDA GPU: {device}")
    else:
        device = torch.device("cpu")
        print(f"🔧 Using CPU: {device}")
    
    return device


def parse_args():
    parser = argparse.ArgumentParser(description='Run OOD experiments efficiently')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--model_type', type=str, choices=['resnet', 'vit'], required=True, help='Model architecture')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'mps', 'cuda'], help='Device to use')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory for results')
    parser.add_argument('--quick_test', action='store_true', help='Run on subset for quick testing')
    return parser.parse_args()


class OODExperimentRunner:
    """Efficient runner for OOD experiments."""
    
    def __init__(self, model, model_type, device, output_dir, quick_test=False):
        self.model = model.to(device)
        self.model_type = model_type
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.quick_test = quick_test
        
        # Initialize components
        self.dataset_manager = DatasetManager(img_size=32, batch_size=32 if quick_test else 64)
        self.attribution_suite = AttributionSuite(model, model_type)
        
        # Results storage
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'model_type': model_type,
            'device': str(device),
            'performance': {},
            'calibration': {},
            'attribution_drift': {},
            'sanity_checks': {}
        }
        
    def run_all_experiments(self):
        """Run the essential experiments efficiently."""
        print("🚀 Starting Efficient OOD Experiment Suite")
        print("=" * 60)
        
        # A4: Sanity checks first (B1-B4)
        print("\n1️⃣ Running Attribution Sanity Checks...")
        self.run_sanity_checks()
        
        # A3: Performance & Calibration (A1 subset + A3)
        print("\n2️⃣ Evaluating Model Performance & Calibration...")
        self.run_performance_evaluation()
        
        # C2: SVHN OOD Evaluation (most important)
        print("\n3️⃣ Running SVHN OOD Analysis...")
        self.run_svhn_ood_analysis()
        
        # C1: CIFAR-10-C subset (3 corruptions instead of 75)
        print("\n4️⃣ Running CIFAR-10-C Corruption Analysis...")
        self.run_corruption_analysis()
        
        # Save all results
        self.save_results()
        print(f"\n✅ All experiments completed! Results saved to {self.output_dir}")
        
    def run_sanity_checks(self):
        """B1-B4: Run attribution method sanity checks."""
        # Create simple linear model for saliency check
        simple_model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 10)
        ).to(self.device)
        
        sanity_results = self.attribution_suite.run_all_sanity_checks(simple_model)
        self.results['sanity_checks'] = sanity_results
        
    def run_performance_evaluation(self):
        """A1 + A3: Model performance and calibration on ID and key datasets."""
        # Get dataloaders
        _, cifar10_test = self.dataset_manager.get_cifar10_loaders()
        svhn_loader = self.dataset_manager.get_svhn_loader()
        
        # Limit samples for quick test
        if self.quick_test:
            cifar10_test = self._limit_dataloader(cifar10_test, max_batches=20)
            svhn_loader = self._limit_dataloader(svhn_loader, max_batches=20)
        
        print("   📊 Evaluating on CIFAR-10 (ID)...")
        cifar10_results = evaluate_model_on_dataset(self.model, cifar10_test, self.device)
        
        print("   📊 Evaluating on SVHN (OOD)...")
        svhn_results = evaluate_model_on_dataset(self.model, svhn_loader, self.device)
        
        self.results['performance'] = {
            'cifar10_accuracy': cifar10_results['accuracy'],
            'svhn_accuracy': svhn_results['accuracy'],
            'cifar10_samples': cifar10_results['total_samples'],
            'svhn_samples': svhn_results['total_samples']
        }
        
        self.results['calibration'] = {
            'cifar10_ece': cifar10_results['ece'],
            'svhn_ece': svhn_results['ece']
        }
        
        print(f"      ✅ CIFAR-10: {cifar10_results['accuracy']:.2f}% acc, {cifar10_results['ece']:.4f} ECE")
        print(f"      ✅ SVHN: {svhn_results['accuracy']:.2f}% acc, {svhn_results['ece']:.4f} ECE")
        
    def run_svhn_ood_analysis(self):
        """C2: Comprehensive SVHN attribution drift analysis."""
        # Get dataloaders
        _, cifar10_test = self.dataset_manager.get_cifar10_loaders()
        svhn_loader = self.dataset_manager.get_svhn_loader()
        
        if self.quick_test:
            cifar10_test = self._limit_dataloader(cifar10_test, max_batches=5)
            svhn_loader = self._limit_dataloader(svhn_loader, max_batches=5)
        
        # Compute attributions on both datasets
        print("   🧠 Computing attributions on CIFAR-10...")
        cifar10_attributions = self._compute_attributions_on_dataset(cifar10_test)
        
        print("   🧠 Computing attributions on SVHN...")
        svhn_attributions = self._compute_attributions_on_dataset(svhn_loader)
        
        # Compute drift metrics
        print("   📈 Computing attribution drift metrics...")
        drift_metrics = self._compute_attribution_drift(cifar10_attributions, svhn_attributions)
        
        self.results['attribution_drift']['svhn'] = drift_metrics
        
        # Print summary
        for method, metrics in drift_metrics.items():
            if metrics:
                print(f"      📊 {method}: IoU={metrics['iou']:.3f}, Pearson={metrics['pearson']:.3f}")
        
    def run_corruption_analysis(self):
        """C1: CIFAR-10-C analysis with subset of corruptions."""
        corruptions = ['gaussian_noise', 'brightness', 'contrast']  # Subset instead of all 75
        severities = [1, 3, 5]
        
        print(f"   🌪️  Testing {len(corruptions)} corruption types...")
        
        corruption_loaders = self.dataset_manager.get_corruption_loaders(corruptions, severities)
        
        # Get clean CIFAR-10 attributions for comparison
        _, cifar10_test = self.dataset_manager.get_cifar10_loaders()
        if self.quick_test:
            cifar10_test = self._limit_dataloader(cifar10_test, max_batches=3)
        
        clean_attributions = self._compute_attributions_on_dataset(cifar10_test)
        
        corruption_results = {}
        
        for corruption in corruptions:
            corruption_results[corruption] = {}
            for severity in severities:
                if self.quick_test and severity > 3:  # Skip highest severity in quick test
                    continue
                    
                print(f"      🔧 {corruption} severity {severity}...")
                
                loader = corruption_loaders[corruption][severity]
                if self.quick_test:
                    loader = self._limit_dataloader(loader, max_batches=3)
                
                # Performance
                perf_results = evaluate_model_on_dataset(self.model, loader, self.device)
                
                # Attribution drift
                corrupt_attributions = self._compute_attributions_on_dataset(loader)
                drift_metrics = self._compute_attribution_drift(clean_attributions, corrupt_attributions)
                
                corruption_results[corruption][severity] = {
                    'accuracy': perf_results['accuracy'],
                    'ece': perf_results['ece'],
                    'attribution_drift': drift_metrics
                }
                
                print(f"         📊 Acc: {perf_results['accuracy']:.1f}%, ECE: {perf_results['ece']:.3f}")
        
        self.results['attribution_drift']['corruptions'] = corruption_results
        
    def _compute_attributions_on_dataset(self, dataloader):
        """Compute attributions for all methods on a dataset."""
        attributions = {
            'saliency': [],
            'gradcam': [],
            'integrated_grads': []
        }
        
        if self.model_type == 'vit':
            attributions['attention_rollout'] = []
        
        self.model.eval()
        
        for batch_idx, (images, _) in enumerate(dataloader):
            if self.quick_test and batch_idx >= 3:  # Limit batches in quick test
                break
                
            images = images.to(self.device)
            
            # Saliency maps
            saliency, _ = self.attribution_suite.saliency.generate(images)
            attributions['saliency'].append(saliency.cpu())
            
            # Grad-CAM
            try:
                gradcam, _ = self.attribution_suite.gradcam.generate(images)
                attributions['gradcam'].append(gradcam.cpu())
            except Exception as e:
                print(f"⚠️  Grad-CAM failed: {e}")
                attributions['gradcam'].append(torch.zeros_like(saliency.cpu()))
            
            # Integrated Gradients
            try:
                ig, _ = self.attribution_suite.integrated_grads.generate(images)
                attributions['integrated_grads'].append(ig.cpu())
            except Exception as e:
                print(f"⚠️  Integrated Gradients failed: {e}")
                attributions['integrated_grads'].append(torch.zeros_like(saliency.cpu()))
            
            # Attention rollout (ViT only)
            if self.model_type == 'vit':
                try:
                    attention, _ = self.attribution_suite.attention_rollout.generate(images)
                    attributions['attention_rollout'].append(attention.cpu())
                except Exception as e:
                    print(f"⚠️  Attention rollout failed: {e}")
                    attributions['attention_rollout'].append(torch.zeros(images.size(0), 1, 8, 8))
        
        # Concatenate all batches
        for method in attributions:
            if attributions[method]:
                attributions[method] = torch.cat(attributions[method], dim=0)
            else:
                attributions[method] = torch.tensor([])
        
        return attributions
    
    def _compute_attribution_drift(self, attr1_dict, attr2_dict):
        """Compute drift metrics between two sets of attributions."""
        drift_metrics = {}
        
        for method in attr1_dict:
            if method in attr2_dict and len(attr1_dict[method]) > 0 and len(attr2_dict[method]) > 0:
                try:
                    # Take first batch for comparison if different sizes
                    min_size = min(len(attr1_dict[method]), len(attr2_dict[method]))
                    attr1 = attr1_dict[method][:min_size]
                    attr2 = attr2_dict[method][:min_size]
                    
                    # Compute metrics
                    metrics = self.attribution_suite.compute_metrics(attr1, attr2)
                    drift_metrics[method] = metrics
                except Exception as e:
                    print(f"⚠️  Failed to compute drift for {method}: {e}")
                    drift_metrics[method] = None
            else:
                drift_metrics[method] = None
        
        return drift_metrics
    
    def _limit_dataloader(self, dataloader, max_batches=10):
        """Limit dataloader to max_batches for quick testing."""
        from torch.utils.data import DataLoader, Subset
        
        # Create subset
        total_samples = min(max_batches * dataloader.batch_size, len(dataloader.dataset))
        indices = list(range(total_samples))
        subset = Subset(dataloader.dataset, indices)
        
        return DataLoader(subset, batch_size=dataloader.batch_size, shuffle=False)
    
    def save_results(self):
        """Save all results to files."""
        # Save JSON results
        results_file = self.output_dir / f'ood_results_{self.model_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save CSV summary
        self._save_csv_summary()
        
        print(f"📁 Results saved to: {results_file}")
    
    def _save_csv_summary(self):
        """Save a CSV summary of key metrics."""
        summary_data = []
        
        # Performance summary
        if 'performance' in self.results:
            perf = self.results['performance']
            summary_data.append({
                'experiment': 'performance',
                'dataset': 'cifar10',
                'metric': 'accuracy',
                'value': perf.get('cifar10_accuracy', 0)
            })
            summary_data.append({
                'experiment': 'performance',
                'dataset': 'svhn',
                'metric': 'accuracy',
                'value': perf.get('svhn_accuracy', 0)
            })
        
        # Calibration summary
        if 'calibration' in self.results:
            cal = self.results['calibration']
            summary_data.append({
                'experiment': 'calibration',
                'dataset': 'cifar10',
                'metric': 'ece',
                'value': cal.get('cifar10_ece', 0)
            })
            summary_data.append({
                'experiment': 'calibration',
                'dataset': 'svhn',
                'metric': 'ece',
                'value': cal.get('svhn_ece', 0)
            })
        
        df = pd.DataFrame(summary_data)
        csv_file = self.output_dir / f'summary_{self.model_type}.csv'
        df.to_csv(csv_file, index=False)
        print(f"📊 Summary CSV saved to: {csv_file}")


def main():
    args = parse_args()
    
    # Setup device
    device = get_best_device(args.device)
    
    print(f"🔧 Using device: {device}")
    
    # Load model
    if args.model_type == 'resnet':
        model = ResNet18(num_classes=10)
    elif args.model_type == 'vit':
        model = create_small_vit_for_cifar10()
    
    # Load trained weights
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"✅ Loaded model from {args.model_path}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Run experiments
    runner = OODExperimentRunner(
        model=model,
        model_type=args.model_type,
        device=device,
        output_dir=args.output_dir,
        quick_test=args.quick_test
    )
    
    runner.run_all_experiments()


if __name__ == '__main__':
    main() 