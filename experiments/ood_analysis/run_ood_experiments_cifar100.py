#!/usr/bin/env python3
"""
CIFAR-100 OOD Experiment Runner
Comprehensive OOD experiments using CIFAR-100 as the primary OOD dataset.
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
    parser = argparse.ArgumentParser(description='Run OOD experiments with CIFAR-100')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--model_type', type=str, choices=['resnet', 'vit'], required=True, help='Model architecture')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'mps', 'cuda'], help='Device to use')
    parser.add_argument('--output_dir', type=str, default='results_cifar100', help='Output directory for results')
    parser.add_argument('--quick_test', action='store_true', help='Run on subset for quick testing')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples for attribution analysis')
    return parser.parse_args()


class CIFAR100OODExperimentRunner:
    """Comprehensive OOD experiment runner using CIFAR-100."""
    
    def __init__(self, model, model_type, device, output_dir, quick_test=False, num_samples=1000):
        self.model = model.to(device)
        self.model_type = model_type
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.quick_test = quick_test
        self.num_samples = num_samples if not quick_test else 200
        
        # Initialize components
        self.dataset_manager = DatasetManager(img_size=32, batch_size=32 if quick_test else 64)
        self.attribution_suite = AttributionSuite(model, model_type)
        
        # Results storage
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'model_type': model_type,
            'device': str(device),
            'num_samples': self.num_samples,
            'performance': {},
            'calibration': {},
            'attribution_drift': {},
            'sanity_checks': {},
            'semantic_analysis': {}
        }
        
        # CIFAR-10 and CIFAR-100 class mappings for semantic analysis
        self.cifar10_classes = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        
        self.cifar100_classes = [
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
            'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
            'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
            'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
            'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
            'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
            'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
            'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
            'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
            'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
            'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
            'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
            'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
            'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
            'worm'
        ]
        
    def run_all_experiments(self):
        """Run comprehensive CIFAR-100 OOD experiments."""
        print("🚀 Starting CIFAR-100 OOD Experiment Suite")
        print("=" * 60)
        
        # 1. Sanity checks
        print("\n1️⃣ Running Attribution Sanity Checks...")
        self.run_sanity_checks()
        
        # 2. Performance & Calibration
        print("\n2️⃣ Evaluating Model Performance & Calibration...")
        self.run_performance_evaluation()
        
        # 3. CIFAR-100 OOD Analysis (main experiment)
        print("\n3️⃣ Running CIFAR-100 OOD Analysis...")
        self.run_cifar100_ood_analysis()
        
        # 4. Semantic coherence analysis
        print("\n4️⃣ Analyzing Semantic Coherence...")
        self.run_semantic_analysis()
        
        # 5. CIFAR-10-C corruption analysis
        print("\n5️⃣ Running CIFAR-10-C Corruption Analysis...")
        self.run_corruption_analysis()
        
        # 6. Cross-dataset comparison (SVHN for comparison)
        print("\n6️⃣ Running Cross-Dataset Comparison...")
        self.run_cross_dataset_comparison()
        
        # Save all results
        self.save_results()
        print(f"\n✅ All experiments completed! Results saved to {self.output_dir}")
        
    def run_sanity_checks(self):
        """Run attribution method sanity checks."""
        # Create simple linear model for saliency check
        simple_model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 10)
        ).to(self.device)
        
        sanity_results = self.attribution_suite.run_all_sanity_checks(simple_model)
        self.results['sanity_checks'] = sanity_results
        
    def run_performance_evaluation(self):
        """Evaluate model performance and calibration on ID and OOD datasets."""
        # Get dataloaders
        _, cifar10_test = self.dataset_manager.get_cifar10_loaders()
        cifar100_loader = self.dataset_manager.get_cifar100_loader()
        
        # Limit samples for quick test
        if self.quick_test:
            cifar10_test = self._limit_dataloader(cifar10_test, max_batches=20)
            cifar100_loader = self._limit_dataloader(cifar100_loader, max_batches=20)
        
        print("   📊 Evaluating on CIFAR-10 (ID)...")
        cifar10_results = evaluate_model_on_dataset(self.model, cifar10_test, self.device)
        
        print("   📊 Evaluating on CIFAR-100 (OOD)...")
        cifar100_results = evaluate_model_on_dataset(self.model, cifar100_loader, self.device)
        
        self.results['performance'] = {
            'cifar10_accuracy': cifar10_results['accuracy'],
            'cifar100_accuracy': cifar100_results['accuracy'],
            'cifar10_samples': cifar10_results['total_samples'],
            'cifar100_samples': cifar100_results['total_samples'],
            'accuracy_drop': cifar10_results['accuracy'] - cifar100_results['accuracy']
        }
        
        self.results['calibration'] = {
            'cifar10_ece': cifar10_results['ece'],
            'cifar100_ece': cifar100_results['ece'],
            'ece_increase': cifar100_results['ece'] - cifar10_results['ece']
        }
        
        print(f"      ✅ CIFAR-10: {cifar10_results['accuracy']:.2f}% acc, {cifar10_results['ece']:.4f} ECE")
        print(f"      ✅ CIFAR-100: {cifar100_results['accuracy']:.2f}% acc, {cifar100_results['ece']:.4f} ECE")
        print(f"      📉 Accuracy drop: {self.results['performance']['accuracy_drop']:.2f}%")
        
    def run_cifar100_ood_analysis(self):
        """Comprehensive CIFAR-100 attribution drift analysis."""
        # Get dataloaders
        _, cifar10_test = self.dataset_manager.get_cifar10_loaders()
        cifar100_loader = self.dataset_manager.get_cifar100_loader()
        
        # Limit for analysis
        cifar10_limited = self._limit_dataloader_by_samples(cifar10_test, self.num_samples)
        cifar100_limited = self._limit_dataloader_by_samples(cifar100_loader, self.num_samples)
        
        # Compute attributions on both datasets
        print("   🧠 Computing attributions on CIFAR-10...")
        cifar10_attributions = self._compute_attributions_on_dataset(cifar10_limited)
        
        print("   🧠 Computing attributions on CIFAR-100...")
        cifar100_attributions = self._compute_attributions_on_dataset(cifar100_limited)
        
        # Compute drift metrics
        print("   📈 Computing attribution drift metrics...")
        drift_metrics = self._compute_attribution_drift(cifar10_attributions, cifar100_attributions)
        
        # Additional CIFAR-100 specific metrics
        print("   🔍 Computing CIFAR-100 specific metrics...")
        cifar100_metrics = self._compute_cifar100_specific_metrics(
            cifar10_attributions, cifar100_attributions, cifar10_limited, cifar100_limited
        )
        
        self.results['attribution_drift']['cifar100'] = {
            'drift_metrics': drift_metrics,
            'specific_metrics': cifar100_metrics
        }
        
        # Print summary
        for method, metrics in drift_metrics.items():
            if metrics:
                print(f"      📊 {method}: IoU={metrics.get('iou', 0):.3f}, Pearson={metrics.get('pearson', 0):.3f}")
        
    def run_semantic_analysis(self):
        """Analyze semantic coherence of predictions."""
        print("   🔍 Analyzing prediction semantic coherence...")
        
        cifar100_loader = self.dataset_manager.get_cifar100_loader()
        cifar100_limited = self._limit_dataloader_by_samples(cifar100_loader, 500)
        
        semantic_results = {
            'prediction_distribution': {},
            'confidence_analysis': {},
            'semantic_categories': {}
        }
        
        self.model.eval()
        all_predictions = []
        all_confidences = []
        all_true_labels = []
        
        with torch.no_grad():
            for images, labels in cifar100_limited:
                images = images.to(self.device)
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)
                confidences = probs.max(dim=1)[0]
                
                all_predictions.extend(preds.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
                all_true_labels.extend(labels.numpy())
        
        # Analyze prediction distribution
        pred_counts = np.bincount(all_predictions, minlength=10)
        semantic_results['prediction_distribution'] = {
            self.cifar10_classes[i]: int(count) for i, count in enumerate(pred_counts)
        }
        
        # Confidence analysis
        semantic_results['confidence_analysis'] = {
            'mean_confidence': float(np.mean(all_confidences)),
            'std_confidence': float(np.std(all_confidences)),
            'low_confidence_ratio': float(np.mean(np.array(all_confidences) < 0.5))
        }
        
        # Semantic category mapping
        semantic_mapping = self._analyze_semantic_mapping(all_true_labels, all_predictions)
        semantic_results['semantic_categories'] = semantic_mapping
        
        self.results['semantic_analysis'] = semantic_results
        
        print(f"      📊 Mean confidence: {semantic_results['confidence_analysis']['mean_confidence']:.3f}")
        print(f"      📊 Low confidence ratio: {semantic_results['confidence_analysis']['low_confidence_ratio']:.3f}")
        
    def run_corruption_analysis(self):
        """CIFAR-10-C analysis with subset of corruptions."""
        corruptions = ['gaussian_noise', 'brightness', 'contrast', 'fog', 'frost']
        severities = [1, 3, 5]
        
        print(f"   🌪️  Testing {len(corruptions)} corruption types...")
        
        try:
            corruption_loaders = self.dataset_manager.get_corruption_loaders(corruptions, severities)
        except Exception as e:
            print(f"   ⚠️  Corruption data not available: {e}")
            self.results['attribution_drift']['corruptions'] = "Not available"
            return
        
        # Get clean CIFAR-10 attributions for comparison
        _, cifar10_test = self.dataset_manager.get_cifar10_loaders()
        cifar10_limited = self._limit_dataloader_by_samples(cifar10_test, 300)
        clean_attributions = self._compute_attributions_on_dataset(cifar10_limited)
        
        corruption_results = {}
        
        for corruption in corruptions:
            corruption_results[corruption] = {}
            for severity in severities:
                if self.quick_test and severity > 3:
                    continue
                    
                print(f"      🔧 {corruption} severity {severity}...")
                
                try:
                    loader = corruption_loaders[corruption][severity]
                    loader_limited = self._limit_dataloader_by_samples(loader, 300)
                    
                    # Performance
                    perf_results = evaluate_model_on_dataset(self.model, loader_limited, self.device)
                    
                    # Attribution drift
                    corrupt_attributions = self._compute_attributions_on_dataset(loader_limited)
                    drift_metrics = self._compute_attribution_drift(clean_attributions, corrupt_attributions)
                    
                    corruption_results[corruption][severity] = {
                        'accuracy': perf_results['accuracy'],
                        'ece': perf_results['ece'],
                        'attribution_drift': drift_metrics
                    }
                    
                    print(f"         📊 Acc: {perf_results['accuracy']:.1f}%, ECE: {perf_results['ece']:.3f}")
                    
                except Exception as e:
                    print(f"         ❌ Failed: {e}")
                    corruption_results[corruption][severity] = None
        
        self.results['attribution_drift']['corruptions'] = corruption_results
        
    def run_cross_dataset_comparison(self):
        """Compare CIFAR-100 with SVHN for contrast."""
        print("   🔄 Comparing CIFAR-100 vs SVHN drift...")
        
        try:
            # Get SVHN data
            svhn_loader = self.dataset_manager.get_svhn_loader()
            svhn_limited = self._limit_dataloader_by_samples(svhn_loader, 300)
            
            # Get CIFAR-10 baseline
            _, cifar10_test = self.dataset_manager.get_cifar10_loaders()
            cifar10_limited = self._limit_dataloader_by_samples(cifar10_test, 300)
            cifar10_attributions = self._compute_attributions_on_dataset(cifar10_limited)
            
            # SVHN attributions
            print("     🧠 Computing SVHN attributions...")
            svhn_attributions = self._compute_attributions_on_dataset(svhn_limited)
            svhn_drift = self._compute_attribution_drift(cifar10_attributions, svhn_attributions)
            
            # Compare with CIFAR-100 results
            cifar100_drift = self.results['attribution_drift']['cifar100']['drift_metrics']
            
            comparison = {}
            for method in cifar100_drift:
                if method in svhn_drift and cifar100_drift[method] and svhn_drift[method]:
                    comparison[method] = {
                        'cifar100_iou': cifar100_drift[method].get('iou', 0),
                        'svhn_iou': svhn_drift[method].get('iou', 0),
                        'cifar100_pearson': cifar100_drift[method].get('pearson', 0),
                        'svhn_pearson': svhn_drift[method].get('pearson', 0)
                    }
            
            self.results['attribution_drift']['cross_dataset_comparison'] = comparison
            
            print("     📊 Cross-dataset comparison completed")
            
        except Exception as e:
            print(f"     ⚠️  Cross-dataset comparison failed: {e}")
            self.results['attribution_drift']['cross_dataset_comparison'] = "Failed"
    
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
            images = images.to(self.device)
            
            # Saliency maps
            try:
                saliency, _ = self.attribution_suite.saliency.generate(images)
                attributions['saliency'].append(saliency.cpu())
            except Exception as e:
                print(f"⚠️  Saliency failed: {e}")
                attributions['saliency'].append(torch.zeros(images.size(0), *images.shape[1:]))
            
            # Grad-CAM
            try:
                gradcam, _ = self.attribution_suite.gradcam.generate(images)
                attributions['gradcam'].append(gradcam.cpu())
            except Exception as e:
                print(f"⚠️  Grad-CAM failed: {e}")
                attributions['gradcam'].append(torch.zeros(images.size(0), *images.shape[1:]))
            
            # Integrated Gradients
            try:
                ig, _ = self.attribution_suite.integrated_grads.generate(images)
                attributions['integrated_grads'].append(ig.cpu())
            except Exception as e:
                print(f"⚠️  Integrated Gradients failed: {e}")
                attributions['integrated_grads'].append(torch.zeros(images.size(0), *images.shape[1:]))
            
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
                    # Take minimum size for comparison
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
    
    def _compute_cifar100_specific_metrics(self, cifar10_attr, cifar100_attr, cifar10_loader, cifar100_loader):
        """Compute CIFAR-100 specific analysis metrics."""
        metrics = {}
        
        # Attribution intensity comparison
        for method in cifar10_attr:
            if method in cifar100_attr and len(cifar10_attr[method]) > 0 and len(cifar100_attr[method]) > 0:
                cifar10_intensity = torch.abs(cifar10_attr[method]).mean().item()
                cifar100_intensity = torch.abs(cifar100_attr[method]).mean().item()
                
                metrics[f'{method}_intensity_ratio'] = cifar100_intensity / cifar10_intensity if cifar10_intensity > 0 else 0
                metrics[f'{method}_cifar10_intensity'] = cifar10_intensity
                metrics[f'{method}_cifar100_intensity'] = cifar100_intensity
        
        return metrics
    
    def _analyze_semantic_mapping(self, true_labels, predictions):
        """Analyze semantic relationships between CIFAR-100 true labels and CIFAR-10 predictions."""
        mapping = {}
        
        # Group by CIFAR-10 predictions
        for pred_class in range(10):
            pred_mask = np.array(predictions) == pred_class
            if np.sum(pred_mask) > 0:
                true_labels_for_pred = np.array(true_labels)[pred_mask]
                unique_labels, counts = np.unique(true_labels_for_pred, return_counts=True)
                
                # Top 3 most common CIFAR-100 classes for this CIFAR-10 prediction
                top_indices = np.argsort(counts)[-3:][::-1]
                top_classes = [(self.cifar100_classes[unique_labels[i]], int(counts[i])) for i in top_indices]
                
                mapping[self.cifar10_classes[pred_class]] = top_classes
        
        return mapping
    
    def _limit_dataloader(self, dataloader, max_batches=10):
        """Limit dataloader to max_batches for quick testing."""
        from torch.utils.data import DataLoader, Subset
        
        total_samples = min(max_batches * dataloader.batch_size, len(dataloader.dataset))
        indices = list(range(total_samples))
        subset = Subset(dataloader.dataset, indices)
        
        return DataLoader(subset, batch_size=dataloader.batch_size, shuffle=False)
    
    def _limit_dataloader_by_samples(self, dataloader, max_samples):
        """Limit dataloader to max_samples."""
        from torch.utils.data import DataLoader, Subset
        
        total_samples = min(max_samples, len(dataloader.dataset))
        indices = list(range(total_samples))
        subset = Subset(dataloader.dataset, indices)
        
        return DataLoader(subset, batch_size=dataloader.batch_size, shuffle=False)
    
    def save_results(self):
        """Save all results to files."""
        # Save JSON results
        results_file = self.output_dir / f'cifar100_ood_results_{self.model_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save detailed CSV summary
        self._save_detailed_csv_summary()
        
        print(f"📁 Results saved to: {results_file}")
    
    def _save_detailed_csv_summary(self):
        """Save a detailed CSV summary of all metrics."""
        summary_data = []
        
        # Performance metrics
        if 'performance' in self.results:
            perf = self.results['performance']
            for dataset in ['cifar10', 'cifar100']:
                summary_data.append({
                    'experiment': 'performance',
                    'dataset': dataset,
                    'metric': 'accuracy',
                    'value': perf.get(f'{dataset}_accuracy', 0)
                })
        
        # Calibration metrics
        if 'calibration' in self.results:
            cal = self.results['calibration']
            for dataset in ['cifar10', 'cifar100']:
                summary_data.append({
                    'experiment': 'calibration',
                    'dataset': dataset,
                    'metric': 'ece',
                    'value': cal.get(f'{dataset}_ece', 0)
                })
        
        # Attribution drift metrics
        if 'attribution_drift' in self.results and 'cifar100' in self.results['attribution_drift']:
            drift_data = self.results['attribution_drift']['cifar100']['drift_metrics']
            for method, metrics in drift_data.items():
                if metrics:
                    for metric_name, value in metrics.items():
                        summary_data.append({
                            'experiment': 'attribution_drift',
                            'dataset': 'cifar100',
                            'metric': f'{method}_{metric_name}',
                            'value': value
                        })
        
        df = pd.DataFrame(summary_data)
        csv_file = self.output_dir / f'cifar100_summary_{self.model_type}.csv'
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
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"✅ Loaded model from {args.model_path}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Run experiments
    runner = CIFAR100OODExperimentRunner(
        model=model,
        model_type=args.model_type,
        device=device,
        output_dir=args.output_dir,
        quick_test=args.quick_test,
        num_samples=args.num_samples
    )
    
    runner.run_all_experiments()


if __name__ == '__main__':
    main() 