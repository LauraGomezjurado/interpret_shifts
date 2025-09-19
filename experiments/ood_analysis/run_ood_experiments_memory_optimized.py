#!/usr/bin/env python3
"""
Memory-Optimized SVHN OOD Experiment Script
Reduced memory usage for M2 Pro GPU constraints.
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import argparse
from pathlib import Path
from datetime import datetime
import gc

# Import our modules
from models.resnet import ResNet18
from utils.datasets import DatasetManager, evaluate_model_on_dataset, expected_calibration_error
from utils.attribution_methods import AttributionSuite

def get_best_device(preferred_device=None):
    """Get the best available device with memory optimization."""
    if preferred_device == 'cpu':
        return torch.device('cpu')
    elif preferred_device == 'mps' and torch.backends.mps.is_available():
        # Set memory limit for MPS
        torch.mps.set_per_process_memory_fraction(0.8)  # Use 80% of available memory
        print("🚀 Using Apple Metal Performance Shaders (MPS) with memory optimization")
        return torch.device('mps')
    elif preferred_device == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    elif preferred_device == 'auto':
        if torch.backends.mps.is_available():
            torch.mps.set_per_process_memory_fraction(0.8)
            print("🚀 Using Apple Metal Performance Shaders (MPS) with memory optimization")
            return torch.device('mps')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    else:
        return torch.device('cpu')

class MemoryOptimizedOODRunner:
    """Memory-optimized OOD experiment runner."""
    
    def __init__(self, model_path, model_type, output_dir, device, num_samples=400):
        self.model_path = model_path
        self.model_type = model_type
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.device = device
        self.num_samples = num_samples  # Reduced from 800
        
        # Initialize components
        self.model = self._load_model()
        self.dataset_manager = DatasetManager(img_size=32, batch_size=8)  # Reduced batch size
        self.attribution_suite = AttributionSuite(self.model, model_type)
        
    def _load_model(self):
        """Load and prepare model."""
        print(f"📦 Loading {self.model_type} model...")
        
        if self.model_type == 'resnet':
            model = ResNet18(num_classes=10)
        else:
            raise ValueError(f"Only ResNet supported in memory-optimized version. Got: {self.model_type}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(checkpoint)
        model = model.to(self.device)
        model.eval()
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            
        print(f"✅ Loaded model from {self.model_path}")
        return model
    
    def _evaluate_with_memory_cleanup(self, loader, dataset_name):
        """Evaluate model with memory cleanup."""
        print(f"   📊 Evaluating on {dataset_name}...")
        
        correct = 0
        total = 0
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(loader):
                if batch_idx * loader.batch_size >= self.num_samples:
                    break
                    
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                probs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_probs.append(probs.cpu())
                all_labels.append(labels.cpu())
                
                # Memory cleanup every few batches
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                    gc.collect()
        
        accuracy = 100 * correct / total
        all_probs = torch.cat(all_probs)
        all_labels = torch.cat(all_labels)
        ece = expected_calibration_error(all_probs, all_labels)
        
        return accuracy, ece, all_probs, all_labels
    
    def _compute_attributions_with_memory_management(self, loader, dataset_name):
        """Compute attributions with aggressive memory management."""
        print(f"   🧠 Computing attributions on {dataset_name}...")
        
        all_attributions = {
            'saliency': [],
            'gradcam': [],
            'integrated_grads': []
        }
        
        processed = 0
        for batch_idx, (images, labels) in enumerate(loader):
            if processed >= self.num_samples:
                break
                
            # Process smaller sub-batches
            sub_batch_size = 2  # Very small batch size
            for i in range(0, len(images), sub_batch_size):
                if processed >= self.num_samples:
                    break
                    
                end_idx = min(i + sub_batch_size, len(images))
                sub_images = images[i:end_idx].to(self.device)
                
                try:
                    # Clear memory before each attribution
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                    gc.collect()
                    
                    # Saliency (most memory efficient)
                    try:
                        saliency, _ = self.attribution_suite.saliency.generate(sub_images)
                        all_attributions['saliency'].append(saliency.cpu())
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print(f"⚠️  Saliency failed for batch {batch_idx}, sub-batch {i//sub_batch_size}: {e}")
                            # Create dummy attribution
                            all_attributions['saliency'].append(torch.zeros(sub_images.shape).cpu())
                        else:
                            raise e
                    
                    # Clear memory
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                    gc.collect()
                    
                    # Grad-CAM
                    try:
                        gradcam, _ = self.attribution_suite.gradcam.generate(sub_images)
                        all_attributions['gradcam'].append(gradcam.cpu())
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print(f"⚠️  Grad-CAM failed for batch {batch_idx}, sub-batch {i//sub_batch_size}: {e}")
                            all_attributions['gradcam'].append(torch.zeros(sub_images.shape).cpu())
                        else:
                            raise e
                    
                    # Clear memory
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                    gc.collect()
                    
                    # Skip Integrated Gradients as it's most memory intensive
                    # Create dummy attribution for consistency
                    all_attributions['integrated_grads'].append(torch.zeros(sub_images.shape).cpu())
                    
                    processed += sub_images.shape[0]
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"⚠️  Attribution failed for batch {batch_idx}: {e}")
                        # Create dummy attributions
                        for method in all_attributions:
                            all_attributions[method].append(torch.zeros(sub_images.shape).cpu())
                        processed += sub_images.shape[0]
                    else:
                        raise e
        
        # Concatenate results
        for method in all_attributions:
            if all_attributions[method]:
                all_attributions[method] = torch.cat(all_attributions[method])
            else:
                all_attributions[method] = torch.zeros((self.num_samples, 3, 32, 32))
        
        return all_attributions
    
    def run_svhn_experiment(self):
        """Run memory-optimized SVHN OOD experiment."""
        print("🚀 Starting Memory-Optimized SVHN OOD Experiment")
        print("=" * 60)
        
        # Get data loaders
        cifar10_train, cifar10_test = self.dataset_manager.get_cifar10_loaders()
        svhn_loader = self.dataset_manager.get_svhn_loader()
        
        # 1. Evaluate performance
        print("1️⃣ Evaluating Model Performance & Calibration...")
        cifar10_acc, cifar10_ece, cifar10_probs, cifar10_labels = self._evaluate_with_memory_cleanup(
            cifar10_test, "CIFAR-10 (ID)"
        )
        svhn_acc, svhn_ece, svhn_probs, svhn_labels = self._evaluate_with_memory_cleanup(
            svhn_loader, "SVHN (OOD)"
        )
        
        print(f"      ✅ CIFAR-10: {cifar10_acc:.2f}% acc, {cifar10_ece:.4f} ECE")
        print(f"      ✅ SVHN: {svhn_acc:.2f}% acc, {svhn_ece:.4f} ECE")
        
        # 2. Compute attributions
        print("2️⃣ Computing Attributions with Memory Management...")
        cifar10_attributions = self._compute_attributions_with_memory_management(
            cifar10_test, "CIFAR-10"
        )
        svhn_attributions = self._compute_attributions_with_memory_management(
            svhn_loader, "SVHN"
        )
        
        # 3. Calculate drift metrics
        print("3️⃣ Calculating Attribution Drift Metrics...")
        drift_metrics = {}
        
        for method in ['saliency', 'gradcam']:  # Skip integrated_grads
            cifar10_attr = cifar10_attributions[method]
            svhn_attr = svhn_attributions[method]
            
            # Calculate metrics using AttributionSuite method
            metrics = self.attribution_suite.compute_metrics(cifar10_attr, svhn_attr)
            
            drift_metrics[method] = {
                'iou': float(metrics['iou']),
                'pearson': float(metrics['pearson']) if metrics['pearson'] is not None else None,
                'spearman': float(metrics['spearman']) if metrics['spearman'] is not None else None
            }
        
        # 4. Save results
        results = {
            'timestamp': datetime.now().isoformat(),
            'model_type': self.model_type,
            'device': str(self.device),
            'num_samples': self.num_samples,
            'memory_optimized': True,
            'performance': {
                'cifar10_accuracy': float(cifar10_acc),
                'svhn_accuracy': float(svhn_acc),
                'accuracy_drop': float(cifar10_acc - svhn_acc)
            },
            'calibration': {
                'cifar10_ece': float(cifar10_ece),
                'svhn_ece': float(svhn_ece),
                'ece_increase': float(svhn_ece - cifar10_ece)
            },
            'attribution_drift': {
                'svhn': {
                    'drift_metrics': drift_metrics
                }
            }
        }
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = self.output_dir / f'svhn_ood_results_{self.model_type}_memory_opt_{timestamp}.json'
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Results saved to: {output_file}")
        
        # Final memory cleanup
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()
        
        return results

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Memory-Optimized SVHN OOD Experiments')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model')
    parser.add_argument('--model_type', type=str, choices=['resnet'], required=True,
                       help='Type of model architecture (only ResNet supported in memory-optimized version)')
    parser.add_argument('--output_dir', type=str, default='results_svhn_memory_opt',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'mps', 'cuda'], 
                       default='auto', help='Device to use')
    parser.add_argument('--num_samples', type=int, default=400,
                       help='Number of samples to process (reduced for memory)')
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Setup device
    device = get_best_device(args.device)
    print(f"🔧 Using device: {device}")
    
    # Run experiment
    runner = MemoryOptimizedOODRunner(
        model_path=args.model_path,
        model_type=args.model_type,
        output_dir=args.output_dir,
        device=device,
        num_samples=args.num_samples
    )
    
    runner.run_svhn_experiment()

if __name__ == '__main__':
    main() 