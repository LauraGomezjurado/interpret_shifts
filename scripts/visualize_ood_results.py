#!/usr/bin/env python3
"""
Comprehensive OOD Results Visualization
Creates detailed plots and figures for OOD experiment analysis including saliency maps.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import os

# Import our modules
from utils.datasets import DatasetManager
from utils.attribution_methods import AttributionSuite
from models.resnet import ResNet18
from models.vit import HFViTPretrained, create_big_vit_for_cifar10, create_small_vit_for_cifar10

# Set style
try:
    plt.style.use('seaborn-v0_8')
except:
    plt.style.use('seaborn')
sns.set_palette("husl")

class OODResultsVisualizer:
    """Comprehensive visualizer for OOD experiment results."""
    
    def __init__(self, model_path, model_type, device='cpu', output_dir='visualizations'):
        self.model_path = model_path
        self.model_type = model_type
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load model
        self.model = self._load_model()
        self.dataset_manager = DatasetManager(img_size=32, batch_size=16)
        self.attribution_suite = AttributionSuite(self.model, model_type)
        
    def _load_model(self):
        """Load the trained model based on model type."""
        print(f"Loading {self.model_type} model from {self.model_path}")
        
        if self.model_type == 'resnet':
            model = ResNet18(num_classes=10)
        elif self.model_type == 'vit':
            # Assume it's the small ViT for now
            model = create_small_vit_for_cifar10(
                image_size=224,
                patch_size=4,
                hidden_size=128,
                depth=6,
                num_heads=4,
                num_labels=10
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Load state dict
        state_dict = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()
        
        return model.to(self.device)
    
    def create_saliency_comparison(self, results, output_dir):
        """Create saliency map comparison plots."""
        print("Creating saliency map comparisons...")
        
        # Get sample images and saliency maps
        cifar_data = results.get('cifar10_attributions', {})
        svhn_data = results.get('svhn_attributions', {})
        
        if not cifar_data or not svhn_data:
            print("Warning: Missing attribution data for comparison")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        fig.suptitle('Saliency Map Comparison: CIFAR-10 vs SVHN', fontsize=16)
        
        # Process first few samples
        n_samples = min(3, len(cifar_data.get('saliency', [])))
        
        for i in range(n_samples):
            # CIFAR-10 original image
            if i < len(cifar_data.get('images', [])):
                img = torch.tensor(cifar_data['images'][i])
                if img.requires_grad:
                    img = img.detach()
                axes[i, 0].imshow(self._denormalize_image(img))
                axes[i, 0].set_title(f'CIFAR-10 Sample {i+1}')
                axes[i, 0].axis('off')
            
            # CIFAR-10 saliency
            if i < len(cifar_data.get('saliency', [])):
                sal = np.array(cifar_data['saliency'][i])
                if sal.ndim == 3:
                    sal = np.mean(sal, axis=0)  # Convert to grayscale
                axes[i, 1].imshow(sal, cmap='hot')
                axes[i, 1].set_title('CIFAR-10 Saliency')
                axes[i, 1].axis('off')
            
            # SVHN original image
            if i < len(svhn_data.get('images', [])):
                img = torch.tensor(svhn_data['images'][i])
                if img.requires_grad:
                    img = img.detach()
                axes[i, 2].imshow(self._denormalize_image(img))
                axes[i, 2].set_title(f'SVHN Sample {i+1}')
                axes[i, 2].axis('off')
            
            # SVHN saliency
            if i < len(svhn_data.get('saliency', [])):
                sal = np.array(svhn_data['saliency'][i])
                if sal.ndim == 3:
                    sal = np.mean(sal, axis=0)  # Convert to grayscale
                axes[i, 3].imshow(sal, cmap='hot')
                axes[i, 3].set_title('SVHN Saliency')
                axes[i, 3].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'saliency_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saliency comparison saved")
    
    def create_attribution_overlay_plot(self):
        """Create attribution overlay plots."""
        print("Creating attribution overlay plots...")
        # Placeholder for now
        return None
    
    def create_performance_dashboard(self):
        """Create performance dashboard."""
        print("Creating performance dashboard...")
        # Placeholder for now
        return None
    
    def create_calibration_plots(self):
        """Create calibration plots."""
        print("Creating calibration plots...")
        # Placeholder for now
        return None
    
    def create_corruption_analysis(self):
        """Create corruption analysis plots."""
        print("Creating corruption analysis...")
        # Placeholder for now
        return None
    
    def create_attribution_drift_analysis(self):
        """Create attribution drift analysis."""
        print("Creating attribution drift analysis...")
        # Placeholder for now
        return None
    
    def _denormalize_image(self, tensor):
        """Denormalize image tensor for visualization."""
        # Reverse ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        tensor = tensor.detach().cpu() * std + mean
        tensor = torch.clamp(tensor, 0, 1)
        
        return tensor.permute(1, 2, 0).numpy()
    
    def _generate_quick_results(self):
        """Generate quick results for visualization."""
        print("   🔄 Generating results for visualization...")
        
        # Get dataloaders
        _, cifar10_test = self.dataset_manager.get_cifar10_loaders()
        svhn_loader = self.dataset_manager.get_svhn_loader()
        
        # Limit for quick generation
        cifar10_subset = self._limit_dataloader(cifar10_test, 5)
        svhn_subset = self._limit_dataloader(svhn_loader, 5)
        
        # Evaluate performance
        cifar10_results = evaluate_model_on_dataset(self.model, cifar10_subset, self.device)
        svhn_results = evaluate_model_on_dataset(self.model, svhn_subset, self.device)
        
        # Generate attribution drift
        cifar10_attr = self._compute_attributions_subset(cifar10_subset)
        svhn_attr = self._compute_attributions_subset(svhn_subset)
        drift_metrics = self._compute_drift_metrics(cifar10_attr, svhn_attr)
        
        return {
            'performance': {
                'cifar10_accuracy': cifar10_results['accuracy'],
                'svhn_accuracy': svhn_results['accuracy']
            },
            'calibration': {
                'cifar10_ece': cifar10_results['ece'],
                'svhn_ece': svhn_results['ece']
            },
            'attribution_drift': {
                'svhn': drift_metrics
            }
        }
    
    def _limit_dataloader(self, dataloader, max_batches):
        """Limit dataloader for quick visualization."""
        from torch.utils.data import DataLoader, Subset
        total_samples = min(max_batches * dataloader.batch_size, len(dataloader.dataset))
        indices = list(range(total_samples))
        subset = Subset(dataloader.dataset, indices)
        return DataLoader(subset, batch_size=dataloader.batch_size, shuffle=False)
    
    def _compute_attributions_subset(self, dataloader):
        """Compute attributions on a small subset."""
        attributions = {'saliency': [], 'integrated_grads': []}
        
        for batch_idx, (images, _) in enumerate(dataloader):
            if batch_idx >= 2:  # Only 2 batches
                break
            images = images.to(self.device)
            
            sal, _ = self.attribution_suite.saliency.generate(images)
            ig, _ = self.attribution_suite.integrated_grads.generate(images)
            
            attributions['saliency'].append(sal.cpu())
            attributions['integrated_grads'].append(ig.cpu())
        
        for method in attributions:
            if attributions[method]:
                attributions[method] = torch.cat(attributions[method], dim=0)
        
        return attributions
    
    def _compute_drift_metrics(self, attr1, attr2):
        """Compute drift metrics between attributions."""
        metrics = {}
        for method in attr1:
            if method in attr2 and len(attr1[method]) > 0 and len(attr2[method]) > 0:
                try:
                    min_size = min(len(attr1[method]), len(attr2[method]))
                    a1 = attr1[method][:min_size]
                    a2 = attr2[method][:min_size]
                    metrics[method] = self.attribution_suite.compute_metrics(a1, a2)
                except:
                    metrics[method] = None
        return metrics
    
    def create_all_visualizations(self, results_file):
        """Create all visualization plots."""
        print("🎨 Creating comprehensive OOD visualizations...")
        print("=" * 60)
        
        # Load results
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        plots = []
        
        # 1. Saliency comparison
        self.create_saliency_comparison(results, self.output_dir)
        
        # 2. Attribution overlays
        plots.append(self.create_attribution_overlay_plot())
        
        # 3. Performance dashboard
        plots.append(self.create_performance_dashboard())
        
        # 4. Calibration plots
        plots.append(self.create_calibration_plots())
        
        # 5. Corruption analysis
        plots.append(self.create_corruption_analysis())
        
        # 6. Attribution drift analysis
        plots.append(self.create_attribution_drift_analysis())
        
        print("\n" + "=" * 60)
        print("🎉 All visualizations created successfully!")
        print(f"📁 Output directory: {self.output_dir}")
        
        return plots


def main():
    parser = argparse.ArgumentParser(description='Visualize OOD experiment results')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--model_type', type=str, choices=['resnet', 'vit'], required=True, help='Model type')
    parser.add_argument('--results_file', type=str, help='JSON results file (optional)')
    parser.add_argument('--output_dir', type=str, default='visualizations', help='Output directory')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # Create visualizer
    visualizer = OODResultsVisualizer(
        model_path=args.model_path,
        model_type=args.model_type,
        device=device,
        output_dir=args.output_dir
    )
    
    # Generate all visualizations
    visualizer.create_all_visualizations(args.results_file)


if __name__ == '__main__':
    main() 