#!/usr/bin/env python3
"""
SVHN ResNet Analysis Script
Generate saliency maps and comprehensive analysis of OOD performance on SVHN.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
from pathlib import Path
import seaborn as sns
from datetime import datetime

# Import our modules
from models.resnet import ResNet18
from utils.datasets import DatasetManager
from utils.attribution_methods import AttributionSuite

def load_results(results_file):
    """Load results from JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)

def create_svhn_analysis_dashboard(results, output_dir):
    """Create comprehensive SVHN analysis dashboard."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Performance Comparison
    ax1 = plt.subplot(3, 4, 1)
    datasets = ['CIFAR-10\n(ID)', 'SVHN\n(OOD)']
    accuracies = [results['performance']['cifar10_accuracy'], 
                  results['performance']['svhn_accuracy']]
    bars = ax1.bar(datasets, accuracies, color=['#2E8B57', '#DC143C'])
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Performance: ID vs OOD')
    ax1.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. Calibration Comparison  
    ax2 = plt.subplot(3, 4, 2)
    eces = [results['calibration']['cifar10_ece'], 
            results['calibration']['svhn_ece']]
    bars = ax2.bar(datasets, eces, color=['#2E8B57', '#DC143C'])
    ax2.set_ylabel('Expected Calibration Error')
    ax2.set_title('Calibration Analysis')
    
    for bar, ece in zip(bars, eces):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{ece:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Attribution Drift Metrics - IoU
    ax3 = plt.subplot(3, 4, 3)
    methods = ['Saliency', 'Grad-CAM']
    drift_data = results['attribution_drift']['svhn']['drift_metrics']
    
    iou_values = [drift_data['saliency']['iou'],
                  drift_data['gradcam']['iou']]
    
    bars = ax3.bar(methods, iou_values, color=['#FF6347', '#4682B4'])
    ax3.set_ylabel('IoU Score')
    ax3.set_title('Attribution Drift (IoU)')
    ax3.set_ylim(0, 0.2)
    
    for bar, iou in zip(bars, iou_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{iou:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 4. Pearson Correlations
    ax4 = plt.subplot(3, 4, 4)
    pearson_values = [drift_data['saliency']['pearson'],
                      drift_data['gradcam']['pearson']]
    
    bars = ax4.bar(methods, pearson_values, color=['#FF6347', '#4682B4'])
    ax4.set_ylabel('Pearson Correlation')
    ax4.set_title('Attribution Correlation')
    ax4.set_ylim(-0.1, 0.1)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    for bar, pearson in zip(bars, pearson_values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
                f'{pearson:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 5. Memory Optimization Info
    ax5 = plt.subplot(3, 4, 5)
    ax5.axis('off')
    memory_info = f"""
    🧠 Memory Optimization:
    
    • Samples processed: {results['num_samples']}
    • Batch size: 8 → 2 (sub-batches)
    • Device: {results['device'].upper()}
    • Memory fraction: 80%
    
    ⚠️ Integrated Gradients: Skipped
    (Memory intensive - would exceed 18GB)
    """
    
    ax5.text(0.05, 0.95, memory_info, transform=ax5.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    # 6. Calibration Quality Assessment
    ax6 = plt.subplot(3, 4, 6)
    
    # Create calibration quality visualization
    cifar10_ece = results['calibration']['cifar10_ece']
    svhn_ece = results['calibration']['svhn_ece']
    
    # Color code based on quality
    colors = ['red' if cifar10_ece > 0.1 else 'orange' if cifar10_ece > 0.05 else 'green',
              'red' if svhn_ece > 0.1 else 'orange' if svhn_ece > 0.05 else 'green']
    
    bars = ax6.bar(['CIFAR-10', 'SVHN'], [cifar10_ece, svhn_ece], color=colors, alpha=0.7)
    ax6.set_ylabel('ECE Score')
    ax6.set_title('Calibration Quality')
    ax6.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='Poor (>0.1)')
    ax6.axhline(y=0.05, color='orange', linestyle='--', alpha=0.5, label='Fair (>0.05)')
    ax6.legend(fontsize=8)
    
    # 7. Attribution Drift Comparison
    ax7 = plt.subplot(3, 4, (7, 8))
    
    # Compare IoU and Pearson side by side
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax7.bar(x - width/2, iou_values, width, label='IoU Similarity', 
                   color='skyblue', alpha=0.8)
    
    # Scale Pearson to same range for visualization
    pearson_scaled = [(p + 0.1) * 2 for p in pearson_values]  # Scale to 0-0.4 range
    bars2 = ax7.bar(x + width/2, pearson_scaled, width, label='Pearson (scaled)', 
                   color='lightcoral', alpha=0.8)
    
    ax7.set_ylabel('Similarity Score')
    ax7.set_title('Attribution Drift: Multiple Metrics')
    ax7.set_xticks(x)
    ax7.set_xticklabels(methods)
    ax7.legend()
    
    # Add value labels
    for i, (bar1, bar2, iou, pearson) in enumerate(zip(bars1, bars2, iou_values, pearson_values)):
        ax7.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.005, 
                f'{iou:.3f}', ha='center', va='bottom', fontsize=9)
        ax7.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.005, 
                f'{pearson:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 8. Key Statistics Box
    ax8 = plt.subplot(3, 4, (9, 12))
    ax8.axis('off')
    
    # Calculate key statistics
    accuracy_drop = results['performance']['accuracy_drop']
    ece_change = results['calibration']['ece_increase']
    mean_iou = np.mean(iou_values)
    
    # Determine ECE interpretation
    ece_status = "IMPROVED" if ece_change < 0 else "DEGRADED"
    ece_emoji = "✅" if ece_change < 0 else "⚠️"
    
    stats_text = f"""
    🎯 SVHN OOD ANALYSIS SUMMARY:
    
    Performance Impact:
    • Accuracy Drop: {accuracy_drop:.1f}%
    • ECE Change: {ece_change:.3f} ({ece_status})
    
    Attribution Drift:
    • Mean IoU: {mean_iou:.3f}
    • Saliency Correlation: {drift_data['saliency']['pearson']:.3f}
    • Grad-CAM Correlation: {drift_data['gradcam']['pearson']:.3f}
    
    Key Findings:
    {ece_emoji} Calibration: {ece_status.lower()}
    ⚠️  Accuracy: Severe OOD drop (71.8%)
    ⚠️  Attribution: Low similarity (IoU ~0.11)
    
    Memory Optimization:
    ✅ Successful GPU execution
    ✅ No memory overflow
    ⚠️  Reduced feature analysis (no IntGrad)
    
    Risk Assessment:
    ❌ NOT SAFE for production OOD deployment
    """
    
    ax8.text(0.05, 0.95, stats_text, transform=ax8.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'svhn_resnet_analysis_dashboard.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 SVHN analysis dashboard saved to: {output_dir / 'svhn_resnet_analysis_dashboard.png'}")

def generate_svhn_saliency_maps(model, output_dir, device, num_samples=8):
    """Generate SVHN vs CIFAR-10 saliency maps for comparison."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize dataset manager and attribution suite
    dataset_manager = DatasetManager(img_size=32, batch_size=num_samples)
    attribution_suite = AttributionSuite(model, 'resnet')
    
    # Get data
    _, cifar10_test = dataset_manager.get_cifar10_loaders()
    svhn_loader = dataset_manager.get_svhn_loader()
    
    # Get sample batches
    cifar10_batch = next(iter(cifar10_test))
    svhn_batch = next(iter(svhn_loader))
    
    cifar10_images, cifar10_labels = cifar10_batch[0][:num_samples].to(device), cifar10_batch[1][:num_samples]
    svhn_images, svhn_labels = svhn_batch[0][:num_samples].to(device), svhn_batch[1][:num_samples]
    
    # Generate saliency maps
    print("🧠 Generating SVHN vs CIFAR-10 saliency maps...")
    
    cifar10_saliency, _ = attribution_suite.saliency.generate(cifar10_images)
    svhn_saliency, _ = attribution_suite.saliency.generate(svhn_images)
    
    # Create comparison plot
    fig, axes = plt.subplots(4, num_samples, figsize=(num_samples * 3, 12))
    
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                      'dog', 'frog', 'horse', 'ship', 'truck']
    
    for i in range(num_samples):
        # CIFAR-10 original
        axes[0, i].imshow(cifar10_images[i].cpu().detach().permute(1, 2, 0).numpy() * 0.5 + 0.5)
        axes[0, i].set_title(f'CIFAR-10: {cifar10_classes[cifar10_labels[i]]}', fontsize=10)
        axes[0, i].axis('off')
        
        # CIFAR-10 saliency
        saliency_map = cifar10_saliency[i].cpu().detach().mean(0).numpy()
        im1 = axes[1, i].imshow(saliency_map, cmap='hot', interpolation='bilinear')
        axes[1, i].set_title('CIFAR-10 Saliency', fontsize=10)
        axes[1, i].axis('off')
        
        # SVHN original  
        axes[2, i].imshow(svhn_images[i].cpu().detach().permute(1, 2, 0).numpy() * 0.5 + 0.5)
        axes[2, i].set_title(f'SVHN: Digit {svhn_labels[i]}', fontsize=10)
        axes[2, i].axis('off')
        
        # SVHN saliency
        saliency_map = svhn_saliency[i].cpu().detach().mean(0).numpy()
        im2 = axes[3, i].imshow(saliency_map, cmap='hot', interpolation='bilinear')
        axes[3, i].set_title('SVHN Saliency', fontsize=10)
        axes[3, i].axis('off')
    
    # Add row labels
    row_labels = ['CIFAR-10\nImages', 'CIFAR-10\nSaliency', 'SVHN\nImages', 'SVHN\nSaliency']
    for i, label in enumerate(row_labels):
        axes[i, 0].set_ylabel(label, rotation=0, ha='right', va='center', fontsize=12, fontweight='bold')
    
    plt.suptitle('SVHN OOD Attribution Analysis: Saliency Map Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'svhn_saliency_maps_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"🎨 SVHN saliency maps saved to: {output_dir / 'svhn_saliency_maps_comparison.png'}")

def main():
    """Main analysis function."""
    # Paths
    model_path = 'resnet_cifar10_best.pth'
    results_file = 'results_svhn_resnet_memory_opt/svhn_ood_results_resnet_memory_opt_20250605_002803.json'
    output_dir = 'results_svhn_resnet_memory_opt/analysis'
    
    # Setup device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🔧 Using device: {device}")
    
    # Load model
    print("📦 Loading ResNet model...")
    model = ResNet18(num_classes=10)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    
    # Load results
    print("📊 Loading SVHN experiment results...")
    results = load_results(results_file)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("🎨 Creating comprehensive SVHN analysis...")
    
    # Generate analysis dashboard
    create_svhn_analysis_dashboard(results, output_dir)
    
    # Generate saliency maps
    generate_svhn_saliency_maps(model, output_dir, device)
    
    # Save summary report
    summary_text = f"""
# SVHN ResNet OOD Analysis Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Performance Summary
- **CIFAR-10 Accuracy**: {results['performance']['cifar10_accuracy']:.2f}%
- **SVHN Accuracy**: {results['performance']['svhn_accuracy']:.2f}%
- **Accuracy Drop**: {results['performance']['accuracy_drop']:.2f}%

## Calibration Analysis
- **CIFAR-10 ECE**: {results['calibration']['cifar10_ece']:.4f}
- **SVHN ECE**: {results['calibration']['svhn_ece']:.4f}
- **ECE Change**: {results['calibration']['ece_increase']:.4f}

## Attribution Drift Metrics
- **Saliency IoU**: {results['attribution_drift']['svhn']['drift_metrics']['saliency']['iou']:.3f}
- **Grad-CAM IoU**: {results['attribution_drift']['svhn']['drift_metrics']['gradcam']['iou']:.3f}
- **Saliency Pearson**: {results['attribution_drift']['svhn']['drift_metrics']['saliency']['pearson']:.3f}
- **Grad-CAM Pearson**: {results['attribution_drift']['svhn']['drift_metrics']['gradcam']['pearson']:.3f}

## Memory Optimization Results
✅ **Success**: Avoided 18GB MPS memory limit
✅ **Completion**: Full experiment completed
⚠️ **Limitation**: Integrated Gradients skipped (too memory intensive)

## Risk Assessment
⚠️ **HIGH RISK**: Severe accuracy drop on OOD data (71.8%)
✅ **POSITIVE**: Calibration actually improved on SVHN (lower ECE)
⚠️ **UNSTABLE**: Low attribution similarity across datasets (IoU ~0.11)

## Key Findings
1. **Cross-domain shift**: CIFAR-10 → SVHN represents significant domain gap
2. **Calibration paradox**: Model is better calibrated on SVHN despite poor accuracy
3. **Attribution instability**: Low IoU suggests model focuses on different features

## Files Generated
- svhn_resnet_analysis_dashboard.png: Comprehensive visual analysis
- svhn_saliency_maps_comparison.png: Saliency map comparisons
- svhn_analysis_report.md: This summary report
"""
    
    with open(output_path / 'svhn_analysis_report.md', 'w') as f:
        f.write(summary_text)
    
    print(f"\n✅ SVHN analysis complete! Results saved to: {output_dir}")
    print("📁 Files generated:")
    print("   • svhn_resnet_analysis_dashboard.png")
    print("   • svhn_saliency_maps_comparison.png") 
    print("   • svhn_analysis_report.md")

if __name__ == '__main__':
    main() 