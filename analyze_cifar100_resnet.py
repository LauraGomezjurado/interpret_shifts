#!/usr/bin/env python3
"""
CIFAR-100 ResNet Analysis Script
Generate saliency maps and comprehensive analysis of OOD performance.
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

def create_analysis_dashboard(results, output_dir):
    """Create comprehensive analysis dashboard."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Performance Comparison
    ax1 = plt.subplot(3, 4, 1)
    datasets = ['CIFAR-10', 'CIFAR-100']
    accuracies = [results['performance']['cifar10_accuracy'], 
                  results['performance']['cifar100_accuracy']]
    bars = ax1.bar(datasets, accuracies, color=['#2E8B57', '#DC143C'])
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Performance Comparison')
    ax1.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. Calibration Comparison  
    ax2 = plt.subplot(3, 4, 2)
    eces = [results['calibration']['cifar10_ece'], 
            results['calibration']['cifar100_ece']]
    bars = ax2.bar(datasets, eces, color=['#2E8B57', '#DC143C'])
    ax2.set_ylabel('Expected Calibration Error')
    ax2.set_title('Calibration Analysis')
    
    for bar, ece in zip(bars, eces):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{ece:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Attribution Drift Metrics
    ax3 = plt.subplot(3, 4, 3)
    methods = ['Saliency', 'Grad-CAM', 'Integrated\nGrads']
    drift_data = results['attribution_drift']['cifar100']['drift_metrics']
    
    iou_values = [drift_data['saliency']['iou'],
                  drift_data['gradcam']['iou'],
                  drift_data['integrated_grads']['iou']]
    
    bars = ax3.bar(methods, iou_values, color=['#FF6347', '#4682B4', '#32CD32'])
    ax3.set_ylabel('IoU Score')
    ax3.set_title('Attribution Drift (IoU)')
    ax3.set_ylim(0, 0.2)
    
    for bar, iou in zip(bars, iou_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{iou:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 4. Pearson Correlations
    ax4 = plt.subplot(3, 4, 4)
    pearson_values = [drift_data['saliency']['pearson'],
                      drift_data['gradcam']['pearson'],
                      drift_data['integrated_grads']['pearson']]
    
    bars = ax4.bar(methods, pearson_values, color=['#FF6347', '#4682B4', '#32CD32'])
    ax4.set_ylabel('Pearson Correlation')
    ax4.set_title('Attribution Correlation')
    ax4.set_ylim(-0.1, 0.1)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    for bar, pearson in zip(bars, pearson_values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
                f'{pearson:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 5. Intensity Ratios
    ax5 = plt.subplot(3, 4, 5)
    specific_metrics = results['attribution_drift']['cifar100']['specific_metrics']
    intensity_ratios = [specific_metrics['saliency_intensity_ratio'],
                       specific_metrics['gradcam_intensity_ratio'],
                       specific_metrics['integrated_grads_intensity_ratio']]
    
    bars = ax5.bar(methods, intensity_ratios, color=['#FF6347', '#4682B4', '#32CD32'])
    ax5.set_ylabel('Intensity Ratio (C100/C10)')
    ax5.set_title('Attribution Intensity Changes')
    ax5.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='No Change')
    
    for bar, ratio in zip(bars, intensity_ratios):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                f'{ratio:.2f}x', ha='center', va='bottom', fontsize=10)
    
    # 6. Corruption Robustness - Accuracy
    ax6 = plt.subplot(3, 4, 6)
    corruptions = results['attribution_drift']['corruptions']
    
    corruption_names = []
    severity_1_acc = []
    severity_5_acc = []
    
    for corruption, severities in corruptions.items():
        if '1' in severities and '5' in severities:
            corruption_names.append(corruption.replace('_', '\n'))
            severity_1_acc.append(severities['1']['accuracy'])
            severity_5_acc.append(severities['5']['accuracy'])
    
    x = np.arange(len(corruption_names))
    width = 0.35
    
    ax6.bar(x - width/2, severity_1_acc, width, label='Severity 1', color='#90EE90')
    ax6.bar(x + width/2, severity_5_acc, width, label='Severity 5', color='#FA8072')
    
    ax6.set_ylabel('Accuracy (%)')
    ax6.set_title('Corruption Robustness')
    ax6.set_xticks(x)
    ax6.set_xticklabels(corruption_names, fontsize=9)
    ax6.legend()
    
    # 7. Semantic Analysis - Prediction Distribution
    ax7 = plt.subplot(3, 4, (7, 8))
    if 'semantic_analysis' in results:
        pred_dist = results['semantic_analysis']['prediction_distribution']
        classes = list(pred_dist.keys())
        counts = list(pred_dist.values())
        
        bars = ax7.bar(classes, counts, color=plt.cm.tab10(np.arange(len(classes))))
        ax7.set_ylabel('Prediction Count')
        ax7.set_title('CIFAR-100 → CIFAR-10 Prediction Distribution')
        ax7.set_xticklabels(classes, rotation=45, ha='right')
        
        # Add counts on bars
        for bar, count in zip(bars, counts):
            ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    str(count), ha='center', va='bottom', fontsize=9)
    
    # 8. Key Statistics Box
    ax8 = plt.subplot(3, 4, (9, 12))
    ax8.axis('off')
    
    # Calculate key statistics
    accuracy_drop = results['performance']['accuracy_drop']
    ece_increase = results['calibration']['ece_increase']
    mean_iou = np.mean([drift_data['saliency']['iou'],
                       drift_data['gradcam']['iou'],
                       drift_data['integrated_grads']['iou']])
    
    stats_text = f"""
    🎯 KEY FINDINGS:
    
    Performance Impact:
    • Accuracy Drop: {accuracy_drop:.1f}%
    • ECE Increase: {ece_increase:.3f} ({ece_increase/results['calibration']['cifar10_ece']:.1f}x worse)
    
    Attribution Drift:
    • Mean IoU: {mean_iou:.3f}
    • Max Intensity Change: {max(intensity_ratios):.1f}x
    
    Risk Assessment:
    • Model Reliability: CRITICAL FAILURE
    • Calibration Status: SEVERELY DEGRADED  
    • Attribution Stability: LOW SIMILARITY
    
    Deployment Recommendation:
    ❌ NOT SAFE for production use on OOD data
    """
    
    ax8.text(0.05, 0.95, stats_text, transform=ax8.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cifar100_resnet_analysis_dashboard.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Analysis dashboard saved to: {output_dir / 'cifar100_resnet_analysis_dashboard.png'}")

def generate_saliency_maps(model, output_dir, device, num_samples=8):
    """Generate actual saliency maps for comparison."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize dataset manager and attribution suite
    dataset_manager = DatasetManager(img_size=32, batch_size=num_samples)
    attribution_suite = AttributionSuite(model, 'resnet')
    
    # Get data
    _, cifar10_test = dataset_manager.get_cifar10_loaders()
    cifar100_loader = dataset_manager.get_cifar100_loader()
    
    # Get sample batches
    cifar10_batch = next(iter(cifar10_test))
    cifar100_batch = next(iter(cifar100_loader))
    
    cifar10_images, cifar10_labels = cifar10_batch[0][:num_samples].to(device), cifar10_batch[1][:num_samples]
    cifar100_images, cifar100_labels = cifar100_batch[0][:num_samples].to(device), cifar100_batch[1][:num_samples]
    
    # Generate saliency maps
    print("🧠 Generating saliency maps...")
    
    cifar10_saliency, _ = attribution_suite.saliency.generate(cifar10_images)
    cifar100_saliency, _ = attribution_suite.saliency.generate(cifar100_images)
    
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
        
        # CIFAR-100 original
        axes[2, i].imshow(cifar100_images[i].cpu().detach().permute(1, 2, 0).numpy() * 0.5 + 0.5)
        axes[2, i].set_title(f'CIFAR-100: Class {cifar100_labels[i]}', fontsize=10)
        axes[2, i].axis('off')
        
        # CIFAR-100 saliency
        saliency_map = cifar100_saliency[i].cpu().detach().mean(0).numpy()
        im2 = axes[3, i].imshow(saliency_map, cmap='hot', interpolation='bilinear')
        axes[3, i].set_title('CIFAR-100 Saliency', fontsize=10)
        axes[3, i].axis('off')
    
    # Add row labels
    row_labels = ['CIFAR-10\nImages', 'CIFAR-10\nSaliency', 'CIFAR-100\nImages', 'CIFAR-100\nSaliency']
    for i, label in enumerate(row_labels):
        axes[i, 0].set_ylabel(label, rotation=0, ha='right', va='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'saliency_maps_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"🎨 Saliency maps saved to: {output_dir / 'saliency_maps_comparison.png'}")

def main():
    """Main analysis function."""
    # Paths
    model_path = 'resnet_cifar10_best.pth'
    results_file = 'results_cifar100_resnet_gpu/cifar100_ood_results_resnet_20250604_235634.json'
    output_dir = 'results_cifar100_resnet_gpu/analysis'
    
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
    print("📊 Loading experiment results...")
    results = load_results(results_file)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("🎨 Creating comprehensive analysis...")
    
    # Generate analysis dashboard
    create_analysis_dashboard(results, output_dir)
    
    # Generate saliency maps
    generate_saliency_maps(model, output_dir, device)
    
    # Save summary report
    summary_text = f"""
# CIFAR-100 ResNet OOD Analysis Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Performance Summary
- **CIFAR-10 Accuracy**: {results['performance']['cifar10_accuracy']:.2f}%
- **CIFAR-100 Accuracy**: {results['performance']['cifar100_accuracy']:.2f}%
- **Accuracy Drop**: {results['performance']['accuracy_drop']:.2f}%

## Calibration Analysis
- **CIFAR-10 ECE**: {results['calibration']['cifar10_ece']:.4f}
- **CIFAR-100 ECE**: {results['calibration']['cifar100_ece']:.4f}
- **ECE Increase**: {results['calibration']['ece_increase']:.4f}

## Attribution Drift Metrics
- **Saliency IoU**: {results['attribution_drift']['cifar100']['drift_metrics']['saliency']['iou']:.3f}
- **Grad-CAM IoU**: {results['attribution_drift']['cifar100']['drift_metrics']['gradcam']['iou']:.3f}
- **Integrated Gradients IoU**: {results['attribution_drift']['cifar100']['drift_metrics']['integrated_grads']['iou']:.3f}

## Risk Assessment
⚠️ **CRITICAL**: Model shows catastrophic failure on OOD data
⚠️ **HIGH RISK**: Severe calibration degradation
⚠️ **UNSTABLE**: Low attribution similarity across datasets

## Files Generated
- analysis_dashboard.png: Comprehensive visual analysis
- saliency_maps_comparison.png: Actual saliency map comparisons
- analysis_report.md: This summary report
"""
    
    with open(output_path / 'analysis_report.md', 'w') as f:
        f.write(summary_text)
    
    print(f"\n✅ Analysis complete! Results saved to: {output_dir}")
    print("📁 Files generated:")
    print("   • cifar100_resnet_analysis_dashboard.png")
    print("   • saliency_maps_comparison.png") 
    print("   • analysis_report.md")

if __name__ == '__main__':
    main() 