#!/usr/bin/env python3
"""
Visualization script for CIFAR-100 OOD experiment results.
Generates charts and plots for the analysis report.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_results(results_file):
    """Load experiment results from JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)

def create_performance_comparison(results, output_dir):
    """Create performance comparison chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy comparison
    datasets = ['CIFAR-10\n(ID)', 'CIFAR-100\n(OOD)']
    accuracies = [results['performance']['cifar10_accuracy'], 
                  results['performance']['cifar100_accuracy']]
    
    bars1 = ax1.bar(datasets, accuracies, color=['#2E8B57', '#DC143C'], alpha=0.8)
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Model Accuracy: ID vs OOD')
    ax1.set_ylim(0, 80)
    
    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # ECE comparison
    eces = [results['calibration']['cifar10_ece'], 
            results['calibration']['cifar100_ece']]
    
    bars2 = ax2.bar(datasets, eces, color=['#2E8B57', '#DC143C'], alpha=0.8)
    ax2.set_ylabel('Expected Calibration Error')
    ax2.set_title('Model Calibration: ID vs OOD')
    ax2.set_ylim(0, 0.7)
    
    # Add value labels on bars
    for bar, ece in zip(bars2, eces):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{ece:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_attribution_drift_chart(results, output_dir):
    """Create attribution drift visualization."""
    drift_data = results['attribution_drift']['cifar100']['drift_metrics']
    
    methods = []
    ious = []
    pearsons = []
    
    for method, metrics in drift_data.items():
        if metrics and metrics['iou'] is not None:
            methods.append(method.replace('_', ' ').title())
            ious.append(metrics['iou'])
            pearsons.append(metrics.get('pearson', 0) if metrics.get('pearson') is not None else 0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # IoU similarity
    bars1 = ax1.bar(methods, ious, color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
    ax1.set_ylabel('IoU Similarity')
    ax1.set_title('Attribution Drift: IoU Similarity\n(Higher = More Similar)')
    ax1.set_ylim(0, 0.2)
    ax1.tick_params(axis='x', rotation=45)
    
    for bar, iou in zip(bars1, ious):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{iou:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Pearson correlation
    bars2 = ax2.bar(methods, pearsons, color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
    ax2.set_ylabel('Pearson Correlation')
    ax2.set_title('Attribution Drift: Pearson Correlation\n(Higher = More Correlated)')
    ax2.set_ylim(-0.05, 0.15)
    ax2.tick_params(axis='x', rotation=45)
    
    for bar, pearson in zip(bars2, pearsons):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{pearson:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'attribution_drift.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_semantic_analysis_chart(results, output_dir):
    """Create semantic analysis visualization."""
    pred_dist = results['semantic_analysis']['prediction_distribution']
    conf_analysis = results['semantic_analysis']['confidence_analysis']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Prediction distribution
    classes = list(pred_dist.keys())
    counts = list(pred_dist.values())
    
    bars1 = ax1.bar(classes, counts, color='skyblue', alpha=0.8)
    ax1.set_ylabel('Prediction Count')
    ax1.set_title('CIFAR-100 → CIFAR-10 Prediction Distribution')
    ax1.tick_params(axis='x', rotation=45)
    
    # Highlight animal vs vehicle classes
    animal_classes = ['bird', 'cat', 'deer', 'dog', 'frog', 'horse']
    for i, (bar, class_name) in enumerate(zip(bars1, classes)):
        if class_name in animal_classes:
            bar.set_color('#FF6B6B')  # Red for animals
        else:
            bar.set_color('#4ECDC4')  # Teal for vehicles/objects
    
    # Confidence analysis pie chart
    low_conf = conf_analysis['low_confidence_ratio']
    high_conf = 1 - low_conf
    
    ax2.pie([high_conf, low_conf], 
            labels=[f'High Confidence\n(≥0.5): {high_conf:.1%}', 
                   f'Low Confidence\n(<0.5): {low_conf:.1%}'],
            colors=['#FF6B6B', '#FFA07A'],
            autopct='%1.1f%%',
            startangle=90)
    ax2.set_title(f'Confidence Distribution\n(Mean: {conf_analysis["mean_confidence"]:.3f})')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'semantic_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_corruption_robustness_chart(results, output_dir):
    """Create corruption robustness visualization."""
    corruption_data = results['attribution_drift']['corruptions']
    
    # Extract data for plotting
    corruptions = []
    severities = []
    accuracies = []
    
    for corruption, severity_data in corruption_data.items():
        if corruption in ['fog', 'frost']:  # Skip problematic corruptions
            continue
        for severity, data in severity_data.items():
            if data:
                corruptions.append(corruption.replace('_', ' ').title())
                severities.append(f'Severity {severity}')
                accuracies.append(data['accuracy'])
    
    # Create DataFrame for easier plotting
    df = pd.DataFrame({
        'Corruption': corruptions,
        'Severity': severities,
        'Accuracy': accuracies
    })
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create grouped bar chart
    corruption_types = df['Corruption'].unique()
    severity_levels = ['Severity 1', 'Severity 3', 'Severity 5']
    
    x = np.arange(len(corruption_types))
    width = 0.25
    
    for i, severity in enumerate(severity_levels):
        severity_data = df[df['Severity'] == severity]
        values = [severity_data[severity_data['Corruption'] == corr]['Accuracy'].iloc[0] 
                 if len(severity_data[severity_data['Corruption'] == corr]) > 0 else 0 
                 for corr in corruption_types]
        
        bars = ax.bar(x + i*width, values, width, 
                     label=severity, alpha=0.8)
        
        # Add value labels
        for bar, val in zip(bars, values):
            if val > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model Robustness Under Corruptions')
    ax.set_xticks(x + width)
    ax.set_xticklabels(corruption_types)
    ax.legend()
    ax.set_ylim(0, 80)
    
    # Add baseline line
    baseline_acc = results['performance']['cifar10_accuracy']
    ax.axhline(y=baseline_acc, color='red', linestyle='--', alpha=0.7, 
               label=f'Clean Accuracy ({baseline_acc:.1f}%)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'corruption_robustness.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_dashboard(results, output_dir):
    """Create a comprehensive summary dashboard."""
    fig = plt.figure(figsize=(16, 12))
    
    # Create a 3x2 grid
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Performance metrics
    ax1 = fig.add_subplot(gs[0, 0])
    datasets = ['CIFAR-10', 'CIFAR-100']
    accuracies = [results['performance']['cifar10_accuracy'], 
                  results['performance']['cifar100_accuracy']]
    bars = ax1.bar(datasets, accuracies, color=['green', 'red'], alpha=0.7)
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Performance Comparison')
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. Calibration metrics
    ax2 = fig.add_subplot(gs[0, 1])
    eces = [results['calibration']['cifar10_ece'], 
            results['calibration']['cifar100_ece']]
    bars = ax2.bar(datasets, eces, color=['green', 'red'], alpha=0.7)
    ax2.set_ylabel('Expected Calibration Error')
    ax2.set_title('Calibration Quality')
    for bar, ece in zip(bars, eces):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{ece:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Attribution drift
    ax3 = fig.add_subplot(gs[1, :])
    drift_data = results['attribution_drift']['cifar100']['drift_metrics']
    methods = []
    ious = []
    for method, metrics in drift_data.items():
        if metrics and metrics['iou'] is not None:
            methods.append(method.replace('_', ' ').title())
            ious.append(metrics['iou'])
    
    bars = ax3.bar(methods, ious, color='orange', alpha=0.7)
    ax3.set_ylabel('IoU Similarity')
    ax3.set_title('Attribution Drift (Lower = More Drift)')
    ax3.tick_params(axis='x', rotation=45)
    for bar, iou in zip(bars, ious):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{iou:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Confidence analysis
    ax4 = fig.add_subplot(gs[2, 0])
    conf_analysis = results['semantic_analysis']['confidence_analysis']
    mean_conf = conf_analysis['mean_confidence']
    low_conf_ratio = conf_analysis['low_confidence_ratio']
    
    ax4.bar(['Mean\nConfidence', 'Low Confidence\nRatio'], 
            [mean_conf, low_conf_ratio], 
            color=['blue', 'purple'], alpha=0.7)
    ax4.set_ylabel('Value')
    ax4.set_title('Confidence Analysis on OOD Data')
    ax4.set_ylim(0, 1)
    
    # 5. Risk assessment
    ax5 = fig.add_subplot(gs[2, 1])
    risk_factors = ['Accuracy\nDrop', 'Calibration\nIncrease', 'Attribution\nDrift']
    risk_values = [
        results['performance']['accuracy_drop'] / 100,  # Normalize to 0-1
        min(results['calibration']['ece_increase'], 1),  # Cap at 1
        1 - np.mean(ious)  # Invert IoU to show drift
    ]
    
    colors = ['red' if val > 0.5 else 'orange' if val > 0.3 else 'green' for val in risk_values]
    bars = ax5.bar(risk_factors, risk_values, color=colors, alpha=0.7)
    ax5.set_ylabel('Risk Level (0-1)')
    ax5.set_title('Risk Assessment')
    ax5.set_ylim(0, 1)
    ax5.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='High Risk Threshold')
    
    plt.suptitle('CIFAR-100 OOD Analysis Dashboard\nVision Transformer Model', 
                 fontsize=16, fontweight='bold')
    
    plt.savefig(output_dir / 'summary_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Load results
    results_file = 'results/results_cifar100_vit/cifar100_ood_results_vit_20250604_215446.json'
    output_dir = Path('results/results_cifar100_vit/visualizations')
    output_dir.mkdir(exist_ok=True)
    
    print("📊 Loading CIFAR-100 OOD experiment results...")
    results = load_results(results_file)
    
    print("📈 Creating performance comparison chart...")
    create_performance_comparison(results, output_dir)
    
    print("📈 Creating attribution drift chart...")
    create_attribution_drift_chart(results, output_dir)
    
    print("📈 Creating semantic analysis chart...")
    create_semantic_analysis_chart(results, output_dir)
    
    print("📈 Creating corruption robustness chart...")
    create_corruption_robustness_chart(results, output_dir)
    
    print("📈 Creating summary dashboard...")
    create_summary_dashboard(results, output_dir)
    
    print(f"✅ All visualizations saved to {output_dir}")
    print("\nGenerated files:")
    for file in output_dir.glob("*.png"):
        print(f"  - {file.name}")

if __name__ == '__main__':
    main() 