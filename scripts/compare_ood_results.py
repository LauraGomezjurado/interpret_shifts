#!/usr/bin/env python3
"""
Compare OOD experiment results between models
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import numpy as np

def load_results(results_dir):
    """Load all result files from the results directory."""
    results_path = Path(results_dir)
    results = {}
    
    for json_file in results_path.rglob("ood_results_*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                model_type = data['model_type']
                results[model_type] = data
                print(f"✅ Loaded {model_type} results from {json_file}")
        except Exception as e:
            print(f"❌ Failed to load {json_file}: {e}")
    
    return results

def create_performance_comparison(results):
    """Create performance comparison plots."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy comparison
    models = list(results.keys())
    datasets = ['cifar10', 'svhn']
    
    accuracy_data = []
    calibration_data = []
    
    for model in models:
        if 'performance' in results[model]:
            perf = results[model]['performance']
            accuracy_data.append([
                perf.get('cifar10_accuracy', 0),
                perf.get('svhn_accuracy', 0)
            ])
        
        if 'calibration' in results[model]:
            cal = results[model]['calibration']
            calibration_data.append([
                cal.get('cifar10_ece', 0),
                cal.get('svhn_ece', 0)
            ])
    
    # Accuracy plot
    if accuracy_data:
        x = np.arange(len(datasets))
        width = 0.35
        
        for i, model in enumerate(models):
            axes[0].bar(x + i * width, accuracy_data[i], width, 
                       label=model.upper(), alpha=0.8)
        
        axes[0].set_xlabel('Dataset')
        axes[0].set_ylabel('Accuracy (%)')
        axes[0].set_title('Model Accuracy Comparison')
        axes[0].set_xticks(x + width / 2)
        axes[0].set_xticklabels(['CIFAR-10 (ID)', 'SVHN (OOD)'])
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # Calibration plot
    if calibration_data:
        for i, model in enumerate(models):
            axes[1].bar(x + i * width, calibration_data[i], width,
                       label=model.upper(), alpha=0.8)
        
        axes[1].set_xlabel('Dataset')
        axes[1].set_ylabel('Expected Calibration Error')
        axes[1].set_title('Model Calibration Comparison')
        axes[1].set_xticks(x + width / 2)
        axes[1].set_xticklabels(['CIFAR-10 (ID)', 'SVHN (OOD)'])
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_attribution_drift_heatmap(results):
    """Create heatmap of attribution drift metrics."""
    # Collect SVHN drift data
    drift_data = {}
    methods = ['saliency', 'gradcam', 'integrated_grads', 'attention_rollout']
    metrics = ['iou', 'pearson']
    
    for model, data in results.items():
        if 'attribution_drift' in data and 'svhn' in data['attribution_drift']:
            svhn_drift = data['attribution_drift']['svhn']
            for method in methods:
                if method in svhn_drift and svhn_drift[method]:
                    for metric in metrics:
                        if metric in svhn_drift[method]:
                            key = f"{model}_{method}_{metric}"
                            drift_data[key] = svhn_drift[method][metric]
    
    if not drift_data:
        print("⚠️  No attribution drift data found")
        return None
    
    # Create DataFrame for heatmap
    df_data = []
    for key, value in drift_data.items():
        parts = key.split('_')
        model = parts[0]
        method = parts[1]
        metric = parts[2]
        df_data.append({'Model': model.upper(), 'Method': method, 'Metric': metric, 'Value': value})
    
    df = pd.DataFrame(df_data)
    pivot_df = df.pivot_table(index=['Model', 'Method'], columns='Metric', values='Value')
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot_df, annot=True, cmap='viridis', ax=ax, fmt='.3f')
    ax.set_title('Attribution Drift: CIFAR-10 → SVHN')
    plt.tight_layout()
    
    return fig

def create_corruption_analysis(results):
    """Create corruption analysis plots."""
    corruption_data = []
    
    for model, data in results.items():
        if 'attribution_drift' in data and 'corruptions' in data['attribution_drift']:
            corruptions = data['attribution_drift']['corruptions']
            for corruption_type, severities in corruptions.items():
                for severity, metrics in severities.items():
                    corruption_data.append({
                        'Model': model.upper(),
                        'Corruption': corruption_type,
                        'Severity': severity,
                        'Accuracy': metrics['accuracy'],
                        'ECE': metrics['ece']
                    })
    
    if not corruption_data:
        print("⚠️  No corruption data found")
        return None
    
    df = pd.DataFrame(corruption_data)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy degradation
    sns.lineplot(data=df, x='Severity', y='Accuracy', hue='Model', 
                style='Corruption', markers=True, ax=axes[0])
    axes[0].set_title('Accuracy vs Corruption Severity')
    axes[0].set_xlabel('Corruption Severity')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].grid(True, alpha=0.3)
    
    # Calibration degradation
    sns.lineplot(data=df, x='Severity', y='ECE', hue='Model',
                style='Corruption', markers=True, ax=axes[1])
    axes[1].set_title('Calibration vs Corruption Severity')
    axes[1].set_xlabel('Corruption Severity')
    axes[1].set_ylabel('Expected Calibration Error')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_summary_table(results):
    """Create a summary table of all results."""
    summary_data = []
    
    for model, data in results.items():
        row = {'Model': model.upper()}
        
        # Performance
        if 'performance' in data:
            perf = data['performance']
            row['CIFAR-10 Acc (%)'] = f"{perf.get('cifar10_accuracy', 0):.1f}"
            row['SVHN Acc (%)'] = f"{perf.get('svhn_accuracy', 0):.1f}"
            row['Accuracy Drop (%)'] = f"{perf.get('cifar10_accuracy', 0) - perf.get('svhn_accuracy', 0):.1f}"
        
        # Calibration
        if 'calibration' in data:
            cal = data['calibration']
            row['CIFAR-10 ECE'] = f"{cal.get('cifar10_ece', 0):.3f}"
            row['SVHN ECE'] = f"{cal.get('svhn_ece', 0):.3f}"
        
        # Attribution drift (average across methods)
        if 'attribution_drift' in data and 'svhn' in data['attribution_drift']:
            svhn_drift = data['attribution_drift']['svhn']
            ious = []
            pearsons = []
            
            for method_data in svhn_drift.values():
                if method_data and 'iou' in method_data:
                    ious.append(method_data['iou'])
                if method_data and 'pearson' in method_data:
                    pearsons.append(method_data['pearson'])
            
            if ious:
                row['Avg IoU'] = f"{np.mean(ious):.3f}"
            if pearsons:
                row['Avg Pearson'] = f"{np.mean(pearsons):.3f}"
        
        summary_data.append(row)
    
    return pd.DataFrame(summary_data)

def main():
    parser = argparse.ArgumentParser(description='Compare OOD experiment results')
    parser.add_argument('--results_dir', type=str, default='results/', help='Results directory')
    parser.add_argument('--output_dir', type=str, default='comparison_plots/', help='Output directory for plots')
    args = parser.parse_args()
    
    print("📊 Comparing OOD Experiment Results")
    print("=" * 50)
    
    # Load results
    results = load_results(args.results_dir)
    
    if not results:
        print("❌ No results found!")
        return
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"\n📈 Creating comparison plots...")
    
    # Performance comparison
    print("   📊 Performance comparison...")
    perf_fig = create_performance_comparison(results)
    perf_fig.savefig(output_path / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(perf_fig)
    
    # Attribution drift heatmap
    print("   🔥 Attribution drift analysis...")
    drift_fig = create_attribution_drift_heatmap(results)
    if drift_fig:
        drift_fig.savefig(output_path / 'attribution_drift_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close(drift_fig)
    
    # Corruption analysis
    print("   🌪️  Corruption analysis...")
    corr_fig = create_corruption_analysis(results)
    if corr_fig:
        corr_fig.savefig(output_path / 'corruption_analysis.png', dpi=300, bbox_inches='tight')
        plt.close(corr_fig)
    
    # Summary table
    print("   📋 Creating summary table...")
    summary_df = create_summary_table(results)
    summary_df.to_csv(output_path / 'results_summary.csv', index=False)
    
    # Display summary
    print("\n📋 Results Summary:")
    print(summary_df.to_string(index=False))
    
    print(f"\n✅ Comparison complete! Plots saved to {output_path}")
    print(f"📁 Files created:")
    print(f"   - performance_comparison.png")
    print(f"   - attribution_drift_heatmap.png (if available)")
    print(f"   - corruption_analysis.png (if available)")
    print(f"   - results_summary.csv")

if __name__ == '__main__':
    main() 