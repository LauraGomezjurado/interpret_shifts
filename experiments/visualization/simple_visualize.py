import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from pathlib import Path

# Set style
try:
    plt.style.use('seaborn-v0_8')
except:
    plt.style.use('seaborn')
sns.set_palette("husl")

def load_results(results_file):
    """Load results from JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)

def create_performance_dashboard(results, output_dir):
    """Create a comprehensive performance dashboard."""
    print("Creating performance dashboard...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('OOD Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Accuracy comparison
    ax = axes[0, 0]
    datasets = ['CIFAR-10\n(ID)', 'SVHN\n(OOD)']
    accuracies = [
        results['performance']['cifar10_accuracy'],
        results['performance']['svhn_accuracy']
    ]
    bars = ax.bar(datasets, accuracies, color=['#2E8B57', '#DC143C'], alpha=0.8)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model Performance')
    ax.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. Calibration comparison
    ax = axes[0, 1]
    eces = [
        results['calibration']['cifar10_ece'],
        results['calibration']['svhn_ece']
    ]
    bars = ax.bar(datasets, eces, color=['#4169E1', '#FF6347'], alpha=0.8)
    ax.set_ylabel('Expected Calibration Error')
    ax.set_title('Model Calibration')
    
    for bar, ece in zip(bars, eces):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{ece:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Performance drop visualization
    ax = axes[0, 2]
    drop = accuracies[0] - accuracies[1]
    ax.bar(['Performance\nDrop'], [drop], color='#FF4500', alpha=0.8)
    ax.set_ylabel('Accuracy Drop (%)')
    ax.set_title('OOD Performance Degradation')
    ax.text(0, drop + 1, f'{drop:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 4. Attribution drift heatmap
    ax = axes[1, 0]
    if 'attribution_drift' in results and 'svhn' in results['attribution_drift']:
        drift_data = results['attribution_drift']['svhn']
        methods = []
        ious = []
        pearsons = []
        
        for method, metrics in drift_data.items():
            if metrics and 'iou' in metrics and 'pearson' in metrics:
                # Skip methods with NaN values
                if not (np.isnan(metrics['iou']) or np.isnan(metrics['pearson'])):
                    methods.append(method.replace('_', ' ').title())
                    ious.append(metrics['iou'])
                    pearsons.append(metrics['pearson'])
        
        if methods:
            x = np.arange(len(methods))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, ious, width, label='IoU', alpha=0.8)
            bars2 = ax.bar(x + width/2, pearsons, width, label='Pearson Correlation', alpha=0.8)
            
            ax.set_xlabel('Attribution Method')
            ax.set_ylabel('Similarity Score')
            ax.set_title('Attribution Drift: CIFAR-10 → SVHN')
            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, val in zip(bars1, ious):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8)
            for bar, val in zip(bars2, pearsons):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 5. Corruption robustness
    ax = axes[1, 1]
    if 'attribution_drift' in results and 'corruptions' in results['attribution_drift']:
        corr_data = results['attribution_drift']['corruptions']
        corruption_types = list(corr_data.keys())
        severities = [1, 3]  # Focus on these severities
        
        for i, corruption in enumerate(corruption_types):
            accs = []
            for sev in severities:
                if str(sev) in corr_data[corruption]:
                    accs.append(corr_data[corruption][str(sev)]['accuracy'])
                else:
                    accs.append(0)
            
            ax.plot(severities, accs, marker='o', label=corruption.replace('_', ' ').title())
        
        ax.set_xlabel('Corruption Severity')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Corruption Robustness')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 6. Model summary text
    ax = axes[1, 2]
    ax.axis('off')
    
    summary_text = f"""
MODEL SUMMARY: {results.get('model_type', 'Unknown').upper()}

🎯 Key Findings:
• Performance Drop: {drop:.1f}% (CIFAR-10 → SVHN)
• Calibration Degradation: {eces[1]/eces[0]:.1f}x increase in ECE
• Attribution Consistency: {'Low' if any(ious) and max(ious) < 0.3 else 'Moderate' if any(ious) and max(ious) < 0.6 else 'High'}

🚨 Risk Assessment: {'HIGH RISK' if drop > 50 else 'MODERATE RISK' if drop > 30 else 'LOW RISK'}

💡 Recommendations:
• {'Implement OOD detection before deployment' if drop > 50 else 'Monitor for distribution shifts'}
• {'Improve robustness training' if drop > 40 else 'Consider ensemble methods'}
• {'Add uncertainty quantification' if eces[1] > 0.3 else 'Current calibration acceptable'}
    """
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Performance dashboard saved")

def create_attribution_drift_plot(results, output_dir):
    """Create detailed attribution drift analysis."""
    print("Creating attribution drift analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Attribution Drift Analysis', fontsize=16, fontweight='bold')
    
    # SVHN drift
    ax = axes[0, 0]
    if 'attribution_drift' in results and 'svhn' in results['attribution_drift']:
        drift_data = results['attribution_drift']['svhn']
        methods = []
        ious = []
        
        for method, metrics in drift_data.items():
            if metrics and 'iou' in metrics and not np.isnan(metrics['iou']):
                methods.append(method.replace('_', ' ').title())
                ious.append(metrics['iou'])
        
        if methods:
            bars = ax.bar(methods, ious, alpha=0.8, color='skyblue')
            ax.set_ylabel('IoU Score')
            ax.set_title('SVHN Attribution Drift (IoU)')
            ax.set_xticklabels(methods, rotation=45, ha='right')
            
            for bar, val in zip(bars, ious):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Corruption severity analysis
    ax = axes[0, 1]
    if 'attribution_drift' in results and 'corruptions' in results['attribution_drift']:
        corr_data = results['attribution_drift']['corruptions']
        
        # Create heatmap of IoU scores across corruptions and severities
        corruption_types = list(corr_data.keys())
        severities = ['1', '3']
        
        heatmap_data = []
        for corruption in corruption_types:
            row = []
            for sev in severities:
                if sev in corr_data[corruption]:
                    # Get average IoU across attribution methods
                    attr_data = corr_data[corruption][sev].get('attribution_drift', {})
                    ious = []
                    for method, metrics in attr_data.items():
                        if metrics and 'iou' in metrics and not np.isnan(metrics['iou']):
                            ious.append(metrics['iou'])
                    row.append(np.mean(ious) if ious else 0)
                else:
                    row.append(0)
            heatmap_data.append(row)
        
        if heatmap_data:
            im = ax.imshow(heatmap_data, cmap='viridis', aspect='auto')
            ax.set_xticks(range(len(severities)))
            ax.set_xticklabels(severities)
            ax.set_yticks(range(len(corruption_types)))
            ax.set_yticklabels([c.replace('_', ' ').title() for c in corruption_types])
            ax.set_xlabel('Corruption Severity')
            ax.set_title('Attribution Consistency Heatmap')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Performance vs Attribution consistency
    ax = axes[1, 0]
    if 'attribution_drift' in results and 'corruptions' in results['attribution_drift']:
        corr_data = results['attribution_drift']['corruptions']
        
        accs = []
        avg_ious = []
        labels = []
        
        for corruption in corr_data.keys():
            for sev in ['1', '3']:
                if sev in corr_data[corruption]:
                    acc = corr_data[corruption][sev]['accuracy']
                    attr_data = corr_data[corruption][sev].get('attribution_drift', {})
                    ious = []
                    for method, metrics in attr_data.items():
                        if metrics and 'iou' in metrics and not np.isnan(metrics['iou']):
                            ious.append(metrics['iou'])
                    
                    if ious:
                        accs.append(acc)
                        avg_ious.append(np.mean(ious))
                        labels.append(f"{corruption.replace('_', ' ').title()} (Sev {sev})")
        
        if accs and avg_ious:
            scatter = ax.scatter(accs, avg_ious, alpha=0.7, s=100)
            ax.set_xlabel('Accuracy (%)')
            ax.set_ylabel('Average Attribution IoU')
            ax.set_title('Performance vs Attribution Consistency')
            ax.grid(True, alpha=0.3)
            
            # Add labels for points
            for i, label in enumerate(labels):
                ax.annotate(label, (accs[i], avg_ious[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    # Calculate summary stats
    svhn_drift = results.get('attribution_drift', {}).get('svhn', {})
    avg_iou = np.mean([m['iou'] for m in svhn_drift.values() 
                       if m and 'iou' in m and not np.isnan(m['iou'])])
    
    summary_stats = f"""
ATTRIBUTION DRIFT SUMMARY

📊 SVHN Drift Metrics:
• Average IoU: {avg_iou:.3f}
• Consistency Level: {'High' if avg_iou > 0.5 else 'Medium' if avg_iou > 0.3 else 'Low'}

🔍 Key Insights:
• Saliency shows {'good' if svhn_drift.get('saliency', {}).get('iou', 0) > 0.3 else 'poor'} consistency
• Integrated Gradients {'stable' if svhn_drift.get('integrated_grads', {}).get('iou', 0) > 0.3 else 'unstable'}
• GradCAM and Attention {'working' if svhn_drift.get('gradcam', {}).get('iou', 0) > 0 else 'not working'}

⚠️  Implications:
• {'High attribution drift indicates model brittleness' if avg_iou < 0.3 else 'Moderate drift suggests some robustness'}
• {'Consider ensemble methods for stability' if avg_iou < 0.4 else 'Current attributions reasonably stable'}
    """
    
    ax.text(0.05, 0.95, summary_stats, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/attribution_drift_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Attribution drift analysis saved")

def main():
    parser = argparse.ArgumentParser(description='Create visualizations from OOD experiment results')
    parser.add_argument('--results_file', type=str, required=True,
                        help='Path to the JSON results file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    print(f"Loading results from {args.results_file}")
    results = load_results(args.results_file)
    
    print("🎨 Creating visualizations...")
    print("=" * 50)
    
    # Create visualizations
    create_performance_dashboard(results, args.output_dir)
    create_attribution_drift_plot(results, args.output_dir)
    
    print("=" * 50)
    print("🎉 All visualizations created successfully!")
    print(f"📁 Output directory: {args.output_dir}")

if __name__ == "__main__":
    main() 