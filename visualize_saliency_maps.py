import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import argparse
import os
from models.vit import create_small_vit_for_cifar10
from utils.attribution_methods import SaliencyMaps, IntegratedGradients

# Set style
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'seaborn')
sns.set_palette("husl")

class SaliencyVisualizer:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        
        # Initialize attribution methods
        self.saliency = SaliencyMaps(model)
        self.integrated_gradients = IntegratedGradients(model)
        
        # CIFAR-10 class names
        self.cifar_classes = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
    
    def get_saliency_map(self, inputs, method='saliency'):
        """Generate saliency map for given inputs."""
        with torch.enable_grad():
            if method == 'saliency':
                saliency_map, _ = self.saliency.generate(inputs)
            elif method == 'integrated_gradients':
                saliency_map, _ = self.integrated_gradients.generate(inputs)
            else:
                raise ValueError(f"Unknown method: {method}")
        
        return saliency_map
    
    def normalize_saliency(self, saliency_map):
        """Normalize saliency map to [0, 1] range."""
        # Take absolute value and normalize per sample
        saliency_abs = torch.abs(saliency_map)
        batch_size = saliency_abs.size(0)
        
        normalized = torch.zeros_like(saliency_abs)
        for i in range(batch_size):
            sal = saliency_abs[i]
            sal_min, sal_max = sal.min(), sal.max()
            if sal_max > sal_min:
                normalized[i] = (sal - sal_min) / (sal_max - sal_min)
            else:
                normalized[i] = sal
        
        return normalized
    
    def create_overlay(self, image, saliency_map, alpha=0.4):
        """Create overlay of image and saliency map."""
        # Convert to numpy and transpose to HWC
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        if isinstance(saliency_map, torch.Tensor):
            saliency_map = saliency_map.detach().cpu().numpy()
        
        if image.ndim == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        if saliency_map.ndim == 3 and saliency_map.shape[0] == 3:
            saliency_map = np.mean(saliency_map, axis=0)
        elif saliency_map.ndim == 3:
            saliency_map = saliency_map[0]
        
        # Normalize image to [0, 1] range (handle normalized inputs)
        image_min, image_max = image.min(), image.max()
        if image_max > image_min:
            image = (image - image_min) / (image_max - image_min)
        
        # Create heatmap
        heatmap = plt.cm.jet(saliency_map)[:, :, :3]  # Remove alpha channel
        
        # Blend with original image
        overlay = (1 - alpha) * image + alpha * heatmap
        return np.clip(overlay, 0, 1)
    
    def visualize_comparison(self, id_data, ood_data, num_samples=8, method='saliency', save_path=None):
        """Create side-by-side comparison of ID vs OOD saliency maps."""
        
        # Get predictions and saliency maps
        id_images, id_labels = id_data
        ood_images, ood_labels = ood_data
        
        with torch.no_grad():
            id_preds = self.model(id_images.to(self.device)).argmax(dim=1)
            ood_preds = self.model(ood_images.to(self.device)).argmax(dim=1)
        
        id_saliency = self.get_saliency_map(id_images.to(self.device), method)
        ood_saliency = self.get_saliency_map(ood_images.to(self.device), method)
        
        # Normalize saliency maps
        id_saliency_norm = self.normalize_saliency(id_saliency)
        ood_saliency_norm = self.normalize_saliency(ood_saliency)
        
        # Create visualization
        fig, axes = plt.subplots(4, num_samples, figsize=(num_samples * 2.5, 10))
        
        for i in range(min(num_samples, id_images.size(0), ood_images.size(0))):
            # Normalize images for display
            id_img = id_images[i].detach().cpu().numpy()
            ood_img = ood_images[i].detach().cpu().numpy()
            
            # Convert to HWC and normalize to [0,1]
            id_img = np.transpose(id_img, (1, 2, 0))
            ood_img = np.transpose(ood_img, (1, 2, 0))
            
            # Normalize to [0,1] range
            id_img = (id_img - id_img.min()) / (id_img.max() - id_img.min())
            ood_img = (ood_img - ood_img.min()) / (ood_img.max() - ood_img.min())
            
            # ID data - original image
            axes[0, i].imshow(id_img)
            axes[0, i].set_title(f'ID: {self.cifar_classes[id_labels[i]]}\nPred: {self.cifar_classes[id_preds[i]]}', 
                               fontsize=10)
            axes[0, i].axis('off')
            
            # ID data - saliency overlay
            id_overlay = self.create_overlay(id_images[i], id_saliency_norm[i])
            axes[1, i].imshow(id_overlay)
            axes[1, i].set_title(f'ID {method.title()}', fontsize=10)
            axes[1, i].axis('off')
            
            # OOD data - original image
            axes[2, i].imshow(ood_img)
            axes[2, i].set_title(f'OOD: SVHN\nPred: {self.cifar_classes[ood_preds[i]]}', fontsize=10)
            axes[2, i].axis('off')
            
            # OOD data - saliency overlay
            ood_overlay = self.create_overlay(ood_images[i], ood_saliency_norm[i])
            axes[3, i].imshow(ood_overlay)
            axes[3, i].set_title(f'OOD {method.title()}', fontsize=10)
            axes[3, i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved comparison to {save_path}")
        
        return fig
    
    def create_saliency_statistics(self, id_data, ood_data, method='saliency', save_path=None):
        """Create statistical comparison of saliency maps."""
        
        id_images, _ = id_data
        ood_images, _ = ood_data
        
        # Generate saliency maps
        id_saliency = self.get_saliency_map(id_images.to(self.device), method)
        ood_saliency = self.get_saliency_map(ood_images.to(self.device), method)
        
        # Calculate statistics
        id_stats = {
            'mean': torch.abs(id_saliency).mean().item(),
            'std': torch.abs(id_saliency).std().item(),
            'max': torch.abs(id_saliency).max().item(),
            'sparsity': (torch.abs(id_saliency) < 0.01).float().mean().item()
        }
        
        ood_stats = {
            'mean': torch.abs(ood_saliency).mean().item(),
            'std': torch.abs(ood_saliency).std().item(),
            'max': torch.abs(ood_saliency).max().item(),
            'sparsity': (torch.abs(ood_saliency) < 0.01).float().mean().item()
        }
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Statistics comparison
        stats_names = list(id_stats.keys())
        id_values = list(id_stats.values())
        ood_values = list(ood_stats.values())
        
        x = np.arange(len(stats_names))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, id_values, width, label='ID (CIFAR-10)', alpha=0.8)
        axes[0, 0].bar(x + width/2, ood_values, width, label='OOD (SVHN)', alpha=0.8)
        axes[0, 0].set_xlabel('Statistics')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].set_title(f'{method.title()} Map Statistics')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(stats_names)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Distribution comparison
        id_flat = torch.abs(id_saliency).flatten().detach().cpu().numpy()
        ood_flat = torch.abs(ood_saliency).flatten().detach().cpu().numpy()
        
        axes[0, 1].hist(id_flat, bins=50, alpha=0.7, label='ID (CIFAR-10)', density=True)
        axes[0, 1].hist(ood_flat, bins=50, alpha=0.7, label='OOD (SVHN)', density=True)
        axes[0, 1].set_xlabel('Saliency Magnitude')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Saliency Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Spatial concentration
        id_spatial = torch.abs(id_saliency).mean(dim=(0, 1)).detach().cpu().numpy()
        ood_spatial = torch.abs(ood_saliency).mean(dim=(0, 1)).detach().cpu().numpy()
        
        im1 = axes[1, 0].imshow(id_spatial, cmap='hot', interpolation='nearest')
        axes[1, 0].set_title('ID Spatial Concentration')
        axes[1, 0].axis('off')
        plt.colorbar(im1, ax=axes[1, 0])
        
        im2 = axes[1, 1].imshow(ood_spatial, cmap='hot', interpolation='nearest')
        axes[1, 1].set_title('OOD Spatial Concentration')
        axes[1, 1].axis('off')
        plt.colorbar(im2, ax=axes[1, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved statistics to {save_path}")
        
        return fig, id_stats, ood_stats


def load_datasets(batch_size=16):
    """Load CIFAR-10 and SVHN datasets."""
    
    # CIFAR-10 transform
    cifar_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # SVHN transform
    svhn_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
    ])
    
    # Load datasets
    cifar_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=cifar_transform)
    svhn_test = datasets.SVHN(root='./data', split='test', download=True, transform=svhn_transform)
    
    # Create data loaders
    cifar_loader = DataLoader(cifar_test, batch_size=batch_size, shuffle=True)
    svhn_loader = DataLoader(svhn_test, batch_size=batch_size, shuffle=True)
    
    return cifar_loader, svhn_loader


def main():
    parser = argparse.ArgumentParser(description='Visualize Saliency Maps for ID vs OOD')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--model_type', type=str, choices=['vit', 'resnet'], required=True, help='Model type')
    parser.add_argument('--method', type=str, choices=['saliency', 'integrated_gradients'], 
                       default='saliency', help='Attribution method')
    parser.add_argument('--num_samples', type=int, default=8, help='Number of samples to visualize')
    parser.add_argument('--output_dir', type=str, default='saliency_visualizations', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.model_type == 'vit':
        model = create_small_vit_for_cifar10()
    else:
        # Import ResNet only when needed
        from models.resnet import ResNet18
        model = ResNet18(num_classes=10)
    
    # Load trained weights
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"✅ Loaded model from {args.model_path}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load datasets
    print("Loading datasets...")
    cifar_loader, svhn_loader = load_datasets(args.batch_size)
    
    # Get sample batches
    id_batch = next(iter(cifar_loader))
    ood_batch = next(iter(svhn_loader))
    
    # Initialize visualizer
    visualizer = SaliencyVisualizer(model, device)
    
    # Create visualizations
    print(f"Creating {args.method} visualizations...")
    
    # 1. Side-by-side comparison
    comparison_path = os.path.join(args.output_dir, f'{args.method}_comparison_{args.model_type}.png')
    fig1 = visualizer.visualize_comparison(
        id_batch, ood_batch, 
        num_samples=args.num_samples, 
        method=args.method,
        save_path=comparison_path
    )
    
    # 2. Statistical analysis
    stats_path = os.path.join(args.output_dir, f'{args.method}_statistics_{args.model_type}.png')
    fig2, id_stats, ood_stats = visualizer.create_saliency_statistics(
        id_batch, ood_batch,
        method=args.method,
        save_path=stats_path
    )
    
    # Print statistics
    print(f"\n📊 {args.method.title()} Statistics:")
    print("ID (CIFAR-10):")
    for key, value in id_stats.items():
        print(f"  {key}: {value:.4f}")
    
    print("OOD (SVHN):")
    for key, value in ood_stats.items():
        print(f"  {key}: {value:.4f}")
    
    # Calculate drift metrics
    mean_drift = abs(id_stats['mean'] - ood_stats['mean']) / id_stats['mean'] * 100
    std_drift = abs(id_stats['std'] - ood_stats['std']) / id_stats['std'] * 100
    
    print(f"\n🔄 Attribution Drift:")
    print(f"  Mean drift: {mean_drift:.2f}%")
    print(f"  Std drift: {std_drift:.2f}%")
    
    print(f"\n✅ Visualizations saved to {args.output_dir}/")
    print("Files created:")
    print(f"  - {comparison_path}")
    print(f"  - {stats_path}")


if __name__ == "__main__":
    main() 