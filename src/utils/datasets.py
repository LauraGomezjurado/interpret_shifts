import torch
import torchvision
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader, Dataset


class SVHNDataset:
    """SVHN dataset for OOD evaluation (far-OOD from CIFAR-10)."""
    
    def __init__(self, root='data', img_size=32, batch_size=64):
        self.root = root
        self.img_size = img_size
        self.batch_size = batch_size
        
        # SVHN preprocessing to match CIFAR-10 training
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
    def get_dataloader(self, split='test'):
        """Get SVHN dataloader."""
        dataset = datasets.SVHN(
            root=self.root,
            split=split,
            download=True,
            transform=self.transform
        )
        
        # Only use first 10 classes to match CIFAR-10
        # SVHN has digits 0-9, CIFAR-10 has 10 classes
        filtered_dataset = self._filter_classes(dataset, num_classes=10)
        
        return DataLoader(
            filtered_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )
    
    def _filter_classes(self, dataset, num_classes=10):
        """Filter dataset to only include specified number of classes."""
        # For SVHN, we already have 10 classes (digits 0-9)
        return dataset


class CIFAR10CDataset:
    """CIFAR-10-C corruption dataset for robustness evaluation."""
    
    def __init__(self, root='data', corruption_type='gaussian_noise', severity=1, img_size=32, batch_size=64):
        self.root = root
        self.corruption_type = corruption_type
        self.severity = severity
        self.img_size = img_size
        self.batch_size = batch_size
        
        # Standard CIFAR-10 preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
    def get_dataloader(self):
        """Get CIFAR-10-C dataloader for specific corruption."""
        # This is a simplified implementation
        # In practice, you'd load the actual CIFAR-10-C dataset
        
        # For now, create synthetic corruptions
        # Use only ToTensor transform for the base dataset
        cifar10_test = datasets.CIFAR10(
            root=self.root,
            train=False,
            download=True,
            transform=transforms.ToTensor()  # Only convert to tensor initially
        )
        
        corrupted_dataset = self._apply_corruption(cifar10_test)
        
        return DataLoader(
            corrupted_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )
    
    def _apply_corruption(self, dataset):
        """Apply corruption to dataset."""
        # Simplified corruption implementation
        class CorruptedDataset(Dataset):
            def __init__(self, original_dataset, corruption_type, severity, final_transform):
                self.dataset = original_dataset
                self.corruption_type = corruption_type
                self.severity = severity
                # Create the final transform (resize + normalize)
                self.final_transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                ])
                
            def __len__(self):
                return len(self.dataset)
                
            def __getitem__(self, idx):
                image, label = self.dataset[idx]  # image is already a tensor from ToTensor()
                
                # Apply corruption to tensor
                if self.corruption_type == 'gaussian_noise':
                    noise = torch.randn_like(image) * (self.severity * 0.1)
                    image = torch.clamp(image + noise, 0, 1)
                elif self.corruption_type == 'brightness':
                    image = torch.clamp(image + self.severity * 0.2, 0, 1)
                elif self.corruption_type == 'contrast':
                    image = torch.clamp(image * (1 + self.severity * 0.3), 0, 1)
                
                # Apply final normalization
                image = self.final_transform(image)
                    
                return image, label
        
        return CorruptedDataset(dataset, self.corruption_type, self.severity, self.transform)


class DatasetManager:
    """Manager for all datasets used in OOD experiments."""
    
    def __init__(self, img_size=32, batch_size=64):
        self.img_size = img_size
        self.batch_size = batch_size
        
    def get_cifar10_loaders(self):
        """Get CIFAR-10 train/test loaders."""
        transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        train_dataset = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='data', train=False, download=True, transform=transform)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)
        
        return train_loader, test_loader
    
    def get_svhn_loader(self):
        """Get SVHN OOD loader."""
        svhn_dataset = SVHNDataset(img_size=self.img_size, batch_size=self.batch_size)
        return svhn_dataset.get_dataloader()
    
    def get_cifar100_loader(self):
        """Get CIFAR-100 OOD loader."""
        transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Use CIFAR-100 test set as OOD data
        cifar100_dataset = datasets.CIFAR100(root='data', train=False, download=True, transform=transform)
        
        return DataLoader(
            cifar100_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )
    
    def get_corruption_loaders(self, corruption_types=['gaussian_noise', 'brightness', 'contrast'], severities=[1, 3, 5]):
        """Get multiple corruption loaders for evaluation."""
        loaders = {}
        
        for corruption in corruption_types:
            loaders[corruption] = {}
            for severity in severities:
                cifar_c = CIFAR10CDataset(
                    corruption_type=corruption,
                    severity=severity,
                    img_size=self.img_size,
                    batch_size=self.batch_size
                )
                loaders[corruption][severity] = cifar_c.get_dataloader()
                
        return loaders


# Calibration utilities
def expected_calibration_error(predictions, labels, n_bins=10):
    """Compute Expected Calibration Error (ECE)."""
    confidences = torch.max(torch.softmax(predictions, dim=1), dim=1)[0]
    accuracies = (predictions.argmax(dim=1) == labels).float()
    
    ece = 0.0
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece.item()


def evaluate_model_on_dataset(model, dataloader, device='cpu'):
    """Evaluate model on a dataset and return predictions, labels, accuracy, ECE."""
    model.eval()
    all_predictions = []
    all_labels = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            all_predictions.append(outputs.cpu())
            all_labels.append(labels.cpu())
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    predictions = torch.cat(all_predictions, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    accuracy = 100.0 * correct / total
    ece = expected_calibration_error(predictions, labels)
    
    return {
        'predictions': predictions,
        'labels': labels,
        'accuracy': accuracy,
        'ece': ece,
        'total_samples': total
    } 