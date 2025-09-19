import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

from models.resnet import ResNet18
from models.vit import HFViTPretrained, create_big_vit_for_cifar10, create_small_vit_for_cifar10

from utils.utils import train, evaluate
from utils.plot_utils import plot_loss_accuracy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet',
                        choices=['resnet', 'vit-hf-pretrained', 'vit-hf-scratch', 'vit-hf-scratch-small'],
                        help='Which model to run? Options are: resnet, vit-hf-pretrained, vit-hf-scratch, vit-hf-scratch-small')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay (for AdamW/Adam, etc.)')
    parser.add_argument('--img_size', type=int, default=224, help='Image size to resize (32 for scratch, 224 for pretrained, etc.)')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Number of warmup epochs for learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value')
    return parser.parse_args()


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * float(num_cycles) * 2.0 * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ----------------------------------------------------------------
    # 1. Data Preparation 
    # ----------------------------------------------------------------
    # Enhanced data augmentation for better training
    if args.model.startswith('vit'):
        # More aggressive augmentation for ViT
        train_transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    test_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.CIFAR10(root='data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root='data', train=False, download=True, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # ----------------------------------------------------------------
    # 2. Initialize Model
    # ----------------------------------------------------------------
    if args.model == 'resnet':
        model = ResNet18(num_classes=10)
    elif args.model == 'vit-hf-pretrained':
        model = HFViTPretrained(
            pretrained_name="google/vit-base-patch16-224",
            num_labels=10
        )
    elif args.model == 'vit-hf-scratch-small':
        # Smaller ViT for CPU training and experimentation
        model = create_small_vit_for_cifar10(
            image_size=args.img_size,
            patch_size=4,
            hidden_size=128,
            depth=6,
            num_heads=4,
            num_labels=10
        )
    else:  # 'vit-hf-scratch'
        model = create_big_vit_for_cifar10(
            image_size=args.img_size,
            patch_size=4,
            hidden_size=256,
            depth=12,
            num_heads=8,
            num_labels=10
        )

    model.to(device)

    # ----------------------------------------------------------------
    # 3. Define Loss and Optimizer with Advanced Settings
    # ----------------------------------------------------------------
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for better generalization

    # Enhanced optimizer settings for ViT
    if args.model.startswith('vit'):
        optimizer = optim.AdamW(model.parameters(), 
                               lr=args.lr, 
                               weight_decay=args.weight_decay,
                               betas=(0.9, 0.999),
                               eps=1e-8)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Learning rate scheduling
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, min_delta=0.001)

    # ----------------------------------------------------------------
    # 4. Enhanced Training Loop
    # ----------------------------------------------------------------
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []

    print(f"Training {args.model} for {args.epochs} epochs...")
    print(f"Batch size: {args.batch_size}, Learning rate: {args.lr}")
    print(f"Warmup epochs: {args.warmup_epochs}, Gradient clipping: {args.grad_clip}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device, 
                                    scheduler=scheduler, grad_clip=args.grad_clip)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(val_loss)
        test_accuracies.append(val_acc)

        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch}/{args.epochs}] '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | '
              f'LR: {current_lr:.6f}')

        # Early stopping check
        if early_stopping(val_loss, model):
            print(f"Early stopping triggered at epoch {epoch}")
            break

    # ----------------------------------------------------------------
    # 5. Plot Curves / Save Model
    # ----------------------------------------------------------------
    plot_loss_accuracy(train_losses, test_losses, train_accuracies, test_accuracies)

    # Save the best model
    model_name = f"{args.model}_cifar10_best.pth"
    torch.save(model.state_dict(), model_name)
    print(f"Best model saved as {model_name}")


if __name__ == '__main__':
    main()
