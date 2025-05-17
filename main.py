import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from models.resnet import ResNet18
from models.vit import HFViTPretrained, create_big_vit_for_cifar10

from utils.utils import train, evaluate
from utils.plot_utils import plot_loss_accuracy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet',
                        choices=['resnet', 'vit-hf-pretrained', 'vit-hf-scratch'],
                        help='Which model to run? Options are: resnet, vit-hf-pretrained, vit-hf-scratch')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay (for AdamW/Adam, etc.)')
    parser.add_argument('--img_size', type=int, default=224, help='Image size to resize (32 for scratch, 224 for pretrained, etc.)')
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------------------------------------------
    # 1. Data Preparation 
    # ----------------------------------------------------------------
    # By default, we do a resize to 'img_size' x 'img_size' if needed.
    # If you plan to train from scratch on CIFAR-10 with 32x32, you can
    # set --img_size=32 for that run.
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        # If you want to replicate ImageNet stats (commonly used):
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # ----------------------------------------------------------------
    # 2. Initialize Model
    # ----------------------------------------------------------------
    if args.model == 'resnet':
        # A simple ResNet18 from scratch
        model = ResNet18(num_classes=10)

    elif args.model == 'vit-hf-pretrained':
        # Use a ViT that is pretrained on ImageNet, then fine-tune
        model = HFViTPretrained(
            pretrained_name="google/vit-base-patch16-224",
            num_labels=10
        )

    else:  # 'vit-hf-scratch'
        # Create a "big" ViT from scratch with custom config
        # If using 32x32 (CIFAR), set --img_size=32. 
        # Then patch_size might be 4 or 2, etc., for enough patches.
        model = create_big_vit_for_cifar10(
            image_size=args.img_size,   # 32 or 224, etc.
            patch_size=4,
            hidden_size=256,     # can tweak up/down for bigger or smaller ViT
            depth=12,
            num_heads=8,
            num_labels=10
        )

    model.to(device)

    # ----------------------------------------------------------------
    # 3. Define Loss and Optimizer
    # ----------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()

    # If you're training from scratch, AdamW is often used for ViT with weight decay
    # For simplicity, we keep the same optimizer for all models
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ----------------------------------------------------------------
    # 4. Training Loop
    # ----------------------------------------------------------------
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(val_loss)
        test_accuracies.append(val_acc)

        print(f'Epoch [{epoch}/{args.epochs}] '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

    # ----------------------------------------------------------------
    # 5. Plot Curves / Save Model
    # ----------------------------------------------------------------
    plot_loss_accuracy(train_losses, test_losses, train_accuracies, test_accuracies)

    # Optionally save model
    # e.g. torch.save(model.state_dict(), f"{args.model}_cifar10_epoch{args.epochs}.pth")


if __name__ == '__main__':
    main()
