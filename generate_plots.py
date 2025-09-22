import numpy as np
from utils.plot_utils import plot_loss_accuracy

# Training data from your completed ResNet training
train_losses = [1.8422, 1.4645, 1.3236, 1.2417, 1.1768, 1.1078, 1.0339, 0.9797, 0.9292, 0.8949, 0.8585, 0.8280, 0.7987, 0.7741, 0.7495, 0.7263, 0.7127, 0.6899, 0.6744, 0.6611, 0.6474, 0.6300, 0.6227, 0.6083, 0.5975, 0.5898, 0.5795, 0.5719]

train_accuracies = [38.17, 56.52, 63.40, 67.28, 70.34, 73.52, 76.87, 79.13, 81.31, 82.89, 84.54, 85.90, 87.28, 88.22, 89.30, 90.41, 90.95, 91.89, 92.72, 93.15, 93.82, 94.49, 94.78, 95.51, 95.92, 96.32, 96.77, 97.06]

test_losses = [1.5553, 1.3630, 1.2789, 1.2557, 1.1610, 1.1455, 1.0643, 1.0190, 1.0095, 1.0042, 0.9806, 0.9713, 0.9796, 0.9639, 0.9714, 0.9855, 0.9734, 0.9619, 0.9779, 0.9905, 0.9813, 0.9995, 0.9874, 1.0110, 1.0165, 1.0088, 1.0249, 1.0179]

test_accuracies = [51.16, 60.74, 65.88, 66.88, 70.99, 72.23, 75.07, 77.06, 77.56, 78.29, 79.18, 79.69, 79.35, 80.68, 80.30, 80.03, 80.33, 81.21, 80.51, 80.51, 81.06, 80.91, 80.81, 80.44, 80.76, 81.31, 80.84, 81.04]

print("🎉 CONGRATULATIONS! Your ResNet Training Completed Successfully!")
print("=" * 60)
print(f"✅ Total epochs completed: {len(train_losses)}")
print(f"🚀 Final training accuracy: {train_accuracies[-1]:.2f}%")
print(f"🎯 Final validation accuracy: {test_accuracies[-1]:.2f}%")
print(f"🏆 Best validation accuracy: {max(test_accuracies):.2f}%")
print(f"📉 Final training loss: {train_losses[-1]:.4f}")
print(f"📈 Final validation loss: {test_losses[-1]:.4f}")
print("=" * 60)

# Generate the plots
plot_loss_accuracy(train_losses, test_losses, train_accuracies, test_accuracies)
print('✅ Training plots generated and saved!')

# Verify model file exists
import os
if os.path.exists('resnet_cifar10_best.pth'):
    size_mb = os.path.getsize('resnet_cifar10_best.pth') / (1024 * 1024)
    print(f"✅ Model saved successfully: resnet_cifar10_best.pth ({size_mb:.1f} MB)")
else:
    print("❌ Model file not found") 