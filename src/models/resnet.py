import torch.nn as nn
import torchvision.models as models

# Simple wrapper around torchvision's ResNet18
class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        # Load a standard ResNet18 (assumes ImageNet-size input)
        # For CIFAR-10, you might adapt the first conv layer or use a smaller variant
        self.resnet = models.resnet18(pretrained=False)
        
        # Change final FC layer to match CIFAR-10 classes
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)
