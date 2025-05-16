import torch.nn as nn
from transformers import ViTForImageClassification, ViTConfig

class HFViTPretrained(nn.Module):
    """
    Fine-tuning from a pretrained Hugging Face ViT (e.g., on ImageNet).
    """
    def __init__(self, pretrained_name="google/vit-base-patch16-224", num_labels=10):
        super().__init__()
        self.model = ViTForImageClassification.from_pretrained(
            pretrained_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )

    def forward(self, x):
        outputs = self.model(pixel_values=x)
        return outputs.logits

def create_big_vit_for_cifar10(image_size=32, patch_size=4, 
                               hidden_size=256, depth=12, num_heads=8,
                               num_labels=10):
    """
    Create a ViT from scratch using a custom ViTConfig. 
    No pretrained weights—pure random init.
    """
    config = ViTConfig(
        image_size=image_size,
        patch_size=patch_size,
        num_labels=num_labels,
        hidden_size=hidden_size,
        num_hidden_layers=depth,
        num_attention_heads=num_heads,
        intermediate_size=hidden_size * 4,  # typical transformer MLP size
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1
    )
    model = ViTForImageClassification(config)
    return model
