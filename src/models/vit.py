import torch.nn as nn
from transformers import ViTForImageClassification, ViTConfig
import torch



class HFViTPretrained(nn.Module):
    """
    Enhanced wrapper for pretrained ViT with better fine-tuning setup.
    """
    def __init__(self, pretrained_name="google/vit-base-patch16-224", num_labels=10):
        super().__init__()
        self.model = ViTForImageClassification.from_pretrained(
            pretrained_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
        
        # Add dropout to the classifier for better regularization
        if hasattr(self.model, 'classifier'):
            hidden_size = self.model.classifier.in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(hidden_size, num_labels)
            )

    def forward(self, x):
        outputs = self.model(pixel_values=x)
        return outputs.logits

# class HFViTPretrained(nn.Module):
#     """
#     Fine-tuning from a pretrained Hugging Face ViT (e.g., on ImageNet).
#     """
#     def __init__(self, pretrained_name="google/vit-base-patch16-224", num_labels=10):
#         super().__init__()
#         self.model = ViTForImageClassification.from_pretrained(
#             pretrained_name,
#             num_labels=num_labels,
#             ignore_mismatched_sizes=True
#         )

#     def forward(self, x):
#         outputs = self.model(pixel_values=x)
#         return outputs.logits



def create_small_vit_for_cifar10(image_size=32, patch_size=4, hidden_size=128, 
                                depth=6, num_heads=4, num_labels=10):
    """
    Smaller ViT configuration optimized for CPU training and faster experimentation.
    """
    config = ViTConfig(
        image_size=image_size,
        patch_size=patch_size,
        num_labels=num_labels,
        hidden_size=hidden_size,
        num_hidden_layers=depth,
        num_attention_heads=num_heads,
        intermediate_size=hidden_size * 2,  # Smaller FFN
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        layer_norm_eps=1e-6,
        initializer_range=0.02,
        hidden_act="gelu",
        qkv_bias=True
    )
    vit_model = ViTForImageClassification(config)

    class ViTWrapper(nn.Module):
        def __init__(self, hf_model):
            super().__init__()
            self.hf_model = hf_model
            
            # Apply better weight initialization
            self._init_weights()
        
        def _init_weights(self):
            """Apply better weight initialization for training from scratch."""
            for module in self.hf_model.modules():
                if isinstance(module, nn.Linear):
                    # Xavier/Glorot initialization for linear layers
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, std=0.02)
        
        def forward(self, x):
            output = self.hf_model(pixel_values=x)
            return output.logits

    return ViTWrapper(vit_model)

def create_big_vit_for_cifar10(image_size=32, patch_size=4, hidden_size=256, 
                               depth=12, num_heads=8, num_labels=10):
    """
    Enhanced ViT configuration optimized for CIFAR-10 training from scratch.
    """
    config = ViTConfig(
        image_size=image_size,
        patch_size=patch_size,
        num_labels=num_labels,
        hidden_size=hidden_size,
        num_hidden_layers=depth,
        num_attention_heads=num_heads,
        intermediate_size=hidden_size * 4,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        layer_norm_eps=1e-6,
        initializer_range=0.02,
        # Enhanced configuration for better training
        hidden_act="gelu",
        qkv_bias=True
    )
    vit_model = ViTForImageClassification(config)

    class ViTWrapper(nn.Module):
        def __init__(self, hf_model):
            super().__init__()
            self.hf_model = hf_model
            
            # Apply better weight initialization
            self._init_weights()
        
        def _init_weights(self):
            """Apply better weight initialization for training from scratch."""
            for module in self.hf_model.modules():
                if isinstance(module, nn.Linear):
                    # Xavier/Glorot initialization for linear layers
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, std=0.02)
        
        def forward(self, x):
            # Hugging Face expects arg 'pixel_values' for images
            output = self.hf_model(pixel_values=x)
            # Return just the raw tensor of shape [batch_size, num_labels]
            return output.logits

    return ViTWrapper(vit_model)
