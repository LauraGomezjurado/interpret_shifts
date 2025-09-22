import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ViTGradCAM:
    """GradCAM specifically designed for Vision Transformers."""
    
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks on the last transformer layer."""
        def backward_hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                self.gradients = grad_output[0]
        
        def forward_hook(module, input, output):
            self.activations = output
        
        # Navigate to the last transformer layer
        # For HuggingFace ViT: model.hf_model.vit.encoder.layer[-1]
        try:
            if hasattr(self.model, 'hf_model'):
                # Your ViTWrapper structure
                last_layer = self.model.hf_model.vit.encoder.layer[-1]
                last_layer.register_forward_hook(forward_hook)
                last_layer.register_backward_hook(backward_hook)
                print("✅ ViT GradCAM: Hooks registered on last transformer layer")
            else:
                print("❌ ViT GradCAM: Could not find transformer layers")
        except Exception as e:
            print(f"❌ ViT GradCAM: Hook registration failed: {e}")
    
    def generate(self, inputs, target_class=None):
        """Generate ViT-specific GradCAM."""
        batch_size = inputs.size(0)
        
        # Reset
        self.gradients = None
        self.activations = None
        
        # Forward pass
        outputs = self.model(inputs)
        if target_class is None:
            target_class = outputs.argmax(dim=1)
        
        # Backward pass
        self.model.zero_grad()
        scores = outputs.gather(1, target_class.unsqueeze(1)).squeeze()
        scores.sum().backward()
        
        if self.gradients is None or self.activations is None:
            print("⚠️  ViT GradCAM: No gradients captured, returning zeros")
            return torch.zeros(batch_size, 1, 8, 8), target_class
        
        # For ViT, activations shape: [batch, seq_len, hidden_dim]
        # seq_len = num_patches + 1 (CLS token)
        
        # Remove CLS token (first token)
        patch_activations = self.activations[:, 1:, :]  # [batch, num_patches, hidden_dim]
        patch_gradients = self.gradients[:, 1:, :]
        
        # Compute importance weights
        weights = patch_gradients.mean(dim=1, keepdim=True)  # [batch, 1, hidden_dim]
        
        # Weighted combination
        cam = (weights * patch_activations).sum(dim=-1)  # [batch, num_patches]
        cam = F.relu(cam)
        
        # Reshape to spatial grid
        patch_size = int(np.sqrt(cam.size(1)))  # Assuming square patches
        cam = cam.view(batch_size, 1, patch_size, patch_size)
        
        # Upsample to input size
        cam = F.interpolate(cam, size=inputs.shape[2:], mode='bilinear', align_corners=False)
        
        return cam, target_class


class ViTAttentionRollout:
    """Proper attention rollout for Vision Transformers."""
    
    def __init__(self, model):
        self.model = model
        self.attention_maps = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks to capture attention weights."""
        def attention_hook(module, input, output):
            # For HuggingFace ViT attention layers
            if hasattr(module, 'attention_probs'):
                self.attention_maps.append(module.attention_probs)
        
        # Register on all attention layers
        try:
            if hasattr(self.model, 'hf_model'):
                for layer in self.model.hf_model.vit.encoder.layer:
                    if hasattr(layer, 'attention'):
                        layer.attention.register_forward_hook(attention_hook)
                print("✅ ViT Attention Rollout: Hooks registered on attention layers")
            else:
                print("❌ ViT Attention Rollout: Could not find attention layers")
        except Exception as e:
            print(f"❌ ViT Attention Rollout: Hook registration failed: {e}")
    
    def generate(self, inputs):
        """Generate attention rollout."""
        self.attention_maps = []
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(inputs)
        
        if not self.attention_maps:
            print("⚠️  ViT Attention Rollout: No attention maps captured")
            return torch.zeros(inputs.size(0), 1, 8, 8), outputs.argmax(dim=1)
        
        # Rollout computation
        batch_size = inputs.size(0)
        num_heads = self.attention_maps[0].size(1)
        seq_len = self.attention_maps[0].size(2)
        
        # Average across heads and add residual connections
        rollout = torch.eye(seq_len).unsqueeze(0).repeat(batch_size, 1, 1)
        
        for attention in self.attention_maps:
            # Average across heads
            avg_attention = attention.mean(dim=1)  # [batch, seq_len, seq_len]
            
            # Add residual connection
            avg_attention = avg_attention + torch.eye(seq_len).unsqueeze(0)
            avg_attention = avg_attention / avg_attention.sum(dim=-1, keepdim=True)
            
            # Multiply with previous rollout
            rollout = torch.bmm(avg_attention, rollout)
        
        # Extract attention to patches (ignore CLS token)
        patch_attention = rollout[:, 0, 1:]  # CLS token attention to patches
        
        # Reshape to spatial grid
        patch_size = int(np.sqrt(patch_attention.size(1)))
        attention_map = patch_attention.view(batch_size, 1, patch_size, patch_size)
        
        # Upsample to input size
        attention_map = F.interpolate(attention_map, size=inputs.shape[2:], 
                                    mode='bilinear', align_corners=False)
        
        return attention_map, outputs.argmax(dim=1)


def test_vit_attributions():
    """Test the ViT-specific attribution methods."""
    from models.vit import create_small_vit_for_cifar10
    
    # Create model
    model = create_small_vit_for_cifar10()
    model.eval()
    
    # Test input
    test_input = torch.randn(2, 3, 32, 32)
    
    # Test ViT GradCAM
    print("Testing ViT GradCAM...")
    vit_gradcam = ViTGradCAM(model)
    cam, _ = vit_gradcam.generate(test_input)
    print(f"GradCAM output shape: {cam.shape}, sum: {cam.sum():.4f}")
    
    # Test ViT Attention Rollout
    print("\nTesting ViT Attention Rollout...")
    vit_attention = ViTAttentionRollout(model)
    attention, _ = vit_attention.generate(test_input)
    print(f"Attention output shape: {attention.shape}, sum: {attention.sum():.4f}")

if __name__ == "__main__":
    test_vit_attributions() 