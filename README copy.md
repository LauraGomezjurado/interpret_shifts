# ViT Training Optimization for CIFAR-10

This repository contains optimized training code for Vision Transformers (ViT) on CIFAR-10, with enhanced convergence strategies.

## Key Improvements for Better ViT Convergence

### 1. **Learning Rate Scheduling**
- **Cosine annealing with warmup**: Gradually increases LR for first few epochs, then decreases following cosine schedule
- **Warmup epochs**: Helps stabilize early training when gradients are large

### 2. **Gradient Clipping**
- Prevents gradient explosion common in transformer training
- Default value: 1.0 (can be adjusted via `--grad_clip`)

### 3. **Enhanced Data Augmentation**
- More aggressive augmentation for ViT models
- Includes: Random rotation, color jitter, horizontal flip
- Helps improve generalization and convergence

### 4. **Early Stopping**
- Monitors validation loss and stops training when no improvement
- Restores best model weights
- Prevents overfitting and saves training time

### 5. **Label Smoothing**
- Reduces overconfidence and improves generalization
- Default smoothing factor: 0.1

### 6. **Optimized Model Architecture**
- Better weight initialization (Xavier/Glorot)
- Enhanced dropout configuration
- QKV bias enabled for better attention learning

## Recommended Training Commands

### For ViT from Scratch (Recommended):
```bash
# High learning rate with larger batch size (if GPU memory allows)
python main.py --model vit-hf-scratch --epochs 100 --img_size 32 --lr 3e-4 --batch_size 128 --warmup_epochs 10 --patience 15 --grad_clip 1.0 --weight_decay 0.05

# Conservative approach with smaller learning rate
python main.py --model vit-hf-scratch --epochs 80 --img_size 32 --lr 1e-4 --batch_size 64 --warmup_epochs 8 --patience 12 --grad_clip 0.5 --weight_decay 0.03
```

### For Pretrained ViT Fine-tuning:
```bash
python main.py --model vit-hf-pretrained --epochs 30 --img_size 224 --lr 1e-5 --batch_size 32 --warmup_epochs 3 --patience 8 --grad_clip 1.0 --weight_decay 0.01
```

## Training Tips

### **Learning Rate Selection**
- **From scratch**: Start with 1e-4 to 3e-4
- **Fine-tuning**: Use 1e-5 to 5e-5 (much lower)
- Monitor the learning rate curve in output

### **Batch Size Impact**
- Larger batch sizes (128+) often help ViT convergence
- If memory limited, use gradient accumulation
- Adjust learning rate proportionally with batch size

### **Convergence Indicators**
- **Good convergence**: Validation loss steadily decreasing, train-val gap < 10%
- **Need more epochs**: Both losses still decreasing
- **Overfitting**: Training accuracy >> validation accuracy
- **Need LR adjustment**: Loss plateaus early or oscillates

### **Troubleshooting Poor Convergence**

1. **Loss explodes/NaN**:
   - Reduce learning rate by 2-5x
   - Increase gradient clipping (try 0.5)
   - Check for bad data samples

2. **Slow convergence**:
   - Increase learning rate
   - Reduce weight decay
   - Increase warmup epochs
   - Check data augmentation isn't too aggressive

3. **Overfitting**:
   - Increase weight decay
   - Add more dropout
   - Use more data augmentation
   - Reduce model size

4. **Underfitting**:
   - Increase model capacity
   - Reduce regularization
   - Train for more epochs
   - Increase learning rate

## Architecture Recommendations

### For CIFAR-10 (32x32):
- **Image size**: 32
- **Patch size**: 4 (gives 8x8 = 64 patches)
- **Hidden size**: 256-512
- **Layers**: 8-12
- **Attention heads**: 8

### Memory Optimization:
- Use `num_workers=2` and `pin_memory=True` for faster data loading
- Enable mixed precision training for larger models
- Consider gradient checkpointing for very deep models

## Expected Performance

With optimized training:
- **ViT from scratch**: 70-85% accuracy (depends on model size)
- **Pretrained ViT**: 85-95% accuracy
- **Training time**: 1-3 hours on modern GPU

## Monitoring Training

The enhanced plotting function provides:
- Real-time convergence analysis
- Moving averages for trend visualization
- Overfitting detection
- Final performance summary

Look for the convergence analysis output to determine if your model needs more training or different hyperparameters.
