# 🚀 Quick Start: OOD Experiments

## What We've Built

✅ **Complete OOD Experiment Framework** focusing on the most critical experiments:
- **Primary OOD**: SVHN dataset for clean distribution shift analysis
- **Attribution Methods**: 4 methods with sanity checks (Saliency, Grad-CAM, Integrated Gradients, Attention Rollout)
- **Corruption Analysis**: CIFAR-10-C with 3 key corruption types
- **Efficiency**: Streamlined experiments with `--quick_test` option

## 🎯 Current Status

- ✅ Environment setup complete
- ✅ Attribution methods working
- ✅ Dataset utilities implemented
- ✅ Experiment runner ready
- ✅ Results comparison tools ready

## 🏃‍♂️ Running Experiments Now

### 1. Quick Test (Recommended First)

```bash
# Test with ResNet (if available)
python run_ood_experiments.py --model_path best_resnet_model.pth --model_type resnet --quick_test --output_dir results/resnet_quick

# Test with ViT (if available)  
python run_ood_experiments.py --model_path best_vit_model.pth --model_type vit --quick_test --output_dir results/vit_quick
```

### 2. Full Experiments

```bash
# Full ResNet experiment
python run_ood_experiments.py --model_path best_resnet_model.pth --model_type resnet --output_dir results/resnet

# Full ViT experiment
python run_ood_experiments.py --model_path best_vit_model.pth --model_type vit --output_dir results/vit
```

### 3. Compare Results

```bash
python compare_ood_results.py --results_dir results/
```

## 📋 What Each Experiment Does

### Phase 1: Attribution Sanity Checks (B1-B4)
- Tests if attribution methods are working correctly
- Uses simple linear models for validation
- **Time**: ~1-2 minutes

### Phase 2: Performance & Calibration (A1, A3)
- Evaluates model accuracy on CIFAR-10 (in-distribution)
- Evaluates model accuracy on SVHN (out-of-distribution)
- Computes Expected Calibration Error (ECE)
- **Time**: ~2-3 minutes

### Phase 3: SVHN Attribution Drift (C2) - **Most Important**
- Computes attributions on both CIFAR-10 and SVHN
- Measures how attribution patterns change (IoU, Pearson correlation)
- **Key insight**: Do explanations remain consistent under distribution shift?
- **Time**: ~5-10 minutes

### Phase 4: Corruption Analysis (C1)
- Tests 3 corruption types (Gaussian noise, brightness, contrast)
- 3 severity levels each
- **Time**: ~5-15 minutes

## 📊 Expected Results

### Performance Drop
- ResNet: ~85% CIFAR-10 → ~40-60% SVHN
- ViT: ~80% CIFAR-10 → ~35-55% SVHN

### Attribution Drift
- **High IoU/Correlation**: Stable explanations
- **Low IoU/Correlation**: Unreliable explanations under shift

### Key Questions to Answer
1. Which model maintains more consistent explanations?
2. How much do attribution patterns change CIFAR-10 → SVHN?
3. Which attribution method is most reliable?

## ⚡ Time Estimates

| Experiment Type | Quick Test | Full Experiment |
|----------------|------------|-----------------|
| ResNet | 3-5 min | 15-25 min |
| ViT | 3-5 min | 15-25 min |
| Both + Comparison | 8-12 min | 35-55 min |

## 🔧 Troubleshooting

### If Model Files Missing
```bash
# Check what models are available
python setup_experiments.py

# Train models if needed (from previous work)
python main.py --model resnet --epochs 50
python main.py --model vit --epochs 50
```

### If Running Out of Memory
```bash
# Use smaller batch size
python run_ood_experiments.py --model_path <path> --model_type <type> --batch_size 16

# Or use CPU
python run_ood_experiments.py --model_path <path> --model_type <type> --device cpu
```

### If Attribution Methods Fail
- This is expected for some methods on simple models
- Real models should work better
- Results will still be generated with available methods

## 🎯 Most Important Insight

**Focus on the SVHN attribution drift analysis** - this tells you whether your explanations can be trusted when the data distribution changes, which is crucial for real-world deployment.

The framework automatically handles all the complexity and gives you clean, interpretable results! 