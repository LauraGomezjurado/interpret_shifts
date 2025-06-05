# Out-of-Distribution (OOD) Experiment Suite

This directory contains a comprehensive framework for evaluating model robustness and attribution method reliability under distribution shifts.

## 🎯 Overview

The experiment suite focuses on efficiently evaluating the **most critical** OOD scenarios:

- **Primary OOD Dataset**: SVHN (Street View House Numbers) as the main distribution shift
- **Corruption Analysis**: CIFAR-10-C with a subset of 3 key corruption types
- **Attribution Methods**: 4 methods with built-in sanity checks
- **Efficiency Focus**: Streamlined experiments that provide maximum insight with minimal compute

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install additional requirements
pip install -r requirements_ood.txt

# Setup experiments (downloads datasets, checks models)
python setup_experiments.py
```

### 2. Run Experiments

The setup script creates `experiment_commands.txt` with ready-to-run commands:

```bash
# Quick test (recommended first)
python run_ood_experiments.py --model_path best_resnet_model.pth --model_type resnet --quick_test

# Full experiment
python run_ood_experiments.py --model_path best_resnet_model.pth --model_type resnet --output_dir results/resnet
```

### 3. Compare Results

```bash
# After running experiments on both models
python compare_ood_results.py --results_dir results/
```

## 📋 Experiment Categories

### A. Core Performance Evaluation
- **A1**: Baseline accuracy on CIFAR-10 (ID) and SVHN (OOD)
- **A3**: Model calibration analysis (Expected Calibration Error)

### B. Attribution Method Validation
- **B1-B4**: Sanity checks for all attribution methods:
  - Gradient sign test for saliency maps
  - Localization check for Grad-CAM
  - Completeness axiom for Integrated Gradients
  - Attention consistency for ViT models

### C. Distribution Shift Analysis
- **C1**: CIFAR-10-C corruption analysis (3 corruption types × 3 severities)
- **C2**: SVHN attribution drift analysis (primary focus)

## 🧠 Attribution Methods

1. **Saliency Maps**: Basic gradient-based attributions
2. **Grad-CAM**: Class activation mapping
3. **Integrated Gradients**: Path-based attribution with baseline
4. **Attention Rollout**: ViT-specific attention analysis

Each method includes built-in sanity checks to ensure reliability.

## 📊 Key Metrics

### Performance Metrics
- **Accuracy**: Classification performance
- **Expected Calibration Error (ECE)**: Confidence calibration

### Attribution Drift Metrics
- **Intersection over Union (IoU)**: Spatial overlap of important regions
- **Pearson Correlation**: Linear relationship between attribution patterns

## 🏗️ Project Structure

```
├── run_ood_experiments.py      # Main experiment runner
├── setup_experiments.py        # Environment setup and validation
├── compare_ood_results.py      # Results comparison and visualization
├── utils/
│   ├── attribution_methods.py  # Attribution method implementations
│   └── datasets.py             # Dataset utilities (SVHN, CIFAR-10-C)
├── requirements_ood.txt        # Additional dependencies
└── README_OOD_EXPERIMENTS.md   # This file
```

## 🎛️ Configuration Options

### Experiment Runner Options
```bash
python run_ood_experiments.py \
    --model_path <path_to_model> \
    --model_type [resnet|vit] \
    --batch_size 32 \
    --device auto \
    --output_dir results/ \
    --quick_test  # For debugging/testing
```

### Setup Options
```bash
python setup_experiments.py \
    --data_dir ./data \
    --skip_download \  # Skip dataset download
    --test_only        # Only run method tests
```

## 📈 Understanding Results

### Result Files
- `ood_results_<model>_<timestamp>.json`: Complete experiment results
- `summary_<model>.csv`: Key metrics in tabular format
- Comparison plots in `comparison_plots/` directory

### Key Insights to Look For

1. **Accuracy Drop**: How much performance degrades from CIFAR-10 → SVHN
2. **Calibration Degradation**: ECE increase indicates overconfidence on OOD data
3. **Attribution Drift**: Low IoU/correlation suggests unreliable explanations
4. **Corruption Robustness**: Graceful degradation vs. cliff-like failure

### Expected Results

**ResNet vs ViT Comparison**:
- ResNet typically shows more graceful degradation
- ViT may be more sensitive to certain corruptions
- Attribution patterns differ significantly between architectures

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   ```bash
   # Use smaller batch size or CPU
   python run_ood_experiments.py --batch_size 16 --device cpu
   ```

2. **Missing Dependencies**:
   ```bash
   pip install captum opencv-python pandas scikit-learn
   ```

3. **Dataset Download Fails**:
   ```bash
   # Manual setup
   python setup_experiments.py --skip_download
   ```

4. **Attribution Methods Fail**:
   ```bash
   # Test individual methods
   python setup_experiments.py --test_only
   ```

### Performance Optimization

- Use `--quick_test` for initial validation (runs on subsets)
- Reduce batch size if memory limited
- Focus on most important experiments if time constrained

## 🔬 Extending the Framework

### Adding New Attribution Methods
1. Implement method class in `utils/attribution_methods.py`
2. Add sanity check method
3. Update `AttributionSuite` class

### Adding New OOD Datasets
1. Create dataset class in `utils/datasets.py`
2. Add to `DatasetManager`
3. Update experiment runner

### Adding New Metrics
1. Implement metric in `utils/attribution_methods.py`
2. Update `compute_metrics` method
3. Add to comparison scripts

## 📚 Key References

- **Attribution Methods**: [Captum Library](https://captum.ai/)
- **SVHN Dataset**: [Street View House Numbers](http://ufldl.stanford.edu/housenumbers/)
- **CIFAR-10-C**: [Benchmarking Neural Network Robustness](https://arxiv.org/abs/1903.12261)
- **Calibration**: [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599)

## 🎯 Efficiency Tips

1. **Start Small**: Always run `--quick_test` first
2. **Prioritize**: Focus on SVHN analysis over corruptions if time limited
3. **Parallel**: Run experiments on different models simultaneously
4. **Monitor**: Use the built-in progress indicators and error handling

## 📋 Experiment Checklist

- [ ] Environment setup completed (`setup_experiments.py`)
- [ ] Both models trained and available
- [ ] Quick test successful
- [ ] Full ResNet experiment completed
- [ ] Full ViT experiment completed
- [ ] Results comparison generated
- [ ] Key insights documented

This framework provides a solid foundation for understanding model robustness and attribution reliability under distribution shifts. Focus on the SVHN analysis for the most impactful insights! 