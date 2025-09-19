# Project Structure Overview

This document provides a comprehensive overview of the organized repository structure for the Distribution Shift Analysis project.

## 📁 Directory Structure

```
interpret_shifts/
├── 📄 README.md                           # Main project documentation
├── 📄 requirements.txt                   # Unified dependencies
├── 📄 PROJECT_STRUCTURE.md              # This file
├── 📁 src/                              # Core source code
│   ├── models/                          # Model architectures
│   │   ├── __init__.py
│   │   ├── resnet.py                    # ResNet-18 implementation
│   │   └── vit.py                       # Vision Transformer implementation
│   ├── utils/                           # Utility functions
│   │   ├── __init__.py
│   │   ├── attribution_methods.py      # Attribution computation methods
│   │   ├── datasets.py                  # Dataset loading utilities
│   │   ├── plot_utils.py               # Plotting utilities
│   │   └── utils.py                     # General utilities
│   └── attribution/                     # Attribution method implementations
├── 📁 experiments/                      # Experiment scripts
│   ├── training/                        # Model training scripts
│   │   ├── main.py                     # Main training script
│   │   ├── main_gpu.py                 # GPU-optimized training
│   │   └── resnet_run.sh               # ResNet training script
│   ├── ood_analysis/                    # Out-of-distribution analysis
│   │   ├── analyze_cifar100_resnet.py  # CIFAR-100 ResNet analysis
│   │   ├── analyze_svhn_resnet.py      # SVHN ResNet analysis
│   │   ├── compare_ood_results.py      # Results comparison
│   │   ├── run_ood_experiments.py      # Main OOD experiment runner
│   │   ├── run_ood_experiments_cifar100.py
│   │   ├── run_ood_experiments_memory_optimized.py
│   │   └── vit_attribution_fix.py      # ViT attribution fixes
│   ├── visualization/                   # Visualization scripts
│   │   ├── generate_plots.py           # Plot generation
│   │   ├── simple_visualize.py         # Simple visualization
│   │   ├── visualize_cifar100_results.py
│   │   ├── visualize_ood_results.py    # Main visualization script
│   │   └── visualize_saliency_maps_cifar100.py
│   ├── gpu_benchmark.py               # GPU performance benchmarking
│   └── setup_experiments.py           # Experiment setup utilities
├── 📁 results/                         # Experimental results
│   └── consolidated/                   # All experimental results
│       ├── results_cifar100_resnet_gpu/
│       │   ├── analysis/               # Analysis reports and dashboards
│       │   ├── cifar100_ood_results_resnet_*.json
│       │   ├── cifar100_summary_resnet.csv
│       │   └── visualizations/         # Generated visualizations
│       ├── results_cifar100_vit/
│       │   ├── cifar100_ood_results_vit_*.json
│       │   ├── cifar100_summary_vit.csv
│       │   └── visualizations/        # ViT-specific visualizations
│       ├── results_svhn_resnet_memory_opt/
│       │   ├── analysis/              # SVHN analysis reports
│       │   └── svhn_ood_results_resnet_*.json
│       ├── vit_quick/                 # Quick ViT experiments
│       ├── saliency_cifar100/         # CIFAR-100 saliency analysis
│       ├── saliency_visualizations/   # Saliency map visualizations
│       ├── Figure_1_ViT.png          # Key figures
│       ├── Figure_1.png
│       ├── ResNet_training.png       # Training curves
│       ├── ViT_training.png
│       ├── training_curves_epoch_*.png
│       ├── resnet_cifar10_best.pth   # Trained models
│       └── vit-hf-scratch-small_cifar10_best.pth
├── 📁 docs/                           # Documentation
│   ├── README.md                      # Documentation index
│   ├── QUICK_START_OOD.md            # Quick start guide
│   ├── README_OOD_EXPERIMENTS.md     # OOD experiment guide
│   ├── reports/                      # Research reports
│   │   ├── CIFAR100_ANALYSIS_SUMMARY.md
│   │   ├── CIFAR100_OOD_ANALYSIS_REPORT.md
│   │   ├── FINAL_EXPERIMENT_SUMMARY.md
│   │   ├── ResNet_vs_ViT_OOD_Analysis_Report.md
│   │   ├── SALIENCY_ANALYSIS_SUMMARY.md
│   │   ├── VISUALIZATION_SUMMARY.md
│   │   └── WHY_GRADCAM_ATTENTION_ZERO.md
│   └── visualizations/               # Documentation visualizations
├── 📁 examples/                       # Usage examples
│   ├── quick_start.py               # Basic usage example
│   ├── custom_analysis.py          # Advanced analysis example
│   └── commands.txt                 # Useful commands
└── 📁 data/                         # Datasets (not tracked in git)
    ├── cifar-10-batches-py/        # CIFAR-10 dataset
    ├── cifar-10-python.tar.gz
    ├── cifar-100-python/           # CIFAR-100 dataset
    ├── cifar-100-python.tar.gz
    └── test_32x32.mat             # SVHN dataset
```

## 🎯 Key Components

### Source Code (`src/`)
- **Models**: ResNet-18 and Vision Transformer implementations
- **Utils**: Core utilities for training, evaluation, and analysis
- **Attribution**: Attribution method implementations (Saliency, Grad-CAM, Integrated Gradients)

### Experiments (`experiments/`)
- **Training**: Model training scripts with advanced optimization
- **OOD Analysis**: Comprehensive out-of-distribution evaluation
- **Visualization**: Analysis and plotting utilities

### Results (`results/consolidated/`)
- **Analysis Reports**: Detailed findings and insights
- **Visualizations**: Generated plots and dashboards
- **Model Weights**: Trained model checkpoints
- **Data Files**: JSON results and CSV summaries

### Documentation (`docs/`)
- **Research Reports**: Comprehensive analysis reports
- **Quick Start Guides**: Getting started documentation
- **Visualizations**: Documentation-specific plots

### Examples (`examples/`)
- **Quick Start**: Basic usage demonstration
- **Custom Analysis**: Advanced analysis examples
- **Commands**: Useful command reference

## 🔧 Usage Patterns

### For Researchers
1. **Start with**: `docs/README.md` for overview
2. **Read reports**: `docs/reports/` for detailed findings
3. **Run experiments**: `experiments/` for custom analysis
4. **View results**: `results/consolidated/` for generated artifacts

### For Practitioners
1. **Quick start**: `examples/quick_start.py`
2. **Training**: `experiments/training/main.py`
3. **Analysis**: `experiments/ood_analysis/run_ood_experiments.py`
4. **Visualization**: `experiments/visualization/visualize_ood_results.py`

### For Developers
1. **Core code**: `src/` for implementation details
2. **Experiments**: `experiments/` for methodology
3. **Examples**: `examples/` for usage patterns
4. **Results**: `results/` for validation

## 📊 Key Findings Summary

### Critical Safety Issues
- **71.73% accuracy drop** on OOD data with maintained confidence
- **84.7% attribution dissimilarity** between ID and OOD
- **16.9× calibration degradation** (ECE: 0.035 → 0.598)

### Architecture Comparison
| Model | ID Accuracy | OOD Accuracy | Attribution IoU | Calibration ECE |
|-------|-------------|-----------------|-------------------|-------------------|
| **ViT** | 72.74% | 1.01% | 0.153 | 0.598 |
| **ResNet** | 77.02% | 0.90% | 0.123 | 0.662 |

### Attribution Method Robustness
- **Saliency Maps**: Most robust gradient-based method
- **Integrated Gradients**: Theoretically sound but drifts
- **Grad-CAM**: Incompatible with ViT architecture
- **Attention Rollout**: Requires architecture-specific fixes

## 🚀 Getting Started

### Installation
```bash
git clone https://github.com/yourusername/interpret_shifts.git
cd interpret_shifts
pip install -r requirements.txt
```

### Basic Usage
```bash
# Train a model
python experiments/training/main.py --model vit-hf-scratch --epochs 50

# Run OOD analysis
python experiments/ood_analysis/run_ood_experiments.py --model vit --ood_dataset cifar100

# Generate visualizations
python experiments/visualization/visualize_ood_results.py
```

### Quick Demo
```bash
# Run quick start example
python examples/quick_start.py --model vit --epochs 10

# Run custom analysis
python examples/custom_analysis.py --models vit resnet --ood_datasets cifar100 svhn
```

## 📈 Research Impact

This project provides:
- **Critical safety insights** for AI deployment
- **Comprehensive evaluation framework** for model robustness
- **Practical guidance** for production systems
- **Open-source tools** for continued research

## 🔮 Future Work

- Hybrid architectures combining CNN and Transformer strengths
- Attribution-aware training for consistent explanations
- Real-world deployment testing and validation
- Regulatory framework development for OOD safety

---

*This organized structure provides a clear, professional presentation of the research project while maintaining all original functionality and results.*
