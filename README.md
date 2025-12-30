# Interpreting Distribution Shifts in Deep Learning Models

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive research project investigating how deep learning models behave under distribution shifts, with focus on attribution method robustness and model interpretability across different data distributions.

## 🎯 Project Overview

This project addresses a critical challenge in AI safety: **What happens when deep learning models encounter data that differs from their training distribution?** Through extensive experiments with Vision Transformers (ViT) and ResNet architectures, we demonstrate that:

- 🚨 **Models fail catastrophically** on out-of-distribution (OOD) data while maintaining high confidence
- 🔍 **Attribution methods become unreliable** precisely when interpretability is most needed  
- ⚖️ **Calibration breaks down** creating dangerous "confident but wrong" scenarios
- 🏗️ **Architecture matters** for both performance and interpretability under distribution shift

## 🔬 Key Research Contributions

### 1. **Comprehensive OOD Analysis Framework**
- Multi-dataset evaluation (CIFAR-10 → CIFAR-100, SVHN)
- Performance, calibration, and attribution drift metrics
- Semantic coherence analysis across distribution shifts

### 2. **Attribution Method Robustness Study**
- Saliency Maps, Grad-CAM, Integrated Gradients, Attention Rollout
- Architecture-specific compatibility analysis
- Correlation and similarity metrics for attribution drift detection

### 3. **Critical Safety Findings**
- **71.73% accuracy drop** on OOD data with maintained 61.9% confidence
- **84.7% attribution dissimilarity** between ID and OOD explanations
- **16.9× calibration degradation** (ECE: 0.035 → 0.598)

## 📊 Experimental Results

### Performance Comparison (CIFAR-100 OOD)

| Model | ID Accuracy | OOD Accuracy | Attribution IoU | Calibration ECE |
|-------|-------------|-----------------|-------------------|-------------------|
| **ViT** | 72.74% | 1.01% | 0.153 | 0.598 |
| **ResNet** | 77.02% | 0.90% | 0.123 | 0.662 |

### Key Visualizations

#### Performance Dashboard
![Performance Dashboard](results/consolidated/results_cifar100_vit/visualizations/performance_comparison.png)
*Comprehensive performance analysis showing catastrophic failure on OOD data*

#### Attribution Drift Analysis  
![Attribution Drift](results/consolidated/results_cifar100_vit/visualizations/attribution_drift.png)
*Attribution method reliability breakdown across distribution shifts*

#### Saliency Maps Comparison
![Saliency Comparison](results/consolidated/saliency_cifar100/saliency_comparison_cifar100_vit.png)
*Visual comparison of attribution methods on ID vs OOD data*

### Key Insights
- **Neither architecture is OOD-safe** without additional safeguards
- **ViT shows better attribution stability** but worse overall performance
- **ResNet offers computational efficiency** but limited attribution consistency
- **Attribution drift serves as OOD indicator** (IoU < 0.2 signals distribution shift)

## 🚀 Quick Start

### 🎯 One-Click Demo
**Try our interactive Colab notebook to see the critical safety findings in action:**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lauragomez/interpret_shifts/blob/main/quick_demo.ipynb)

*This notebook demonstrates the key findings using cached experimental results - no training required!*

### Installation
```bash
git clone https://github.com/lauragomez/interpret_shifts.git
cd interpret_shifts
pip install -r requirements.txt
```

### Basic Usage

#### 1. Train a Model
```bash
# Train ViT from scratch
python main.py --model vit-hf-scratch --epochs 100 --lr 3e-4 --batch_size 128

# Train ResNet
python main.py --model resnet --epochs 50 --lr 1e-3 --batch_size 64
```

#### 2. Run OOD Analysis
```bash
# Analyze ViT on CIFAR-100
python scripts/run_ood_experiments.py --model vit --ood_dataset cifar100

# Analyze ResNet on SVHN  
python scripts/run_ood_experiments.py --model resnet --ood_dataset svhn
```

#### 3. Generate Visualizations
```bash
# Create comprehensive analysis dashboard
python scripts/visualize_ood_results.py --results_dir results/
```

## 📁 Project Structure

```
interpret_shifts/
├── 📄 README.md                    # This file
├── 📄 requirements.txt            # Dependencies
├── 📄 quick_demo.ipynb           # One-click Colab demo
├── 📁 src/                        # Core source code
│   ├── models/                    # Model architectures
│   ├── utils/                     # Utility functions
│   └── attribution/               # Attribution methods
├── 📁 scripts/                    # Analysis and experiment scripts
│   ├── run_ood_experiments*.py   # OOD analysis scripts
│   ├── visualize_*.py             # Visualization scripts
│   └── resnet_run.sh             # Training scripts
├── 📁 experiments/                # Experiment configurations
│   ├── training/                  # Model training
│   ├── ood_analysis/              # OOD experiments
│   └── visualization/             # Analysis scripts
├── 📁 results/                    # Experimental results
│   ├── consolidated/             # All results consolidated
│   ├── results_cifar100_vit/     # ViT CIFAR-100 results
│   └── results_cifar100_resnet/  # ResNet CIFAR-100 results
├── 📁 analysis/                   # Analysis reports and summaries
│   ├── ResNet_vs_ViT_OOD_Analysis_Report.md
│   └── [other analysis reports]
├── 📁 docs/                      # Documentation
│   ├── reports/                  # Detailed analysis reports
│   └── visualizations/           # Generated plots
├── 📁 models/                    # Trained model checkpoints
└── 📁 examples/                  # Usage examples
    ├── quick_start.py           # Basic usage
    └── custom_analysis.py         # Custom experiments
```

## 🔍 Research Methodology

### Datasets
- **In-Distribution**: CIFAR-10 (10 classes, 32×32 images)
- **Out-of-Distribution**: 
  - CIFAR-100 (100 classes, similar domain)
  - SVHN (different domain, overlapping classes)

**Dataset Selection Rationale**: We deliberately chose CIFAR-10/100 and SVHN as our evaluation benchmarks for several critical reasons:
- **Controlled Environment**: These datasets provide a controlled laboratory setting for mechanistic analysis of distribution shift effects
- **Computational Efficiency**: Enables extensive experimentation with multiple models, attribution methods, and hyperparameter sweeps
- **Established Baselines**: Well-studied datasets with known failure modes, allowing for systematic comparison with existing literature
- **Reproducibility**: Standardized datasets ensure experimental reproducibility and facilitate comparison across research groups
- **Mechanistic Insights**: The relatively simple visual features in these datasets make it easier to understand and interpret model behavior under distribution shift

### Models
- **Vision Transformer (ViT)**: Transformer-based architecture
- **ResNet-18**: Convolutional neural network with residual connections

### Attribution Methods
- **Saliency Maps**: Gradient-based pixel importance
- **Grad-CAM**: Class activation mapping (CNN-specific)
- **Integrated Gradients**: Path-integrated attributions
- **Attention Rollout**: Transformer attention visualization

### Evaluation Metrics
- **Performance**: Accuracy, F1-score
- **Calibration**: Expected Calibration Error (ECE)
- **Attribution Drift**: IoU similarity, Pearson/Spearman correlation
- **Semantic Analysis**: Prediction distribution, confidence analysis

## 📈 Key Findings

### 🚨 Critical Safety Issues
1. **Silent Failure Mode**: Models fail catastrophically while appearing confident
2. **Explanation Unreliability**: Attribution methods become meaningless on OOD data
3. **No Built-in Detection**: Models lack mechanisms to identify distribution shift
4. **Systematic Biases**: Predictable failure patterns that could be exploited

### 🏆 Architecture-Specific Insights
- **ViT Strengths**: Better attribution stability, superior calibration behavior
- **ResNet Strengths**: Computational efficiency, broad interpretability compatibility
- **Both Limitations**: Catastrophic OOD performance, unreliable confidence estimates

## 🛠️ Technical Implementation

### Core Features
- **Modular Design**: Easy to extend with new models and attribution methods
- **Comprehensive Evaluation**: Multi-faceted analysis combining performance, calibration, and interpretability
- **Reproducible Results**: Fixed random seeds and complete parameter specifications
- **Memory Optimization**: Efficient data loading and GPU utilization

### Advanced Capabilities
- **Cross-dataset validation**: Multiple OOD scenarios
- **Corruption robustness**: Performance under various corruption levels
- **Semantic coherence**: Analysis of prediction patterns
- **Real-time monitoring**: Training progress and convergence analysis

## 📊 Generated Artifacts

### Analysis Reports
- **Comprehensive OOD Analysis**: Detailed performance and calibration breakdown
- **Attribution Drift Study**: Method-specific robustness analysis
- **Architecture Comparison**: ResNet vs ViT comprehensive evaluation
- **Safety Assessment**: Risk analysis for production deployment

### Visualizations
- **Performance Dashboards**: Multi-metric comparison charts
- **Attribution Drift Plots**: IoU and correlation analysis
- **Semantic Analysis**: Prediction distribution and confidence patterns
- **Calibration Assessment**: Reliability diagrams and ECE analysis

## 🎯 Applications

### Research Use Cases
- **AI Safety Research**: Understanding model limitations under distribution shift
- **Interpretability Studies**: Evaluating explanation method robustness
- **Model Evaluation**: Comprehensive assessment beyond accuracy metrics
- **OOD Detection**: Developing methods for distribution shift identification

### Industry Applications
- **Medical AI**: Ensuring reliable diagnosis across diverse patient populations
- **Autonomous Systems**: Safe deployment in novel environments
- **Financial Systems**: Robust risk assessment under changing market conditions
- **Quality Assurance**: Model reliability testing for production deployment

## 🔮 Future Work

### Immediate Next Steps
- [ ] Complete ResNet analysis on additional OOD datasets
- [ ] Implement proposed OOD detection methods
- [ ] Test calibration improvement techniques
- [ ] Extend to vision-language models

### Long-term Research
- [ ] Hybrid architectures combining CNN and Transformer strengths
- [ ] Attribution-aware training for consistent explanations
- [ ] Real-world deployment testing
- [ ] Regulatory framework development for OOD safety

## 📚 References

### Key Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer architecture
- [ResNet](https://arxiv.org/abs/1512.03385) - Residual networks
- [Grad-CAM](https://arxiv.org/abs/1610.02391) - Class activation mapping
- [Integrated Gradients](https://arxiv.org/abs/1703.01365) - Attribution method

### Related Work
- Distribution shift detection methods
- Model calibration techniques
- Attribution method robustness
- AI safety and reliability

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Areas for Contribution
- New attribution methods
- Additional OOD datasets
- Improved visualization tools
- Documentation improvements
- Performance optimizations

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

**Laura Gomez** - *Initial work and comprehensive analysis*
- GitHub: [@lauragomez](https://github.com/lauragomez)
- Email: laura.gomez@stanford.edu

## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- Hugging Face for transformer model implementations
- Captum team for attribution method implementations
- CIFAR and SVHN dataset creators for providing evaluation benchmarks

## 📞 Contact

For questions, suggestions, or collaboration opportunities:
- **Email**: laura.gomez@stanford.edu
- **GitHub Issues**: [Create an issue](https://github.com/lauragomez/interpret_shifts/issues)
- **Discussions**: [Join the conversation](https://github.com/lauragomez/interpret_shifts/discussions)

---

**⚠️ Important Notice**: This research demonstrates critical safety limitations in current deep learning models. Models should never be deployed in production without proper OOD detection and uncertainty quantification mechanisms.

**🔬 Research Impact**: This work provides both a sobering assessment of current AI limitations and a roadmap for building more reliable, interpretable, and safe AI systems.