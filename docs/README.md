# Documentation Index

This directory contains comprehensive documentation for the Distribution Shift Analysis project.

## 📚 Research Reports

### Main Analysis Reports
- **[ResNet vs ViT OOD Analysis Report](reports/ResNet_vs_ViT_OOD_Analysis_Report.md)** - Comprehensive comparison of ResNet and ViT architectures under distribution shift
- **[CIFAR-100 OOD Analysis Report](reports/CIFAR100_OOD_ANALYSIS_REPORT.md)** - Detailed analysis of model behavior on CIFAR-100 out-of-distribution data
- **[Final Experiment Summary](reports/FINAL_EXPERIMENT_SUMMARY.md)** - Complete overview of all experiments and findings

### Specialized Analysis
- **[CIFAR-100 Analysis Summary](reports/CIFAR100_ANALYSIS_SUMMARY.md)** - Summary of CIFAR-100 specific findings
- **[Saliency Analysis Summary](reports/SALIENCY_ANALYSIS_SUMMARY.md)** - Deep dive into saliency map analysis
- **[Visualization Summary](reports/VISUALIZATION_SUMMARY.md)** - Overview of generated visualizations
- **[Why Grad-CAM Attention Zero](reports/WHY_GRADCAM_ATTENTION_ZERO.md)** - Technical explanation of Grad-CAM issues with ViT

## 🚀 Quick Start Guides

- **[Quick Start OOD](QUICK_START_OOD.md)** - Getting started with out-of-distribution experiments
- **[README OOD Experiments](README_OOD_EXPERIMENTS.md)** - Detailed guide for OOD experiment setup

## 📊 Visualizations

The `visualizations/` directory contains:
- Performance comparison charts
- Attribution drift analysis plots
- Calibration assessment diagrams
- Semantic analysis visualizations

## 🔬 Research Methodology

### Key Findings Summary

1. **Critical Safety Issues**
   - Models fail catastrophically on OOD data while maintaining high confidence
   - Attribution methods become unreliable when most needed
   - No built-in mechanisms for distribution shift detection

2. **Architecture-Specific Insights**
   - **ViT**: Better attribution stability, superior calibration behavior
   - **ResNet**: Computational efficiency, broad interpretability compatibility
   - **Both**: Catastrophic OOD performance, unreliable confidence estimates

3. **Attribution Method Robustness**
   - Saliency Maps: Most robust gradient-based method
   - Integrated Gradients: Theoretically sound but drifts under distribution shift
   - Grad-CAM: Incompatible with ViT architecture
   - Attention Rollout: Requires architecture-specific implementation

### Experimental Design

- **Datasets**: CIFAR-10 (ID) → CIFAR-100, SVHN (OOD)
- **Models**: Vision Transformer (ViT), ResNet-18
- **Metrics**: Accuracy, Calibration (ECE), Attribution Drift (IoU), Semantic Analysis
- **Methods**: Saliency Maps, Grad-CAM, Integrated Gradients, Attention Rollout

## 📈 Results Overview

### Performance Comparison (CIFAR-100 OOD)

| Model | ID Accuracy | OOD Accuracy | Attribution IoU | Calibration ECE |
|-------|-------------|-----------------|-------------------|-------------------|
| **ViT** | 72.74% | 1.01% | 0.153 | 0.598 |
| **ResNet** | 77.02% | 0.90% | 0.123 | 0.662 |

### Key Metrics
- **Accuracy Drop**: 71.73% (ViT), 76.12% (ResNet)
- **Attribution Dissimilarity**: 84.7% between ID and OOD
- **Calibration Degradation**: 16.9× increase in ECE

## 🛠️ Technical Implementation

### Code Organization
- **`src/`**: Core source code (models, utilities, attribution methods)
- **`experiments/`**: Experiment scripts (training, OOD analysis, visualization)
- **`results/`**: Experimental results and generated artifacts
- **`examples/`**: Usage examples and quick start scripts

### Key Features
- Modular design for easy extension
- Comprehensive evaluation framework
- Reproducible results with fixed seeds
- Memory-optimized for efficient processing

## 🎯 Applications

### Research Use Cases
- AI safety research and model reliability assessment
- Interpretability method robustness evaluation
- Distribution shift detection and mitigation
- Model evaluation beyond accuracy metrics

### Industry Applications
- Medical AI: Reliable diagnosis across diverse populations
- Autonomous Systems: Safe deployment in novel environments
- Financial Systems: Robust risk assessment under changing conditions
- Quality Assurance: Model reliability testing for production

## 🔮 Future Work

### Immediate Next Steps
- Complete ResNet analysis on additional OOD datasets
- Implement proposed OOD detection methods
- Test calibration improvement techniques
- Extend to vision-language models

### Long-term Research
- Hybrid architectures combining CNN and Transformer strengths
- Attribution-aware training for consistent explanations
- Real-world deployment testing
- Regulatory framework development for OOD safety

## 📞 Contact

For questions about the research or collaboration opportunities:
- **Email**: laura.gomez@example.com
- **GitHub**: [@lauragomez](https://github.com/lauragomez)

---

*This documentation represents a comprehensive analysis of distribution shift effects in deep learning models, providing both critical safety insights and practical guidance for model deployment.*
