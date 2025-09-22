
# SVHN ResNet OOD Analysis Report

Generated: 2025-06-05 00:33:42

## Performance Summary
- **CIFAR-10 Accuracy**: 81.25%
- **SVHN Accuracy**: 9.50%
- **Accuracy Drop**: 71.75%

## Calibration Analysis
- **CIFAR-10 ECE**: 0.6100
- **SVHN ECE**: 0.0942
- **ECE Change**: -0.5158

## Attribution Drift Metrics
- **Saliency IoU**: 0.116
- **Grad-CAM IoU**: 0.106
- **Saliency Pearson**: 0.021
- **Grad-CAM Pearson**: -0.044

## Memory Optimization Results
✅ **Success**: Avoided 18GB MPS memory limit
✅ **Completion**: Full experiment completed
⚠️ **Limitation**: Integrated Gradients skipped (too memory intensive)

## Risk Assessment
⚠️ **HIGH RISK**: Severe accuracy drop on OOD data (71.8%)
✅ **POSITIVE**: Calibration actually improved on SVHN (lower ECE)
⚠️ **UNSTABLE**: Low attribution similarity across datasets (IoU ~0.11)

## Key Findings
1. **Cross-domain shift**: CIFAR-10 → SVHN represents significant domain gap
2. **Calibration paradox**: Model is better calibrated on SVHN despite poor accuracy
3. **Attribution instability**: Low IoU suggests model focuses on different features

## Files Generated
- svhn_resnet_analysis_dashboard.png: Comprehensive visual analysis
- svhn_saliency_maps_comparison.png: Saliency map comparisons
- svhn_analysis_report.md: This summary report
