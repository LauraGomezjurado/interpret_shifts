
# CIFAR-100 ResNet OOD Analysis Report

Generated: 2025-06-05 00:04:58

## Performance Summary
- **CIFAR-10 Accuracy**: 77.02%
- **CIFAR-100 Accuracy**: 0.90%
- **Accuracy Drop**: 76.12%

## Calibration Analysis
- **CIFAR-10 ECE**: 0.0645
- **CIFAR-100 ECE**: 0.6616
- **ECE Increase**: 0.5972

## Attribution Drift Metrics
- **Saliency IoU**: 0.123
- **Grad-CAM IoU**: 0.104
- **Integrated Gradients IoU**: 0.128

## Risk Assessment
⚠️ **CRITICAL**: Model shows catastrophic failure on OOD data
⚠️ **HIGH RISK**: Severe calibration degradation
⚠️ **UNSTABLE**: Low attribution similarity across datasets

## Files Generated
- analysis_dashboard.png: Comprehensive visual analysis
- saliency_maps_comparison.png: Actual saliency map comparisons
- analysis_report.md: This summary report
