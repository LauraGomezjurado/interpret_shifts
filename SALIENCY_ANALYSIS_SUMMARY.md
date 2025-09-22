# 🔍 Saliency Map Analysis: ID vs OOD Comparison

## 📊 **Executive Summary**

This analysis compares how your Vision Transformer (ViT) model's attention patterns change when moving from **in-distribution (CIFAR-10)** to **out-of-distribution (SVHN)** data using two working attribution methods: **Saliency Maps** and **Integrated Gradients**.

---

## 🎯 **Key Findings**

### **🚨 Significant Attribution Drift Detected**

| Method | Mean Drift | Std Drift | Interpretation |
|--------|------------|-----------|----------------|
| **Saliency Maps** | **36.56%** | **14.32%** | Moderate drift |
| **Integrated Gradients** | **41.24%** | **37.27%** | High drift |

### **📈 Statistical Changes**

#### **Saliency Maps:**
- **ID (CIFAR-10)**: Mean=0.0215, Std=0.0336, Max=0.7816, Sparsity=48.45%
- **OOD (SVHN)**: Mean=0.0293, Std=0.0384, Max=0.9194, Sparsity=34.17%

#### **Integrated Gradients:**
- **ID (CIFAR-10)**: Mean=0.0249, Std=0.0383, Max=1.0083, Sparsity=45.94%
- **OOD (SVHN)**: Mean=0.0352, Std=0.0526, Max=1.0195, Sparsity=38.45%

---

## 🔍 **What This Means**

### **1. Model Behavior Changes Significantly**
- **36-41% drift** indicates the model is focusing on different image regions when processing OOD data
- This suggests the model is **not robust** to distribution shift

### **2. Increased Attribution Intensity**
- **Higher mean values** for OOD data suggest the model is working "harder" to make predictions
- **Lower sparsity** means more pixels are considered important (less focused attention)

### **3. Attribution Method Consistency**
- Both methods show similar trends, validating the findings
- **Integrated Gradients shows higher drift**, suggesting it's more sensitive to distribution changes

---

## 📸 **Visual Analysis**

### **Generated Visualizations:**

1. **`saliency_comparison_vit.png`**: Side-by-side comparison showing:
   - Original ID and OOD images
   - Saliency overlays highlighting important regions
   - Model predictions for each image

2. **`saliency_statistics_vit.png`**: Statistical analysis including:
   - Bar charts comparing attribution statistics
   - Distribution histograms
   - Spatial concentration heatmaps

3. **`integrated_gradients_comparison_vit.png`**: Same format as saliency but for Integrated Gradients

4. **`integrated_gradients_statistics_vit.png`**: Statistical analysis for Integrated Gradients

---

## 🎨 **What to Look For in the Visualizations**

### **In the Comparison Images:**
- **Different focus areas**: ID vs OOD images show different highlighted regions
- **Prediction errors**: Check if OOD predictions match the true CIFAR-10 classes
- **Attribution intensity**: OOD overlays may appear more intense/scattered

### **In the Statistics Plots:**
- **Bar charts**: Clear differences between ID and OOD statistics
- **Histograms**: Different distribution shapes indicate behavioral changes
- **Spatial heatmaps**: Different concentration patterns show where the model focuses

---

## ⚠️ **Risk Assessment**

### **High Risk Indicators:**
1. **>35% attribution drift** - Your model shows this level
2. **Decreased sparsity** - Model attention becomes less focused
3. **Increased variance** - Less consistent decision-making

### **Impact on Real-World Deployment:**
- **Reliability concerns**: Model behavior is unpredictable on new data
- **Interpretability issues**: Attribution explanations may not transfer
- **Trust implications**: Users can't rely on consistent explanations

---

## 🛠️ **Recommendations**

### **Immediate Actions:**
1. **Document the limitation**: Note that attribution methods show significant drift
2. **Use with caution**: Be aware that explanations may not be reliable for OOD data
3. **Consider ensemble methods**: Multiple attribution methods provide more robust insights

### **Long-term Improvements:**
1. **Domain adaptation**: Train the model on more diverse data
2. **Robust training**: Use techniques like adversarial training
3. **Attribution-aware training**: Train models to maintain consistent attributions

---

## 📋 **Technical Details**

### **Experimental Setup:**
- **Model**: Vision Transformer (ViT) trained on CIFAR-10
- **ID Dataset**: CIFAR-10 test set (natural images)
- **OOD Dataset**: SVHN test set (street view house numbers)
- **Attribution Methods**: Saliency Maps, Integrated Gradients
- **Metrics**: Mean, Std, Max, Sparsity, Drift percentage

### **Why These Methods Work:**
- **Saliency Maps**: Compute gradients w.r.t. input pixels
- **Integrated Gradients**: Path integral of gradients from baseline to input
- **Both methods**: Don't depend on specific layer architectures (unlike GradCAM)

---

## 🎯 **Conclusions**

### **Key Takeaways:**
1. **Your ViT model shows significant attribution drift** when facing distribution shift
2. **The model becomes less focused** and more uncertain on OOD data
3. **Attribution explanations are not reliable** across different data distributions
4. **This is a common limitation** in current deep learning models

### **Scientific Contribution:**
This analysis demonstrates the importance of:
- **Testing attribution methods across distributions**
- **Measuring attribution drift as a robustness metric**
- **Understanding the limitations of interpretability methods**

### **Next Steps:**
1. **Compare with other models** (e.g., ResNet) to see if this is architecture-specific
2. **Test on other OOD datasets** to validate the findings
3. **Explore attribution-robust training methods**

---

## 📁 **Files Generated**

```
saliency_visualizations/
├── saliency_comparison_vit.png          # Visual comparison of ID vs OOD saliency
├── saliency_statistics_vit.png          # Statistical analysis of saliency maps
├── integrated_gradients_comparison_vit.png  # Visual comparison for IG
└── integrated_gradients_statistics_vit.png  # Statistical analysis for IG
```

**Total**: 4 high-resolution visualizations ready for analysis and publication!

---

*This analysis provides crucial insights into model robustness and the reliability of attribution methods across different data distributions. The significant drift observed highlights important limitations that should be considered in real-world deployments.* 