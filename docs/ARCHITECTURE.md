# Project Architecture

## Overview

The Chest X-Ray Pneumonia Classifier is built using a transfer learning approach with ResNet50 and a production-ready web interface powered by Gradio.

## Directory Structure

```
.
├── src/
│   ├── app.py              # Gradio web application
│   └── model.py            # Model definitions and utilities
├── notebooks/
│   └── d_classifier.ipynb  # Full training pipeline and exploration
├── tests/
│   └── test_model.py       # Model evaluation tests
├── data/
│   └── .gitkeep            # Placeholder for data
├── models/
│   └── .gitkeep            # Store model weights
├── docs/
│   ├── ARCHITECTURE.md     # This file
│   ├── TRAINING.md         # Training details
│   └── API.md              # API documentation
├── requirements.txt
├── README.md
├── LICENSE
└── CONTRIBUTING.md
```

## Model Architecture

### Backbone
- **ResNet50** pre-trained on ImageNet
- Input: 224×224 RGB images
- Output: 2048-dimensional feature vector

### Feature Extraction Strategy
- Layers frozen: `conv1`, `layer1`, `layer2`, `layer3`
- Fine-tuned layer: `layer4` (last residual block)
- Rationale: Preserve general visual features, adapt to medical imaging

### Classification Head
```
2048 features
    ↓
Linear(2048 → 1024) + BatchNorm + ReLU + Dropout(0.4)
    ↓
Linear(1024 → 512) + BatchNorm + ReLU + Dropout(0.4)
    ↓
Linear(512 → 1)
    ↓
Sigmoid → [0, 1] prediction
```

### Why This Design?
- **Batch Normalization**: Stabilizes training, reduces internal covariate shift
- **Dropout**: Prevents overfitting in the head
- **Sigmoid**: Binary classification output (pneumonia vs normal)
- **3-layer head**: Better feature discrimination than 2-layer

## Data Pipeline

### Preprocessing
All images resized to 224×224 (ImageNet standard)

### Training Transforms
- Random rotation: ±15°
- Random affine: ±10% translation
- Random crop: 224×224 with 8px padding
- Random horizontal flip: 50%
- Color jitter: brightness & contrast ±0.2
- Normalize: ImageNet statistics

### Validation/Test Transforms
- Resize: 224×224
- Normalize: ImageNet statistics (no augmentation)

### Normalization Constants
```python
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
```

## Training Pipeline

### Optimizer
- Adam with differential learning rates
- Layer4: 1e-4 (fine-tuning)
- Head: 1e-3 (main learning)
- Weight decay: 1e-4

### Loss Function
- Binary Cross-Entropy with Logits

### Learning Rate Schedule
- Cosine Annealing
- T_max: 15 epochs
- eta_min: 1e-5

### Training Hyperparameters
- Batch size: 32
- Epochs: 15
- Early stopping: patience=3
- Gradient clipping: max_norm=1.0

## Model Inference

### Prediction Pipeline
1. Load image (JPEG)
2. Resize to 224×224
3. Convert to tensor
4. Apply ImageNet normalization
5. Pass through model
6. Apply sigmoid → probability
7. Threshold at 0.5 for classification

### Output Format
- Label: "PNEUMONIA" or "NORMAL"
- Confidence: [0, 1] probability

## Explainability

### Grad-CAM (Gradient-weighted Class Activation Map)
- Visualizes which regions influenced the prediction
- Target layer: `model.layer4[-1]` (last residual block)
- Output: Heatmap overlaid on original image

### Usage
- Red/warm colors: High influence on prediction
- Blue/cool colors: Low influence

## Performance Metrics

### Test Set Results (624 images)
- Accuracy: 90.1%
- Pneumonia Recall: 96% (important for clinical use)
- Pneumonia Precision: 89%
- F1-Score: 0.92
- ROC-AUC: 0.9647

### Why Recall Matters
In medical diagnosis, false negatives (missed cases) are more costly than false positives. 96% recall means we catch 96% of actual pneumonia cases.

## Deployment

### Local
```bash
python src/app.py
```
Runs on http://localhost:7860

### Production Considerations
- Model is CPU-compatible (can add GPU support)
- Gradio handles request/response serialization
- Input validation on image upload
- Error handling for invalid inputs

## Future Improvements

1. **Multi-class classification**: Distinguish bacterial vs viral pneumonia
2. **Ensemble methods**: Combine multiple models
3. **Model compression**: Quantization for edge deployment
4. **Confidence calibration**: Better probability estimates
5. **Batch inference**: Process multiple images efficiently
6. **Model monitoring**: Track performance on production data
7. **Uncertainty estimation**: Provide confidence intervals

## References

- He et al. (2015) - Deep Residual Learning for Image Recognition
- Selvaraju et al. (2016) - Grad-CAM: Visual Explanations from Deep Networks
- Kaggle Dataset - Chest X-Ray Images (Pneumonia)
