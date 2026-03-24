# API Documentation

## Overview

The Chest X-Ray Pneumonia Classifier provides a simple interface for classifying chest X-rays. This document outlines the key functions and their usage.

## Functions

### `find_weights_path(filename: str = "xray_model.pth") -> Path`

Locates model weights file in the current working directory or parent directories.

**Parameters:**
- `filename` (str, optional): Name of the weights file. Defaults to `"xray_model.pth"`.

**Returns:**
- `Path`: Absolute path to the weights file.

**Raises:**
- `FileNotFoundError`: If the weight file cannot be found in CWD or parent directories.

**Example:**
```python
from pathlib import Path
weights_path = find_weights_path("xray_model_best.pth")
print(f"Model weights found at: {weights_path}")
```

**Notes:**
- Searches recursively from current directory up through parent directories
- Useful for locating model checkpoints in different runtime environments (local, Docker, cloud)
- Prioritizes current directory, then searches parent directories

---

### `build_model(device: torch.device) -> torch.nn.Module`

Constructs and returns a ResNet50-based model with a custom 3-layer classification head.

**Parameters:**
- `device` (torch.device): Target device for model (e.g., "cpu", "cuda:0", "mps").

**Returns:**
- `torch.nn.Module`: ResNet50 model in evaluation mode.

**Architecture Details:**
- Base: ResNet50 (pretrained weights not loaded)
- Custom FC Head:
  - Layer 1: 2048 → 1024 (BatchNorm, ReLU, Dropout 0.4)
  - Layer 2: 1024 → 512 (BatchNorm, ReLU, Dropout 0.4)
  - Layer 3: 512 → 1 (Binary classification logit)

**Example:**
```python
import torch
from torch import device

device = torch.device("cpu")
model = build_model(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
```

**Notes:**
- Model is set to evaluation mode by default
- Weights are randomly initialized (expects to be loaded from checkpoint)
- Designed for binary classification (pneumonia vs. normal)

---

### `predict(image: np.ndarray) -> Tuple[str, np.ndarray]`

Classifies a chest X-ray image and generates explainability visualization.

**Parameters:**
- `image` (np.ndarray): Input image array (typically from Gradio upload).
  - Shape: (H, W, 3) for RGB or (H, W) for grayscale
  - Value range: 0-255

**Returns:**
- Tuple of:
  - `str`: Prediction label and confidence (e.g., "PNEUMONIA (92.5% confidence)")
  - `np.ndarray`: Grad-CAM heatmap overlay (shape: (224, 224, 3))

**Example:**
```python
import numpy as np
from PIL import Image

# Load and prepare image
img = Image.open("sample_xray.jpg")
img_array = np.array(img)

# Get prediction
label, heatmap = predict(img_array)
print(f"Prediction: {label}")
print(f"Heatmap shape: {heatmap.shape}")
```

**Classification Logic:**
- Probability ≥ 0.5 → "PNEUMONIA"
- Probability < 0.5 → "NORMAL"

**Confidence Calculation:**
- For PNEUMONIA: confidence = probability
- For NORMAL: confidence = 1 - probability

**Visualization:**
- Grad-CAM highlights regions the model used for its decision
- Red regions = higher activation
- Blue regions = lower activation

**Notes:**
- Input image is resized to 224×224
- Uses ImageNet normalization (mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225])
- Model operates in inference mode (no gradient computation)

---

## Data Preprocessing

### ImageNet Normalization

The model expects ImageNet-normalized input:

```python
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])
```

**Important:** Always use the same normalization as training.

---

## Model Specifications

| Metric | Value |
|--------|-------|
| Architecture | ResNet50 with Custom Head |
| Input Size | 224 × 224 pixels |
| Input Channels | 3 (RGB) |
| Output Type | Binary Logit |
| Model Size | ~103 MB |
| Inference Device | CPU (or GPU/MPS if available) |
| Accuracy (Test Set) | 90.1% |
| Pneumonia Recall | 96% |
| ROC-AUC | 0.9647 |

---

## Usage Examples

### Basic Prediction

```python
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Initialize
device = torch.device("cpu")
model = build_model(device)
weights = find_weights_path("xray_model_best.pth")
state = torch.load(weights, map_location=device)
model.load_state_dict(state, strict=True)

# Load image
img = Image.open("chest_xray.jpg").convert("RGB")
img_array = np.array(img)

# Predict
label, heatmap = predict(img_array)
print(label)
```

### Batch Inference

```python
import torch
import torchvision.transforms as transforms
from pathlib import Path

def batch_predict(image_paths: list):
    """Predict on multiple images."""
    results = []
    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        img_array = np.array(img)
        label, heatmap = predict(img_array)
        results.append({
            'path': img_path,
            'prediction': label,
            'heatmap': heatmap
        })
    return results
```

### Integration with Web Service

```python
# The app already includes Gradio integration
# Run with: python app.py
# Access at: http://localhost:7860
```

---

## Performance Characteristics

### Inference Time
- Single image: ~50-100ms (CPU)
- Includes preprocessing and Grad-CAM generation

### Memory Requirements
- Model weights: ~103 MB
- Per-image memory: ~50-100 MB
- Minimum recommended: 512 MB RAM

### Device Support
- CPU: ✓ Supported
- CUDA: ✓ Supported (if PyTorch CUDA installed)
- Apple Metal Performance Shaders (MPS): ✓ Supported

---

## Error Handling

### No Image Provided
```
Input: None
Output: "Please upload an image.", None
```

### Model Not Found
```
FileNotFoundError: Missing xray_model_best.pth
```

**Solution:** Ensure checkpoint exists in project directory or parent directories.

### Device Issues
```python
# Automatically handles device selection
device = torch.device("cpu")  # Falls back to CPU if GPU unavailable
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Model file not found | Check working directory and parent folders for xray_model.pth |
| Out of memory | Reduce batch size, close other applications, or use CPU instead of GPU |
| Slow inference | Ensure you're using the correct device (GPU if available) |
| Import errors | Run `pip install -r requirements.txt` |

---

## References

- [PyTorch Documentation](https://pytorch.org/docs/)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [Grad-CAM Paper](https://arxiv.org/abs/1610.02055)
- [Gradio Documentation](https://gradio.app/)
