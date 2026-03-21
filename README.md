# Chest X-Ray Pneumonia Classifier

A deep learning model for detecting pneumonia in chest X-rays. Uses ResNet50 with transfer learning and shows you where the model is looking via Grad-CAM heatmaps.

## The numbers

**90.1% accuracy** on 624 test images. More importantly, it catches **96% of pneumonia cases** — that's what actually matters clinically. ROC-AUC is 0.9647.

Breakdown:
- Pneumonia recall: 96% (catches the cases)
- Pneumonia precision: 89%
- F1-score: 0.92

## What you get

- **Web interface** — upload an X-ray, get a prediction instantly
- **Heatmap visualization** — see which parts of the image the model paid attention to
- **ResNet50 backbone** — transfer learned from ImageNet
- **Proper pre-processing** — ImageNet normalization, data augmentation
- **Clean training pipeline** — early stopping, learning rate scheduling, gradient clipping

## Quick start

Need Python 3.8+.

```bash
git clone https://github.com/sameerfcb/Chest-X-Ray-Disease-Classifier.git
cd Chest-X-Ray-Disease-Classifier
python -m venv venv
source venv/bin/activate  # or: venv\Scripts\activate (Windows)
pip install -r requirements.txt
python app.py
```

Then open http://localhost:7860 in your browser. Upload an X-ray and you're done.

To see the training pipeline:
```bash
jupyter notebook .devcontainer/d_classifier.ipynb
```

## How it works

**Backbone:** ResNet50 from ImageNet. We freeze the early layers (conv1, layer1-3) and fine-tune layer4 to adapt to X-rays. This balances keeping general knowledge with learning domain-specific patterns.

**Classification head:** Three-layer network:
```
2048 features → 1024 (batch norm + relu + dropout)
   1024 → 512 (batch norm + relu + dropout)
   512 → 1 (sigmoid for 0-1 predictions)
```

**Training:** Adam optimizer with different learning rates for different layers (1e-4 for layer4, 1e-3 for the head). Cosine annealing scheduler over 15 epochs with early stopping. Standard ImageNet normalization. Data augmentation: rotations, affine transforms, color jitter.

## What improved from baseline

Started at 66.7% accuracy. These changes helped:
- Added ImageNet normalization (was missing)
- Proper data augmentation
- Fine-tuning layer4 instead of freezing everything
- Better classification head with batch norm
- Longer training with early stopping
- Differential learning rates

Ended up at **90.1%, +23.4% improvement**.

## Results on test set

```
                ACTUAL
              NORMAL  PNEUMONIA
PRED  NORMAL    187        15
      PNEUMONIA   47       375
```

- 96% of pneumonia cases caught (375/390)
- Only 15 false negatives (the misses)
- 93% precision on normal cases
- ROC-AUC 0.9647

## Using it

### Via the web app
```bash
python app.py
```
Upload images at http://localhost:7860.

### Via Python
```python
import torch
from torchvision import models, transforms
from PIL import Image

# Setup
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

model = models.resnet50(weights=None)
model.fc = torch.nn.Sequential(
    torch.nn.Linear(2048, 1024),
    torch.nn.BatchNorm1d(1024),
    torch.nn.ReLU(inplace=True),
    torch.nn.Dropout(0.4),
    torch.nn.Linear(1024, 512),
    torch.nn.BatchNorm1d(512),
    torch.nn.ReLU(inplace=True),
    torch.nn.Dropout(0.4),
    torch.nn.Linear(512, 1),
)
model.load_state_dict(torch.load('xray_model_best.pth', map_location='cpu'))
model.eval()

# Predict
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

image = Image.open('xray.jpg').convert('RGB')
x = transform(image).unsqueeze(0)

with torch.no_grad():
    logit = model(x).item()
    prob = torch.sigmoid(torch.tensor(logit)).item()
    label = "PNEUMONIA" if prob > 0.5 else "NORMAL"
    confidence = prob if label == "PNEUMONIA" else 1 - prob

print(f"{label} ({confidence*100:.1f}%)")
```

## Project layout

```
.
├── app.py                  # Web interface
├── requirements.txt        # Dependencies
├── xray_model_best.pth    # Weights (90.1% accuracy)
├── xray_model.pth         # Latest checkpoint
├── README.md              # This file
├── .devcontainer/
│   └── d_classifier.ipynb # Full training notebook
└── flagged/               # Gradio cache
```

## Data

From [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). 5,800 training images, 1,000 validation, 600 test. Binary: Normal vs Pneumonia.

## Setup options

**Local:** `pip install -r requirements.txt` and `python app.py`.

**Dev Container:** Have Docker? Open in VS Code and click "Reopen in Container".

**Codespaces:** [![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/sameerfcb/Chest-X-Ray-Disease-Classifier)

## Dependencies

See `requirements.txt`. Main ones: PyTorch, Torchvision, Gradio, PyTorch-Grad-CAM, scikit-learn.

## Important

**This is research/learning code, not a medical device.** Don't use for actual diagnosis. It's not FDA-approved and only trained on one dataset. Always consult healthcare professionals for real medical decisions.

Model from Kermany et al., 2018.

## References

- He et al. (2015) — Deep Residual Learning for Image Recognition (ResNet)
- Selvaraju et al. (2016) — Grad-CAM: Visual Explanations from Deep Networks

## License

MIT.

## What you'll learn

- Transfer learning
- Medical imaging classification
- Data augmentation strategies
- Model explainability
- Building web interfaces with Gradio
- Hyperparameter tuning
- Production deployment

## Ideas for extension

- Multi-class (bacterial vs viral pneumonia)
- Model ensemble
- Batch processing
- Mobile optimization
- Confidence calibration

## Author

Sameer

---

Last updated: March 21, 2026  
Model v2.0 • 90.1% accuracy
