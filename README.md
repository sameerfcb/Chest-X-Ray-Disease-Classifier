# Chest X-Ray Pneumonia Classifier

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Model Accuracy](https://img.shields.io/badge/Accuracy-90.1%25-brightgreen.svg)]()
[![Pneumonia Recall](https://img.shields.io/badge/Recall-96%25-brightgreen.svg)]()

Deep learning model for detecting pneumonia in chest X-rays. Trained on ResNet50 with some tweaks that actually worked really well. Also shows you where the model's looking using Grad-CAM, which is kinda cool for understanding why it makes certain predictions.

## Why I made this

Started working on this after realizing how hard it is to accurately spot pneumonia in X-rays. Baseline model was sitting at ~66% which... wasn't great. Played around with different approaches - data augmentation, fine-tuning strategies, better training - and managed to bump it up to 90.1%. The important thing though? It catches 96% of actual pneumonia cases. That matters way more than just raw accuracy when you're dealing with medical stuff.

## Results

**90.1%** accuracy on 624 test images. But honestly the recall is what I'm proud of - **96%** of pneumonia cases get caught. ROC-AUC is 0.9647, which means the model's pretty good at distinguishing between the two classes without messing up too much.

Quick breakdown:
- 96% recall (catches pneumonia when it's there)
- 89% precision (doesn't flag normal X-rays as pneumonia too often)
- Only 15 false negatives out of 390 pneumonia cases (that's good)

## Getting it running

Just need Python 3.8+. Clone and set up a virtual environment:

```bash
git clone https://github.com/sameerfcb/Chest-X-Ray-Disease-Classifier.git
cd Chest-X-Ray-Disease-Classifier
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501 and start uploading X-rays. The model spits out a prediction plus a heatmap showing which parts influenced the decision.

If you want to see how everything was trained, there's a Jupyter notebook in the `.devcontainer/` folder with the full pipeline.

## The technical stuff

I used ResNet50 as the backbone since it's been trained on ImageNet and already knows how to extract useful features. The trick was not freezing everything - I kept the first few layers frozen (they learn general stuff) but fine-tuned the last residual block to adapt to X-rays specifically.

The classification head is a 3-layer network:

```
2048 → 1024 (batch norm + ReLU + dropout)
 ↓
1024 → 512 (batch norm + ReLU + dropout)
 ↓
512 → 1 (sigmoid)
```

Nothing fancy but it works. Training uses Adam with different learning rates for different layers, cosine annealing over 15 epochs, and early stopping if validation doesn't improve. Images get normalized with ImageNet stats and some basic augmentation to avoid overfitting.

## How I got from 66% to 90%

Started with a baseline that wasn't doing much. Main issues were:
- No normalization (just raw tensors)
- No data augmentation
- Fully frozen backbone (couldn't adapt to X-rays)
- Weak 2-layer classification head
- Only 5 epochs of training

So I:
1. Added proper ImageNet normalization
2. Threw in decent data augmentation
3. Unfroze layer4 for medical imaging adaptation
4. Built a better 3-layer head with batch norm
5. Trained for 15 epochs with validation monitoring
6. Implemented differential learning rates

Result: +23.4% improvement. Not bad.

## Using it

### Web app

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser and start uploading X-rays.

### In Python

```python
import torch
from torchvision import models, transforms
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Load model
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
    prob = torch.sigmoid(model(x)).item()
    label = "PNEUMONIA" if prob > 0.5 else "NORMAL"
    confidence = prob if label == "PNEUMONIA" else 1 - prob

print(f"{label} ({confidence*100:.1f}%)")
```

## Project structure

```
.
├── app.py                  # Streamlit web app
├── requirements.txt        # Dependencies
├── xray_model_best.pth    # Best weights (use this one)
├── xray_model.pth         # Latest checkpoint
├── .devcontainer/
│   └── d_classifier.ipynb # Full training code/notebook
└── docs/
    ├── ARCHITECTURE.md    # Technical deep dive
    ├── TRAINING.md        # Training details
    └── DEPLOYMENT.md      # How to deploy
```

## Data

Used the [Kaggle chest X-ray dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). About 5,800 training images, 1,000 validation, 600 test. Just binary - normal or pneumonia.

## Setup

**Local:** Install requirements and run `streamlit run app.py`. That's it.

**Docker:** If you want everything containerized, just use the Dockerfile.

**Cloud:** Can deploy to Streamlit Cloud, Heroku, AWS, or Google Cloud Run. See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

## Important stuff

Seriously - **don't use this for actual medical diagnosis.** It's a fun ML project, not an FDA-approved medical device. Model was trained on one specific dataset, there's no telling how it'll perform on different X-ray equipment or patient populations. Always let actual doctors make medical decisions.

Also trained on data from Kermany et al. (2018) which has its own limitations.

## What surprised me

The Grad-CAM visualization honestly. When you see which parts of the X-ray the model focused on, it sometimes makes you realize it's looking at reasonable anatomical features. Other times you're like "why are you looking there?" but then you figure out the pattern anyway. Really helpful for building intuition about what the model's doing.

Also how much data augmentation actually mattered. Thought the improvements would come from the architecture tweaks, but augmentation legit moved the needle.

## What could be better

- Validation on diverse datasets (different equipment, hospitals, etc)
- Ensemble with other models
- Multi-class classification (differentiate pneumonia types)
- Batch inference pipeline
- Confidence calibration
- Some kind of active learning setup
- Mobile deployment

## References

- He et al. (2015) - ResNet paper
- Selvaraju et al. (2016) - Grad-CAM
- Dataset creators: Kermany et al. (2018)

## License

MIT. Do whatever with it.

---

Built by Sameer  
March 2026

