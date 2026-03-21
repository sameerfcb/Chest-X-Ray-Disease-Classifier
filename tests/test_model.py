#!/usr/bin/env python
"""Quick test of the model on sample pneumonia and normal X-rays."""

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Patch huggingface_hub like in app.py
import importlib.util

spec = importlib.util.find_spec("huggingface_hub")
if spec is not None:
    module = importlib.util.module_from_spec(spec)
    sys.modules["huggingface_hub"] = module

    class HfFolder:
        @staticmethod
        def get_token():
            return None

    module.HfFolder = HfFolder
    module.whoami = lambda: None
    spec.loader.exec_module(module)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_model(device):
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(2048, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(512, 1),
    )
    return model.to(device).eval()


def find_weights():
    cwd = Path.cwd().resolve()
    for base in [cwd, *cwd.parents]:
        candidate = base / "xray_model_best.pth"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("xray_model_best.pth not found")


def predict_image(model, image_path, transform, device):
    """Make prediction on a single image."""
    try:
        image = Image.open(image_path).convert("RGB")
        x = transform(image).unsqueeze(0).to(device)

        with torch.inference_mode():
            logit = model(x).item()
            prob = torch.sigmoid(torch.tensor(logit)).item()

        label = "PNEUMONIA" if prob >= 0.5 else "NORMAL"
        confidence = prob if label == "PNEUMONIA" else 1 - prob

        return label, confidence
    except Exception as e:
        return None, str(e)


def main():
    device = torch.device("cpu")
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    # Load model
    weights_path = find_weights()
    model = build_model(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state, strict=True)
    print(f"✓ Model loaded from {weights_path.name}\n")

    # Find test images
    data_path = (
        Path.home()
        / ".cache/kagglehub/datasets/paultimothymooney/chest-xray-pneumonia/versions/2/chest_xray/chest_xray/test"
    )
    if not data_path.exists():
        print(f"⚠ Test data not found at {data_path}")
        print("Run the notebook to download the dataset first.")
        return

    print("=" * 60)
    print("TESTING MODEL ON SAMPLE X-RAYS")
    print("=" * 60)

    # Test NORMAL X-rays
    normal_dir = data_path / "NORMAL"
    if normal_dir.exists():
        normal_images = list(normal_dir.glob("*.jpeg"))[:3]  # Take first 3
        if normal_images:
            print("\n📋 NORMAL X-RAY PREDICTIONS:")
            print("-" * 60)
            for img_path in normal_images:
                label, conf = predict_image(model, img_path, transform, device)
                status = "✓" if label == "NORMAL" else "✗"
                print(
                    f"{status} {img_path.name[:40]:40s} → {label:10s} ({conf*100:.1f}%)"
                )

    # Test PNEUMONIA X-rays
    pneumonia_dir = data_path / "PNEUMONIA"
    if pneumonia_dir.exists():
        pneumonia_images = list(pneumonia_dir.glob("*.jpeg"))[:3]  # Take first 3
        if pneumonia_images:
            print("\n🫁 PNEUMONIA X-RAY PREDICTIONS:")
            print("-" * 60)
            for img_path in pneumonia_images:
                label, conf = predict_image(model, img_path, transform, device)
                status = "✓" if label == "PNEUMONIA" else "✗"
                print(
                    f"{status} {img_path.name[:40]:40s} → {label:10s} ({conf*100:.1f}%)"
                )

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
