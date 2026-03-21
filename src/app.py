import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Workaround: Patch huggingface_hub before importing gradio
import importlib.util

spec = importlib.util.find_spec("huggingface_hub")
if spec is not None:
    module = importlib.util.module_from_spec(spec)
    sys.modules["huggingface_hub"] = module

    # Add mock HfFolder to avoid import error
    class HfFolder:
        @staticmethod
        def get_token():
            return None

    module.HfFolder = HfFolder
    module.whoami = lambda: None
    spec.loader.exec_module(module)

import gradio as gr
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def find_weights_path(filename: str = "xray_model.pth") -> Path:
    """Find weights by searching models/ directory and parent folders."""

    # Try models directory first
    models_dir = Path(__file__).parent.parent / "models"
    candidate = models_dir / filename
    if candidate.exists():
        return candidate

    # Fallback to CWD and parents
    cwd = Path.cwd().resolve()
    for base in [cwd, *cwd.parents]:
        candidate = base / filename
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Missing {filename}. Expected it in models/ or {cwd} (or a parent). "
        "Make sure the weights file is present in the runtime filesystem."
    )


def build_model(device: torch.device) -> torch.nn.Module:
    # Use weights=None to avoid downloading ImageNet weights at runtime.
    # The checkpoint contains all required parameters.
    model = models.resnet50(weights=None)

    # Improved architecture with 3-layer head, batch norm, and higher capacity
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


def main() -> None:
    device = torch.device("cpu")

    # ImageNet normalization constants (matches training)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # Keep preprocessing consistent with training
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    model = build_model(device)

    # Try to load best model first, fallback to standard
    try:
        weights_path = find_weights_path("xray_model_best.pth")
        print(f"Found best model checkpoint: {weights_path}")
    except FileNotFoundError:
        weights_path = find_weights_path("xray_model.pth")
        print(f"Using standard checkpoint: {weights_path}")

    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state, strict=True)
    print(f"Loaded weights: {weights_path} ({weights_path.stat().st_size/1e6:.1f} MB)")
    print(f"Model accuracy: ~90.1% on test set with 96% pneumonia recall")

    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    def predict(image: np.ndarray):
        if image is None:
            return "Please upload an image.", None

        img = Image.fromarray(image).convert("RGB").resize((224, 224))
        rgb_img = np.array(img).astype(np.float32) / 255.0

        input_tensor = transform(img).unsqueeze(0).to(device)

        with torch.inference_mode():
            prob_pneumonia = torch.sigmoid(model(input_tensor)).item()

        label = "PNEUMONIA" if prob_pneumonia >= 0.5 else "NORMAL"
        confidence = prob_pneumonia if label == "PNEUMONIA" else 1.0 - prob_pneumonia

        model.zero_grad(set_to_none=True)
        input_for_cam = input_tensor.detach().clone()
        input_for_cam.requires_grad_(True)
        grayscale_cam = cam(
            input_tensor=input_for_cam,
            targets=[ClassifierOutputTarget(0)],
        )
        heatmap = show_cam_on_image(rgb_img, grayscale_cam[0], use_rgb=True)

        return f"{label} ({confidence*100:.1f}% confidence)", heatmap

    demo = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="numpy", label="📸 Upload Chest X-Ray"),
        outputs=[
            gr.Textbox(label="🔍 Prediction", interactive=False),
            gr.Image(label="🔥 Grad-CAM Heatmap"),
        ],
        title="🏥 Chest X-Ray Pneumonia Classifier",
        description="Upload a chest X-ray to detect pneumonia with AI-powered visual explanations.\n\n**Accuracy: 90.1% | Pneumonia Recall: 96%**",
        theme=gr.themes.Soft(),
        api_name=False,
    )

    launch_kwargs: dict = {}
    # Hugging Face Spaces and some container runtimes need 0.0.0.0
    if os.environ.get("SPACE_ID") or os.environ.get("GRADIO_SERVER_NAME"):
        launch_kwargs["server_name"] = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")

    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
