import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Import configuration
from config import config

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



def find_weights_path(filename: str = None) -> Path:
    """Locate model weights using centralized configuration.
    
    Uses config.get_weights_path() to search for model weights across
    multiple directories (cwd, parents, project root).
    
    Args:
        filename (str, optional): Weights filename. Uses config default if None.
    
    Returns:
        Path: Absolute path to weights file
    
    Raises:
        FileNotFoundError: If weights not found in search paths
    """
    return config.get_weights_path(filename)


def build_model(device: torch.device) -> torch.nn.Module:
    """Build ResNet50 model with custom 3-layer classification head.
    
    Creates a ResNet50 backbone with a custom head optimized for pneumonia
    detection. The head includes batch normalization and dropout for better
    generalization on medical imaging tasks.
    
    Architecture (configured in config.py):
        - ResNet50 backbone: 2048-dim feature extractor
        - Custom head: 2048 → 1024 → 512 → 1 (binary classification)
        - BatchNorm and Dropout between layers for regularization
    
    Args:
        device (torch.device): Target device (cpu, cuda, mps, etc.)
    
    Returns:
        torch.nn.Module: ResNet50 model in eval() mode, ready for inference
    
    Notes:
        - Pretrained weights are NOT loaded (weights=None)
        - Model expects external checkpoint loading via torch.load()
        - Output is unbounded logit scale; apply sigmoid() for probability
        - See config.py for architecture dimensions
    """
    # Avoid downloading ImageNet weights at runtime
    model = models.resnet50(weights=None)

    # Build 3-layer head using parameters from config
    layers = []
    head_dims = config.MODEL_HEAD_LAYERS
    
    for i in range(len(head_dims) - 1):
        in_features = head_dims[i]
        out_features = head_dims[i + 1]
        
        # Add linear layer
        layers.append(nn.Linear(in_features, out_features))
        
        # Add batch norm and activation (except for output layer)
        if i < len(head_dims) - 2:
            layers.append(nn.BatchNorm1d(out_features))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(config.DROPOUT_RATE))
    
    model.fc = nn.Sequential(*layers)
    return model.to(device).eval()


def main() -> None:
    """Initialize and launch the Gradio web interface for X-ray classification.
    
    This function orchestrates the entire application:
    1. Initializes PyTorch device from config
    2. Builds and loads the trained ResNet50 model
    3. Configures image preprocessing using ImageNet normalization
    4. Sets up Grad-CAM for model interpretability
    5. Creates Gradio interface with prediction and visualization
    6. Launches web server with config-based settings
    
    All configuration is centralized in config.py including:
        - Model architecture and paths
        - Normalization parameters
        - Classification thresholds
        - Gradio UI settings
        - Server configuration
    
    Environment Variables (override config):
        MODEL_DEVICE: GPU/CPU selection
        GRADIO_SERVER_NAME: Server address
        GRADIO_SERVER_PORT: Server port
        DEBUG_MODE: Enable debug logging
    """
    # Initialize device and model
    device = torch.device(config.DEVICE)
    model = build_model(device)
    
    # Load model weights (try best checkpoint first, then fallback)
    try:
        weights_path = find_weights_path(config.MODEL_WEIGHTS_FILENAME)
        print(f"✓ Loaded best model: {weights_path.name}")
    except FileNotFoundError:
        weights_path = find_weights_path(config.MODEL_WEIGHTS_FALLBACK)
        print(f"✓ Loaded fallback model: {weights_path.name}")
    
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state, strict=True)
    size_mb = weights_path.stat().st_size / 1e6
    print(f"  Model size: {size_mb:.1f} MB")
    print(f"  Performance: {config.MODEL_ACCURACY*100:.1f}% accuracy, "
          f"{config.MODEL_PNEUMONIA_RECALL*100:.0f}% pneumonia recall")
    
    # Setup image preprocessing (ImageNet normalization from config)
    transform = transforms.Compose([
        transforms.Resize((config.MODEL_INPUT_SIZE, config.MODEL_INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])
    
    # Setup Grad-CAM for interpretability visualization
    target_layers = [model.layer4[-1]]  # Last residual block
    cam = GradCAM(model=model, target_layers=target_layers)
    
    def predict(image: np.ndarray):
        """Predict pneumonia and generate Grad-CAM visualization.
        
        Args:
            image (np.ndarray): Input chest X-ray (H, W, 3), values 0-255
        
        Returns:
            tuple: (prediction_text, heatmap_image)
                - prediction_text: "PNEUMONIA (confidence%)" or "NORMAL (confidence%)"
                - heatmap_image: Grad-CAM overlay visualization
        """
        if image is None:
            return "Please upload an image.", None
        
        # Preprocess image
        pil_img = Image.fromarray(image).convert("RGB")
        normalized_img = np.array(pil_img).astype(np.float32) / 255.0
        input_tensor = transform(pil_img).unsqueeze(0).to(device)
        
        # Model inference
        with torch.inference_mode() if config.USE_INFERENCE_MODE else torch.no_grad():
            logit = model(input_tensor).item()
            prob = torch.sigmoid(torch.tensor(logit, device=device)).item()
        
        # Classify (threshold from config)
        is_pneumonia = prob >= config.CLASSIFICATION_THRESHOLD
        label = config.CLASS_LABELS[1 if is_pneumonia else 0]
        confidence = prob if is_pneumonia else (1.0 - prob)
        
        # Generate Grad-CAM heatmap
        model.zero_grad(set_to_none=True)
        input_for_cam = input_tensor.detach().clone()
        input_for_cam.requires_grad_(True)
        grayscale_cam = cam(
            input_tensor=input_for_cam,
            targets=[ClassifierOutputTarget(0)],
        )
        heatmap = show_cam_on_image(
            normalized_img,
            grayscale_cam[0],
            use_rgb=config.GRADCAM_USE_RGB
        )
        
        # Format prediction text
        decimals = config.CONFIDENCE_DECIMAL_PLACES
        confidence_str = f"{confidence*100:.{decimals}f}"
        prediction_text = f"{label} ({confidence_str}% confidence)"
        
        if config.LOG_PREDICTIONS:
            print(f"  Prediction: {prediction_text}")
        
        return prediction_text, heatmap
    
    # Create Gradio interface
    interface = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="numpy", label=config.GRADIO_INPUT_LABEL),
        outputs=[
            gr.Textbox(label=config.GRADIO_OUTPUT_PREDICTION_LABEL, interactive=False),
            gr.Image(label=config.GRADIO_OUTPUT_HEATMAP_LABEL),
        ],
        title=config.GRADIO_TITLE,
        description=config.GRADIO_DESCRIPTION,
        theme=gr.themes.Soft() if config.GRADIO_THEME == "soft" else gr.themes.Default(),
        api_name=config.GRADIO_API_NAME,
    )
    
    # Configure and launch server
    launch_kwargs = {
        "share": config.GRADIO_SHARE,
    }
    
    # Override server settings for Hugging Face Spaces
    if os.environ.get("SPACE_ID") or config.GRADIO_SERVER_NAME:
        launch_kwargs["server_name"] = config.GRADIO_SERVER_NAME or "0.0.0.0"
    
    if config.GRADIO_SERVER_PORT:
        launch_kwargs["server_port"] = config.GRADIO_SERVER_PORT
    
    if config.DEBUG_MODE:
        print(f"🐛 Debug mode enabled")
        print(f"   Device: {device}")
        print(f"   Config: {config.__class__.__name__}")
    
    print(f"\n🚀 Launching web interface...")
    interface.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
