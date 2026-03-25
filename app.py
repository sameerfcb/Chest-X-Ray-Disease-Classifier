from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

from grad_cam import GradCAM
from grad_cam.utils.image import show_cam_on_image
from grad_cam.utils.model_targets import ClassifierOutputTarget


def find_weights_path(filename: str = "xray_model.pth") -> Path:
    """Find model weights file including models/ folder."""
    cwd = Path.cwd().resolve()

    possible_paths = [
        cwd / filename,
        cwd / "models" / filename,
        *[p / filename for p in cwd.parents],
        *[p / "models" / filename for p in cwd.parents],
    ]

    for path in possible_paths:
        if path.exists():
            return path

    raise FileNotFoundError(f"Missing {filename}. Checked: {possible_paths}")


def build_model(device: torch.device) -> torch.nn.Module:
    """
    Build ResNet50 model with custom 3-layer classification head.

    Constructs a ResNet50-based architecture optimized for pneumonia detection.
    The custom head improves upon the standard ResNet50 fc layer with:
    - Multi-layer architecture (2048 -> 1024 -> 512 -> 1)
    - Batch normalization between layers
    - Dropout regularization (0.4)
    - Better generalization on medical imaging tasks

    Args:
        device (torch.device): Target device (cpu, cuda:0, mps, etc.)

    Returns:
        torch.nn.Module: ResNet50 model in eval mode, ready for inference

    Note:
        - Weights are not pretrained (weights=None)
        - Should be loaded from checkpoint file
        - Expects 224x224 RGB input images
    """
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
    """
    Initialize and launch the Streamlit web interface for X-ray classification.

    This function:
    1. Sets up the device (CPU/GPU/MPS)
    2. Loads the trained model and weights
    3. Configures image preprocessing pipeline
    4. Creates the Streamlit interface
    5. Handles image uploads and predictions

    The web interface accepts X-ray images and returns:
    - Classification prediction (PNEUMONIA or NORMAL)
    - Confidence score
    - Grad-CAM visualization showing which parts influenced the decision
    """
    st.set_page_config(
        page_title="Chest X-Ray Pneumonia Classifier",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Add custom styling
    st.markdown(
        """
        <style>
        .main { padding: 2rem; }
        h1 { color: #1f77b4; text-align: center; }
        .metric-box { padding: 1rem; background-color: #f0f2f6; border-radius: 0.5rem; }
        </style>
    """,
        unsafe_allow_html=True,
    )

    st.title("🏥 Chest X-Ray Pneumonia Classifier")
    st.markdown(
        """
    **AI-Powered Medical Imaging Analysis**

    Upload a chest X-ray to detect pneumonia with deep learning and \
visual explanations.
    """
    )

    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", "90.1%")
    with col2:
        st.metric("Pneumonia Recall", "96%")
    with col3:
        st.metric("ROC-AUC", "0.9647")

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

    # Load model
    model = build_model(device)

    # Try to load best model first, fallback to standard
    try:
        weights_path = find_weights_path("xray_model_best.pth")
        st.info(f"✅ Loaded best model checkpoint: {weights_path}")
    except FileNotFoundError:
        weights_path = find_weights_path("xray_model.pth")
        st.info(f"✅ Loaded model checkpoint: {weights_path}")

    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state, strict=True)

    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    def predict(image: np.ndarray):
        """
        Classify a chest X-ray image and generate explainability visualization.

        This function performs:
        1. Image validation and preprocessing
        2. Model inference to get pneumonia probability
        3. Grad-CAM visualization for model interpretability

        Args:
            image (np.ndarray): Input image (H, W, 3) with values 0-255 (uint8)

        Returns:
            Tuple[str, np.ndarray, np.ndarray]:
                - Prediction label (PNEUMONIA or NORMAL)
                - Confidence score
                - Grad-CAM heatmap overlay or None if invalid
        """
        if image is None:
            return "UNKNOWN", 0.0, None

        # Convert to PIL Image and preserve original size
        original_img = Image.fromarray(image).convert("RGB")
        original_size = original_img.size  # (width, height)

        # Create resized version for model input
        img_resized = original_img.resize((224, 224))

        # Prepare for model inference
        input_tensor = transform(img_resized).unsqueeze(0).to(device)

        with torch.inference_mode():
            logit = model(input_tensor).item()
            prob_pneumonia = torch.sigmoid(torch.tensor(logit)).item()

        # Use higher threshold for PNEUMONIA to reduce false positives (0.6 instead of 0.5)
        # This makes the model more conservative about PNEUMONIA predictions
        threshold = 0.6
        label = "PNEUMONIA" if prob_pneumonia >= threshold else "NORMAL"

        # Calculate confidence - how far from the decision boundary
        if label == "PNEUMONIA":
            # Confidence is how much it exceeds the threshold
            confidence = (prob_pneumonia - threshold) / (1.0 - threshold)
            confidence = min(max(confidence, 0.5), 1.0)  # Clamp between 0.5 and 1.0
        else:
            # Confidence is how much it's below the threshold
            confidence = (threshold - prob_pneumonia) / threshold
            confidence = min(max(confidence, 0.5), 1.0)  # Clamp between 0.5 and 1.0

        # Generate Grad-CAM
        model.zero_grad(set_to_none=True)
        input_for_cam = input_tensor.detach().clone()
        input_for_cam.requires_grad_(True)
        grayscale_cam = cam(
            input_tensor=input_for_cam,
            targets=[ClassifierOutputTarget(0)],
        )

        # Prepare original image for visualization
        rgb_img_original = np.array(original_img).astype(np.float32) / 255.0

        # Resize CAM to match original image size
        cam_resized = Image.fromarray((grayscale_cam[0] * 255).astype(np.uint8)).resize(
            original_size, Image.BILINEAR
        )
        cam_resized_array = np.array(cam_resized).astype(np.float32) / 255.0

        # Generate heatmap with original image size
        heatmap = show_cam_on_image(rgb_img_original, cam_resized_array, use_rgb=True)

        return label, confidence, heatmap

    # Streamlit UI
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "📸 Upload Chest X-Ray",
        type=["jpg", "jpeg", "png", "bmp", "gif"],
        help="Upload a chest X-ray image in JPG, PNG, or BMP format",
    )

    if uploaded_file is not None:
        # Load image once
        image = Image.open(uploaded_file).convert("RGB")
        image_array = np.array(image)

        # Make prediction
        with st.spinner("🔍 Analyzing image..."):
            label, confidence, heatmap = predict(image_array)

        # Display results in columns
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📸 Original X-Ray")
            st.image(image, use_column_width=True, caption="Uploaded Image")

        with col2:
            st.subheader("🔥 Grad-CAM Heatmap")
            st.markdown("Red areas show regions that most influenced the prediction.")
            if heatmap is not None:
                st.image(heatmap, use_column_width=True, caption="Model Attention Map")
            else:
                st.warning("Could not generate Grad-CAM visualization.")

        st.markdown("---")
        st.subheader("🔍 Analysis Results")

        col1, col2, col3 = st.columns(3)
        with col1:
            if label == "PNEUMONIA":
                st.error(f"**{label}**\n{confidence*100:.1f}% confidence", icon="❌")
            else:
                st.success(f"**{label}**\n{confidence*100:.1f}% confidence", icon="✓")

        with col2:
            st.metric("Confidence Level", f"{confidence*100:.1f}%")

        with col3:
            st.progress(confidence, text=f"Score: {confidence:.3f}")

        st.markdown("---")
        st.markdown(
            """
            ### ⚠️ Important Disclaimer
            - **Educational Purpose Only**: This tool is for demonstration purposes.
            - **Not a Medical Device**: Not FDA-approved or clinically validated.
            - **Always Consult Professionals**: Medical decisions must be made by licensed healthcare providers.
            - **Dataset Limitation**: Model trained on specific X-ray dataset; may not generalize to all equipment/populations.
            """
        )
    else:
        st.info("👆 Upload a chest X-ray image to get started")


if __name__ == "__main__":
    main()
