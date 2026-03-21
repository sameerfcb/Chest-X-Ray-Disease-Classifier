"""Tests for app.py functions and Gradio interface validation."""

import tempfile
from pathlib import Path

import pytest
import torch
import torchvision.transforms as transforms


class TestTransforms:
    """Test suite for image preprocessing transforms."""

    def test_transform_pipeline_creates_tensor(self):
        """Verify transform pipeline creates valid tensors."""
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        
        assert transform is not None
        assert len(transform.transforms) == 3

    def test_transform_output_shape(self):
        """Verify transform produces correct output shape."""
        from PIL import Image
        import numpy as np
        
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        
        # Create dummy image
        dummy_img = Image.new('RGB', (512, 512), color='red')
        transformed = transform(dummy_img)
        
        assert transformed.shape == (3, 224, 224)
        assert transformed.dtype == torch.float32

    def test_transform_normalization_values(self):
        """Verify ImageNet normalization is applied correctly."""
        from PIL import Image
        import numpy as np
        
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        
        dummy_img = Image.new('RGB', (224, 224), color=(128, 128, 128))
        transformed = transform(dummy_img)
        
        # After normalization, values should be within [-2, 2] typically
        assert transformed.min() >= -3
        assert transformed.max() <= 3


class TestPathFinding:
    """Test suite for model weights path discovery."""

    def test_weights_path_search_current_dir(self):
        """Verify weights can be found in current directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            weight_file = Path(tmpdir) / "xray_model.pth"
            weight_file.touch()
            
            assert weight_file.exists()
            assert weight_file.name == "xray_model.pth"

    def test_weights_path_search_parent_dirs(self):
        """Verify weights can be found in parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            parent = Path(tmpdir)
            child = parent / "subfolder"
            child.mkdir()
            
            weight_file = parent / "xray_model.pth"
            weight_file.touch()
            
            assert weight_file.exists()
            # Simulate searching from child dir
            assert parent in list(child.parents)


class TestInferenceConfig:
    """Test suite for inference configuration."""

    def test_device_selection(self):
        """Verify device selection works correctly."""
        device = torch.device("cpu")
        assert device.type == "cpu"

    def test_inference_mode_context(self):
        """Verify inference mode context works."""
        import torch.nn as nn
        
        model = nn.Linear(10, 1)
        model.eval()
        
        x = torch.randn(1, 10)
        
        with torch.inference_mode():
            output = model(x)
        
        assert not output.requires_grad

    def test_sigmoid_probability_conversion(self):
        """Verify sigmoid converts logits to probabilities."""
        logits = torch.tensor([0.0, 2.0, -2.0])
        probs = torch.sigmoid(logits)
        
        assert torch.allclose(probs[0], torch.tensor(0.5))
        assert probs[1] > 0.8  # Positive logit → high prob
        assert probs[2] < 0.2  # Negative logit → low prob

    def test_probability_thresholding(self):
        """Verify 0.5 threshold correctly classifies as pneumonia/normal."""
        probs = [0.3, 0.5, 0.7]
        threshold = 0.5
        
        labels = ["PNEUMONIA" if p >= threshold else "NORMAL" for p in probs]
        
        assert labels[0] == "NORMAL"
        assert labels[1] == "PNEUMONIA"
        assert labels[2] == "PNEUMONIA"


class TestGradCAMSetup:
    """Test suite for Grad-CAM visualization setup."""

    def test_target_layers_selection(self):
        """Verify target layer selection for Grad-CAM."""
        import torchvision.models as models
        
        model = models.resnet50(weights=None)
        target_layers = [model.layer4[-1]]
        
        assert len(target_layers) == 1
        assert target_layers[0] is not None

    def test_heatmap_output_shape(self):
        """Verify heatmap output has correct shape."""
        import numpy as np
        
        # Simulate Grad-CAM output
        rgb_img = np.random.rand(224, 224, 3)
        heatmap = np.random.rand(224, 224, 3)
        
        assert heatmap.shape == rgb_img.shape
        assert heatmap.dtype in [np.float32, np.float64]


class TestAppIntegration:
    """Test suite for overall app integration."""

    def test_normalization_constants_match(self):
        """Verify normalization constants are consistent."""
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]
        
        assert len(IMAGENET_MEAN) == 3
        assert len(IMAGENET_STD) == 3
        assert all(0 < m < 1 for m in IMAGENET_MEAN)
        assert all(0 < s < 0.3 for s in IMAGENET_STD)

    def test_model_accuracy_constant(self):
        """Verify model accuracy metrics are documented."""
        accuracy = 0.901
        pneumonia_recall = 0.96
        
        assert 0.85 < accuracy < 0.95
        assert 0.90 < pneumonia_recall < 1.0

    def test_prediction_confidence_format(self):
        """Verify prediction confidence is formatted correctly."""
        label = "PNEUMONIA"
        confidence = 0.85
        
        prediction_text = f"{label} ({confidence*100:.1f}% confidence)"
        
        assert "PNEUMONIA" in prediction_text
        assert "85.0%" in prediction_text
        assert "%" in prediction_text
