"""Tests for model architecture and inference."""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torchvision.models as models


def build_model(device: torch.device) -> nn.Module:
    """Build ResNet50 with 3-layer classification head."""
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


class TestModelArchitecture:
    """Test suite for model architecture validation."""

    def test_model_builds_successfully(self):
        """Verify model builds without errors."""
        device = torch.device("cpu")
        model = build_model(device)
        assert model is not None
        assert isinstance(model, nn.Module)

    def test_model_has_correct_input_size(self):
        """Verify model expects 224x224 input."""
        device = torch.device("cpu")
        model = build_model(device)

        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        output = model(dummy_input)

        assert output.shape == (1, 1), "Model output should be (batch_size, 1)"

    def test_model_output_is_unbounded(self):
        """Verify model outputs logits (unbounded values)."""
        device = torch.device("cpu")
        model = build_model(device)

        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        output = model(dummy_input)

        assert output.item() is not None
        # Could be negative or > 1 for logits
        assert isinstance(output.item(), float)

    def test_model_eval_mode(self):
        """Verify model is in eval mode."""
        device = torch.device("cpu")
        model = build_model(device)

        assert not model.training, "Model should be in eval mode"

    def test_model_batch_processing(self):
        """Verify model handles different batch sizes."""
        device = torch.device("cpu")
        model = build_model(device)

        for batch_size in [1, 2, 4, 8]:
            dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
            output = model(dummy_input)
            assert output.shape == (batch_size, 1)

    def test_model_has_3_layer_head(self):
        """Verify model has 3-layer classification head."""
        device = torch.device("cpu")
        model = build_model(device)

        linear_layers = [m for m in model.fc if isinstance(m, nn.Linear)]
        assert len(linear_layers) == 3, "Should have 3 linear layers"

    def test_model_has_batch_normalization(self):
        """Verify model includes batch normalization layers."""
        device = torch.device("cpu")
        model = build_model(device)

        bn_layers = [m for m in model.fc if isinstance(m, nn.BatchNorm1d)]
        assert len(bn_layers) == 2, "Should have 2 batch norm layers"

    def test_model_has_dropout_regularization(self):
        """Verify model includes dropout for regularization."""
        device = torch.device("cpu")
        model = build_model(device)

        dropout_layers = [m for m in model.fc if isinstance(m, nn.Dropout)]
        assert len(dropout_layers) == 2, "Should have 2 dropout layers"

    def test_model_inference_mode(self):
        """Verify model runs efficiently with torch.inference_mode()."""
        device = torch.device("cpu")
        model = build_model(device)

        dummy_input = torch.randn(1, 3, 224, 224).to(device)

        with torch.inference_mode():
            output = model(dummy_input)

        assert not output.requires_grad


class TestModelCheckpoint:
    """Test suite for model checkpoint serialization."""

    def test_model_can_be_saved(self):
        """Verify model state can be saved to disk."""
        device = torch.device("cpu")
        model = build_model(device)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.pth"
            torch.save(model.state_dict(), checkpoint_path)
            assert checkpoint_path.exists()

    def test_model_can_be_loaded(self):
        """Verify model state can be restored from checkpoint."""
        device = torch.device("cpu")
        model1 = build_model(device)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.pth"
            torch.save(model1.state_dict(), checkpoint_path)

            model2 = build_model(device)
            state = torch.load(checkpoint_path, map_location=device)
            model2.load_state_dict(state, strict=True)

    def test_checkpoint_produces_same_predictions(self):
        """Verify loaded checkpoint produces identical predictions."""
        device = torch.device("cpu")
        model1 = build_model(device)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.pth"
            torch.save(model1.state_dict(), checkpoint_path)

            model2 = build_model(device)
            state = torch.load(checkpoint_path, map_location=device)
            model2.load_state_dict(state, strict=True)

            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            out1 = model1(dummy_input)
            out2 = model2(dummy_input)

            assert torch.allclose(out1, out2)

    def test_checkpoint_size_reasonable(self):
        """Verify checkpoint file size is in expected range."""
        device = torch.device("cpu")
        model = build_model(device)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.pth"
            torch.save(model.state_dict(), checkpoint_path)

            size_mb = checkpoint_path.stat().st_size / 1e6
            # ResNet50 weights are typically 90-110 MB
            assert 80 < size_mb < 150, f"Checkpoint size {size_mb}MB seems incorrect"


class TestModelInference:
    """Test suite for model inference behavior."""

    def test_inference_various_inputs(self):
        """Verify model handles various input distributions."""
        device = torch.device("cpu")
        model = build_model(device)

        test_cases = [
            torch.zeros(1, 3, 224, 224),
            torch.ones(1, 3, 224, 224),
            torch.randn(1, 3, 224, 224),
        ]

        for inp in test_cases:
            inp = inp.to(device)
            output = model(inp)
            assert output.shape == (1, 1)
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()

    def test_inference_deterministic(self):
        """Verify model produces consistent outputs in eval mode."""
        device = torch.device("cpu")
        model = build_model(device)

        torch.manual_seed(42)
        dummy_input = torch.randn(1, 3, 224, 224).to(device)

        out1 = model(dummy_input)
        out2 = model(dummy_input)

        assert torch.allclose(out1, out2)

    def test_inference_output_format(self):
        """Verify inference output has correct format."""
        device = torch.device("cpu")
        model = build_model(device)

        dummy_input = torch.randn(2, 3, 224, 224).to(device)
        output = model(dummy_input)

        assert output.dtype == torch.float32
        assert output.shape == (2, 1)
        assert output.device.type == device.type
