import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path

# Setup
device = torch.device("cpu")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def build_model(device):
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
    return model.to(device).eval()


# Load model
model = build_model(device)

# Try to load weights
try:
    weights_path = None
    cwd = Path.cwd().resolve()
    possible_paths = [
        cwd / "xray_model_best.pth",
        cwd / "models" / "xray_model_best.pth",
        cwd / "xray_model.pth",
        cwd / "models" / "xray_model.pth",
    ]

    for path in possible_paths:
        if path.exists():
            weights_path = path
            break

    if weights_path:
        print(f"Loading model from: {weights_path}")
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state, strict=True)
        print("✓ Model weights loaded successfully")
    else:
        print("✗ Model weights not found")
        print(f"Searched in: {possible_paths}")
except Exception as e:
    print(f"✗ Error loading weights: {e}")

# Create a test image (random noise like a non-pneumonia image)
test_image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
test_pil = Image.fromarray(test_image).convert("RGB")
input_tensor = transform(test_pil).unsqueeze(0).to(device)

with torch.inference_mode():
    logit = model(input_tensor).item()
    prob = torch.sigmoid(torch.tensor(logit)).item()

print(f"\nTest on random image:")
print(f"  Raw logit: {logit:.4f}")
print(f"  Sigmoid prob: {prob:.4f}")
print(f"  Prediction: {'PNEUMONIA' if prob >= 0.5 else 'NORMAL'}")

# Check model output statistics
print(f"\nModel output range check:")
# Test on multiple random images
logits = []
for i in range(10):
    test_img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    test_pil = Image.fromarray(test_img).convert("RGB")
    input_tensor = transform(test_pil).unsqueeze(0).to(device)
    with torch.inference_mode():
        logit = model(input_tensor).item()
        logits.append(logit)

print(f"  Min logit: {min(logits):.4f}")
print(f"  Max logit: {max(logits):.4f}")
print(f"  Mean logit: {np.mean(logits):.4f}")
print(f"  Std logit: {np.std(logits):.4f}")

probs = [1.0 / (1.0 + np.exp(-l)) for l in logits]
print(f"  Min prob: {min(probs):.4f}")
print(f"  Max prob: {max(probs):.4f}")
print(f"  Mean prob: {np.mean(probs):.4f}")
