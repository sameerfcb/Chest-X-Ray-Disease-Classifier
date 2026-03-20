# Chest-X-Ray-Disease-Classifier
Deep learning–powered chest X-ray classifier built with CNN architectures (ResNet-50/EfficientNet) to identify 14 lung diseases, enhanced with Grad-CAM heatmaps for model interpretability and clinical insight.

## Getting Started with VS Code

### Option 1: Dev Container (Recommended)

This repository includes a [Dev Container](https://containers.dev/) configuration for a fully reproducible development environment.

**Prerequisites:** [Docker](https://www.docker.com/get-started) and the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) for VS Code.

1. Clone this repository and open it in VS Code.
2. When prompted, click **"Reopen in Container"**, or open the Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`) and select **"Dev Containers: Reopen in Container"**.
3. VS Code will build the container and install all dependencies automatically.

### Option 2: Local Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/sameerfcb/Chest-X-Ray-Disease-Classifier.git
   cd Chest-X-Ray-Disease-Classifier
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. Open the folder in VS Code. When prompted, install the **recommended extensions** from `.vscode/extensions.json`.

### GitHub Codespaces

You can also open this repository directly in [GitHub Codespaces](https://github.com/features/codespaces) for a zero-setup cloud development environment:

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/sameerfcb/Chest-X-Ray-Disease-Classifier)
