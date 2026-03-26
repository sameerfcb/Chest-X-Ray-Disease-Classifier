"""Configuration management for Chest X-Ray Disease Classifier.

This module centralizes all configuration parameters used throughout the application,
making it easy to adjust settings without modifying source code. Supports environment
variable overrides for deployment flexibility.

Environment Variables:
    MODEL_WEIGHTS_PATH: Override default model weights location
    MODEL_DEVICE: Override device selection (cpu, cuda, mps)
    CLASSIFICATION_THRESHOLD: Override confidence threshold (0.0-1.0)
    SERVER_NAME: Override Gradio server address
    SERVER_PORT: Override Gradio server port
    DEBUG_MODE: Enable debug logging (true/false)
"""

import os
from pathlib import Path


class Config:
    """Configuration container for model and application settings.
    
    This class defines all configurable parameters with sensible defaults.
    Values can be overridden via environment variables for deployment scenarios.
    
    Attributes:
        MODEL_ARCHITECTURE (str): Architecture type (resnet50)
        MODEL_INPUT_SIZE (int): Expected input image size in pixels
        MODEL_CHANNELS (int): Number of color channels (RGB = 3)
        MODEL_WEIGHTS_FILENAME (str): Default weights filename to search for
        
        DEVICE (str): PyTorch device (cpu, cuda, mps)
        
        IMAGENET_MEAN (list): ImageNet normalization mean values
        IMAGENET_STD (list): ImageNet normalization std values
        
        CLASSIFICATION_THRESHOLD (float): Probability threshold for PNEUMONIA label
        PNEUMONIA_CLASS_ID (int): Output index for pneumonia class
        
        GRADIO_SERVER_NAME (str): Server address (0.0.0.0 for external access)
        GRADIO_SERVER_PORT (int): Port number for web interface
        GRADIO_THEME (str): Gradio UI theme
        
        MAX_FILE_SIZE (int): Maximum upload file size in bytes
        BATCH_SIZE (int): Default batch size for inference
    """
    
    # ===== MODEL ARCHITECTURE =====
    MODEL_ARCHITECTURE = "resnet50"
    MODEL_INPUT_SIZE = 224  # ResNet50 expects 224x224 images
    MODEL_CHANNELS = 3  # RGB images
    MODEL_WEIGHTS_FILENAME = "xray_model_best.pth"  # Search for best model first
    MODEL_WEIGHTS_FALLBACK = "xray_model.pth"  # Fallback to standard model
    
    # Model architecture details
    MODEL_HEAD_LAYERS = [2048, 1024, 512, 1]  # 3-layer head: 2048->1024->512->1
    DROPOUT_RATE = 0.4  # Dropout in classification head
    
    # ===== DEVICE CONFIGURATION =====
    # Auto-detect or override via environment variable
    DEVICE = os.getenv("MODEL_DEVICE", "cpu")
    
    # ===== PREPROCESSING PARAMETERS =====
    # ImageNet normalization values (same as training)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    # ===== CLASSIFICATION PARAMETERS =====
    # Binary classification: probability >= 0.5 means PNEUMONIA
    CLASSIFICATION_THRESHOLD = float(os.getenv("CLASSIFICATION_THRESHOLD", 0.5))
    PNEUMONIA_CLASS_ID = 1  # Output neuron index for pneumonia
    NORMAL_CLASS_ID = 0
    
    # Class labels
    CLASS_LABELS = {
        0: "NORMAL",
        1: "PNEUMONIA"
    }
    
    # Confidence formatting
    CONFIDENCE_DECIMAL_PLACES = 1  # Display as "92.3%"
    
    # ===== MODEL PERFORMANCE METRICS =====
    # These are documented for README and API reference
    MODEL_ACCURACY = 0.901
    MODEL_PNEUMONIA_RECALL = 0.96
    MODEL_SPECIFICITY = 0.85
    MODEL_ROC_AUC = 0.9647
    
    # ===== GRADIO WEB INTERFACE =====
    GRADIO_SERVER_NAME = os.getenv("GRADIO_SERVER_NAME", None)
    GRADIO_SERVER_PORT = int(os.getenv("GRADIO_SERVER_PORT", 7860))
    GRADIO_THEME = "soft"  # Theme options: default, soft, glass
    GRADIO_API_NAME = False  # Disable API endpoint for privacy
    GRADIO_SHARE = os.getenv("GRADIO_SHARE", "false").lower() == "true"
    
    # UI Labels and descriptions
    GRADIO_TITLE = "🏥 Chest X-Ray Pneumonia Classifier"
    GRADIO_DESCRIPTION = (
        "Upload a chest X-ray to detect pneumonia with AI-powered visual explanations.\n\n"
        f"**Accuracy: {MODEL_ACCURACY*100:.1f}% | Pneumonia Recall: {MODEL_PNEUMONIA_RECALL*100:.0f}%**"
    )
    GRADIO_INPUT_LABEL = "📸 Upload Chest X-Ray"
    GRADIO_OUTPUT_PREDICTION_LABEL = "🔍 Prediction"
    GRADIO_OUTPUT_HEATMAP_LABEL = "🔥 Grad-CAM Heatmap"
    
    # ===== FILE UPLOAD CONSTRAINTS =====
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
    ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    
    # ===== INFERENCE PARAMETERS =====
    BATCH_SIZE = 8  # Default batch size for batch inference
    USE_INFERENCE_MODE = True  # Use torch.inference_mode() for memory efficiency
    
    # ===== GRAD-CAM CONFIGURATION =====
    # Which layer to visualize attributions from
    GRADCAM_TARGET_LAYER = "layer4"  # Last residual block for ResNet50
    GRADCAM_USE_RGB = True  # Use RGB for heatmap overlay
    
    # ===== DEBUG AND LOGGING =====
    DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
    LOG_PREDICTIONS = True  # Log each prediction
    LOG_INFERENCE_TIME = True  # Log model inference duration
    
    # ===== MODEL PATHS =====
    @staticmethod
    def get_project_root() -> Path:
        """Get absolute path to project root directory.
        
        Returns:
            Path: Project root directory containing app.py and model weights
        """
        return Path(__file__).parent.resolve()
    
    @staticmethod
    def get_weights_search_path() -> list:
        """Get list of directories to search for model weights.
        
        Searches in order: current directory, parent, then project root.
        
        Returns:
            list: List of Path objects to search for weights
        """
        cwd = Path.cwd().resolve()
        project_root = Config.get_project_root()
        
        # Remove duplicates while preserving order
        search_paths = []
        for path in [cwd, *cwd.parents, project_root]:
            if path not in search_paths:
                search_paths.append(path)
        
        return search_paths
    
    @staticmethod
    def get_weights_path(filename: str = None) -> Path:
        """Locate model weights file with fallback logic.
        
        Args:
            filename (str, optional): Weights filename to search for.
                                     Defaults to MODEL_WEIGHTS_FILENAME.
        
        Returns:
            Path: Absolute path to the weights file
        
        Raises:
            FileNotFoundError: If weights not found in any search path
        """
        filename = filename or Config.MODEL_WEIGHTS_FILENAME
        
        for search_path in Config.get_weights_search_path():
            candidate = search_path / filename
            if candidate.exists():
                return candidate
        
        # If primary weights not found, try fallback
        if filename == Config.MODEL_WEIGHTS_FILENAME:
            for search_path in Config.get_weights_search_path():
                candidate = search_path / Config.MODEL_WEIGHTS_FALLBACK
                if candidate.exists():
                    return candidate
        
        raise FileNotFoundError(
            f"Model weights '{filename}' not found in:\n"
            f"{chr(10).join(str(p) for p in Config.get_weights_search_path())}\n"
            f"Place weights in project root or current working directory."
        )


# ===== ENVIRONMENT-SPECIFIC CONFIGURATIONS =====

class DevelopmentConfig(Config):
    """Configuration for local development.
    
    - Debug mode enabled
    - All logging enabled
    - Gradio shares output URL
    """
    DEBUG_MODE = True
    LOG_PREDICTIONS = True
    LOG_INFERENCE_TIME = True
    GRADIO_SHARE = True


class ProductionConfig(Config):
    """Configuration for production deployment.
    
    - Debug mode disabled
    - Minimal logging
    - Server listens on 0.0.0.0 for external access
    """
    DEBUG_MODE = False
    LOG_PREDICTIONS = False
    GRADIO_SERVER_NAME = "0.0.0.0"
    GRADIO_SHARE = False


class TestingConfig(Config):
    """Configuration for unit testing.
    
    - Reduced batch size
    - CPU device only
    - No file uploads
    """
    DEVICE = "cpu"
    BATCH_SIZE = 1
    MAX_FILE_SIZE = 1024 * 1024  # 1 MB for tests


# ===== CONFIGURATION SELECTION =====

def get_config(environment: str = None) -> Config:
    """Get configuration object based on environment.
    
    Args:
        environment (str, optional): Environment name (development, production, testing).
                                    Defaults to FLASK_ENV or "production".
    
    Returns:
        Config: Configuration object with appropriate settings
    
    Examples:
        >>> config = get_config("development")
        >>> print(config.DEBUG_MODE)
        True
        
        >>> config = get_config()  # Uses environment variable
        >>> model_path = config.get_weights_path()
    """
    environment = environment or os.getenv("ENV", "production").lower()
    
    config_map = {
        "development": DevelopmentConfig,
        "dev": DevelopmentConfig,
        "production": ProductionConfig,
        "prod": ProductionConfig,
        "testing": TestingConfig,
        "test": TestingConfig,
    }
    
    return config_map.get(environment, ProductionConfig)()


# Default configuration instance
config = get_config()
