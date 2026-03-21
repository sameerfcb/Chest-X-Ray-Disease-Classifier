# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2026-03-21

### Added
- Gradio web interface for streamlined model inference
- Production-ready deployment configuration
- Comprehensive test suite with sample predictions
- README badges and improved documentation

### Improved
- Enhanced UI with Grad-CAM visualization
- Better error handling and user feedback
- Optimized inference pipeline

## [1.1.0] - 2026-03-19

### Added
- Comprehensive project documentation (ARCHITECTURE.md, TRAINING.md)
- GitHub issue templates and CONTRIBUTING guide
- Professional project structure (src/, tests/, docs/, models/)
- Extended evaluation metrics and visualization

### Improved
- Model accuracy from 66.7% to 90.1%
- 3-layer classification head with batch normalization
- Data augmentation strategy

## [1.0.0] - 2026-03-13

### Added
- ResNet50 transfer learning model
- Data loading and preprocessing pipeline
- Training loop with validation monitoring
- Grad-CAM explainability features
- Initial evaluation suite

### Features
- 90.1% test accuracy
- 96% pneumonia recall (clinical-grade performance)
- 0.9647 ROC-AUC score
- Early stopping and gradient clipping

## [0.2.0] - 2026-03-09

### Added
- Data exploration notebook
- ImageNet normalization pipeline
- Baseline model architecture
- Initial hyperparameter tuning

## [0.1.0] - 2026-03-07

### Added
- Project initialization
- Basic requirements.txt
- Initial README
- Dataset integration (Kaggle Chest X-Ray)
- License and .gitignore
