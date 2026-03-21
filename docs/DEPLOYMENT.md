# Deployment Guide

## Local Deployment

### Quick Start

```bash
# 1. Clone repository
git clone https://github.com/sameerfcb/Chest-X-Ray-Disease-Classifier.git
cd Chest-X-Ray-Disease-Classifier

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download model weights
# (Included in repo: xray_model_best.pth)

# 5. Launch app
python src/app.py
```

**Access**: Open browser to http://localhost:7860

## Docker Deployment

### Build Image

```bash
docker build -t chest-xray-classifier .
```

### Run Container

```bash
docker run -p 7860:7860 chest-xray-classifier
```

**Access**: http://localhost:7860

## Cloud Deployment Options

### Option 1: Hugging Face Spaces

1. Create account at [Hugging Face](https://huggingface.co)
2. Create new Space
3. Upload repository files
4. Space automatically detects and runs `app.py`
5. Get public URL in minutes

**Advantages**: Free, instant deployment, easy sharing

### Option 2: AWS

Deploy on **AWS SageMaker**:

```bash
# Create endpoint
aws sagemaker create-endpoint \
  --endpoint-name xray-classifier \
  --endpoint-config-name xray-config

# Query predictions
aws sagemaker-runtime invoke-endpoint \
  --endpoint-name xray-classifier \
  --body image.jpg \
  --content-type image/jpeg \
  output.json
```

### Option 3: Google Cloud

Deploy on **Google Cloud Run**:

```bash
# Build and push image
gcloud builds submit --tag gcr.io/PROJECT-ID/xray-classifier

# Deploy
gcloud run deploy xray-classifier \
  --image gcr.io/PROJECT-ID/xray-classifier \
  --platform managed
```

### Option 4: Azure

Deploy on **Azure Container Instances**:

```bash
# Push to Azure Container Registry
az acr build --registry myregistry --image xray-classifier .

# Deploy instance
az container create \
  --resource-group mygroup \
  --name xray-app \
  --image myregistry.azurecr.io/xray-classifier \
  --ports 7860
```

## Environment Configuration

### Environment Variables

```bash
# Server configuration
export GRADIO_SERVER_NAME=0.0.0.0
export GRADIO_SERVER_PORT=7860

# Optional: Performance tuning
export TORCH_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0  # GPU device (if available)
```

### Performance Tuning

**CPU-only mode** (default):
```bash
python src/app.py
```

**GPU acceleration**:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
python src/app.py  # Automatically uses GPU
```

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 2 cores | 4+ cores |
| RAM | 2 GB | 4+ GB |
| Storage | 500 MB | 1 GB |
| Internet | For UI | Not needed |

## Monitoring & Logging

### Gradio Logs

```bash
# Run with verbose logging
python src/app.py 2>&1 | tee app.log

# Monitor in real-time
tail -f app.log
```

### Predictions Cache

Gradio saves predictions to `flagged/` directory:
```
flagged/
├── images/        # User submitted images
├── predictions/   # Model outputs
└── heatmaps/      # Grad-CAM visualizations
```

## Security Considerations

1. **Model Weights**: Keep `xray_model_best.pth` secure
2. **Medical Data**: Ensure HIPAA compliance for patient data
3. **API Keys**: Never commit secrets to git
4. **HTTPS**: Use SSL/TLS in production
5. **Authentication**: Consider adding access control

## Troubleshooting

### Port Already in Use

```bash
# Find process using port 7860
lsof -i :7860

# Kill process
kill -9 <PID>

# Or use different port
export GRADIO_SERVER_PORT=7861
python src/app.py
```

### GPU Not Detected

```bash
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall GPU PyTorch version
```

### Out of Memory

```bash
# Reduce batch processing or limit concurrent users
# In app.py: consider smaller image sizes

# Or run on more capable hardware
```

## Production Checklist

- [ ] Use production-grade model repository
- [ ] Enable HTTPS/SSL
- [ ] Set up monitoring and alerts
- [ ] Configure logging pipeline
- [ ] Implement rate limiting
- [ ] Regular model updates
- [ ] User feedback collection
- [ ] Documentation maintenance
- [ ] Backup strategy
- [ ] Disaster recovery plan

## Performance Benchmarks

| Metric | Value |
|--------|-------|
| Average Inference Time | ~200ms (CPU) |
| Memory Usage | ~400 MB |
| Max Concurrent Users | 10 (CPU) |
| Throughput | 5-10 predictions/sec |

## Support & Resources

- **Issues**: [GitHub Issues](https://github.com/sameerfcb/Chest-X-Ray-Disease-Classifier/issues)
- **Docs**: [README.md](../README.md)
- **Architecture**: [ARCHITECTURE.md](ARCHITECTURE.md)

---

For questions or deployment support, open an issue on GitHub!
