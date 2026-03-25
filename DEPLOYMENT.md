# Deployment Guide

This guide covers how to deploy the Chest X-Ray Pneumonia Classifier Streamlit application to various platforms.

## Local Development

### Prerequisites
- Python 3.8 or higher
- Virtual environment (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/sameerfcb/Chest-X-Ray-Disease-Classifier.git
cd Chest-X-Ray-Disease-Classifier
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the Streamlit app:
```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

## Cloud Deployment

### Streamlit Cloud (Recommended)

1. Push your code to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app"
4. Connect your GitHub repository
5. Select the branch and `app.py` as the main file
6. Click "Deploy"

The app will be available at `https://[your-username]-[repo-name].streamlit.app`

### Heroku

1. Install Heroku CLI
2. Login to Heroku:
```bash
heroku login
```

3. Create a Heroku app:
```bash
heroku create your-app-name
```

4. Push to Heroku:
```bash
git push heroku main
```

Note: The `Procfile` is already configured for Streamlit deployment.

### Docker

1. Create a Dockerfile:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .
COPY models/ models/
COPY src/ src/

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.headless=true"]
```

2. Build and run:
```bash
docker build -t chest-xray-classifier .
docker run -p 8501:8501 chest-xray-classifier
```

### AWS EC2

1. Launch an EC2 instance (Ubuntu 20.04 LTS)
2. SSH into the instance
3. Install Python and pip:
```bash
sudo apt-get update
sudo apt-get install python3-pip python3-venv
```

4. Clone the repository and setup:
```bash
git clone https://github.com/sameerfcb/Chest-X-Ray-Disease-Classifier.git
cd Chest-X-Ray-Disease-Classifier
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

5. Run Streamlit with a background process manager (e.g., tmux or pm2):
```bash
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
```

6. Configure security group to allow port 8501

### Google Cloud Run

1. Create a Container in Google Cloud
2. Build and push the Docker image:
```bash
docker build -t gcr.io/[PROJECT_ID]/chest-xray-classifier .
docker push gcr.io/[PROJECT_ID]/chest-xray-classifier
```

3. Deploy to Cloud Run:
```bash
gcloud run deploy chest-xray-classifier \
  --image gcr.io/[PROJECT_ID]/chest-xray-classifier \
  --platform managed \
  --region us-central1 \
  --port 8501
```

## Environment Variables

Create a `.env` file for local development or set environment variables in your deployment platform:

```
# Example .env
DEBUG=false
```

## Performance Optimization

- The model runs on CPU by default
- For production, consider using GPU instances
- Model weights are cached for faster inference
- Images are cached to prevent re-processing

## Troubleshooting

### Model weights not found
- Ensure `xray_model.pth` or `xray_model_best.pth` is in the `models/` directory
- Check file permissions

### Memory issues
- Reduce `server.maxUploadSize` in `.streamlit/config.toml`
- Use deployment environments with sufficient RAM (at least 2GB required)

### Port already in use
- Change the port in `.streamlit/config.toml` or use `--server.port` flag

## Security Considerations

1. **Secrets Management**
   - Use environment variables for sensitive data
   - Never commit `.streamlit/secrets.toml` to version control
   - Use deployment platform's secret management

2. **Input Validation**
   - The app validates image formats
   - File size limits are enforced via Streamlit configuration

3. **HTTPS**
   - Use HTTPS in production
   - Most cloud platforms handle this automatically

## Monitoring

For production deployments, consider:
- Set up application logging
- Monitor resource usage (CPU, memory)
- Track inference times and model performance
- Set up alerts for errors

## Support

For issues or questions:
- Check the [GitHub Issues](https://github.com/sameerfcb/Chest-X-Ray-Disease-Classifier/issues)
- Review the [README](README.md)
- Check Streamlit documentation: https://docs.streamlit.io
