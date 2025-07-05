# 🎤 Urdu ASR – Modern MLOps Pipeline

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache2.0-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![BentoML](https://img.shields.io/badge/BentoML-2.0+-green.svg)](https://bentoml.com/)

A production-ready end-to-end MLOps pipeline for Urdu Automatic Speech Recognition using fine-tuned Whisper-v3, featuring:

- 🔄 **Automated Pipeline**: Prefect orchestration for data processing, training, and deployment
- 📊 **Experiment Tracking**: MLflow integration for comprehensive model tracking
- 🚀 **Modern Serving**: BentoML 2.0+ with observability and monitoring
- 🎯 **Interactive UI**: Gradio interface for easy testing and demonstration
- 📈 **Observability**: Built-in metrics, logging, and health monitoring

## ✨ Features

- **Fine-tuned Whisper-large-v3** on Common Voice Urdu dataset
- **Real-time transcription** with audio duration and processing time metrics
- **Comprehensive error handling** with detailed error types and messages
- **Health monitoring** and service status endpoints
- **Scalable deployment** with configurable resources and concurrency
- **Development-friendly** with hot reloading and debugging support

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.11+
python --version

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export HF_TOKEN="your_huggingface_token"
export MLFLOW_TRACKING_URI="your_mlflow_uri"  # optional
```

### Run the Complete Pipeline

```bash
# Execute the full MLOps pipeline
prefect worker start --pool default-agent-pool &
prefect deploy --name urdu-asr-pipeline
prefect deployment run asr_pipeline/urdu-asr-pipeline

# Or run locally
python prefect/pipeline.py
```

### Serve the Model

```bash
# Build and serve locally
bentoml build
bentoml serve urdu-asr:latest

# Or deploy to BentoCloud
bentoml deploy urdu-asr:latest
```

### Launch Gradio Interface

```bash
# Update endpoint in gradio_app/app.py
# Then launch the interface
python gradio_app/app.py
```

## 📋 Pipeline Steps

1. **Data Download** (`scripts/download_dataset.py`)
   - Downloads Common Voice v17 Urdu dataset
   - Handles authentication with HuggingFace

2. **Data Preprocessing** (`scripts/preprocess_dataset.py`)
   - Filters for valid Urdu text using regex
   - Resamples audio to 16kHz mono FLAC
   - Parallel processing for efficiency

3. **Model Training** (`scripts/train.py`)
   - Fine-tunes Whisper-large-v3 on Urdu data
   - MLflow experiment tracking
   - Automatic checkpoint management
   - Early stopping and best model selection

4. **Model Evaluation** (`scripts/evaluate.py`)
   - Calculates Word Error Rate (WER)
   - Generates prediction samples

5. **Model Serving** (`bento_service/service.py`)
   - BentoML service with observability
   - Health checks and monitoring
   - Error handling and validation

## 🔧 Configuration

### Environment Variables

```bash
# Required
export HF_TOKEN="your_token"

# Optional - Training
export MODEL_ID="openai/whisper-large-v3"
export NUM_EPOCHS="3"
export TRAIN_BATCH_SIZE="16"
export LEARNING_RATE="1e-5"

# Optional - Serving
export BENTO_ENDPOINT="http://localhost:3000/transcribe"
export MAX_FILE_SIZE_MB="25"

# Optional - MLflow
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
```

### Resource Configuration

The BentoML service is configured for:
- **Memory**: 2Gi
- **GPU**: 1 (if available)
- **Concurrency**: 4 requests
- **Timeout**: 30 seconds

## 📊 Monitoring & Observability

The service includes comprehensive monitoring:

- **Request Metrics**: Count, duration, success/error rates
- **Audio Metrics**: File size, duration, format validation
- **Model Metrics**: Processing time, inference performance
- **Health Checks**: Service status and model availability

Access metrics at:
- Health: `GET /health`
- Metrics: `GET /metrics` (Prometheus format)
- Docs: `GET /docs` (OpenAPI)

## 🧪 Testing

```bash
# Test the service locally
curl -X POST "http://localhost:3000/transcribe" \
     -F "audio_file=@test_audio.wav"

# Health check
curl "http://localhost:3000/health"
```

## 📁 Project Structure

```
.
├── bento_service/
│   └── service.py          # Modern BentoML service
├── gradio_app/
│   ├── app.py              # Enhanced Gradio interface
│   └── requirements.txt    # UI dependencies
├── scripts/
│   ├── download_dataset.py # Data acquisition
│   ├── preprocess_dataset.py # Data processing
│   ├── train.py           # Model training
│   └── evaluate.py        # Model evaluation
├── prefect/
│   └── pipeline.py        # Orchestration pipeline
├── bentofile.yaml         # BentoML configuration
├── requirements.txt       # Core dependencies
└── README.md
```

