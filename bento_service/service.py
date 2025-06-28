"""BentoML service for Urdu ASR using fine-tuned Whisper model."""

import time
from typing import Annotated

import bentoml
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from pydantic import BaseModel, Field
import soundfile as sf
import io

# Response models
class TranscriptionResponse(BaseModel):
    transcript: str
    language: str = "urdu"
    processing_time: float
    audio_duration: float

class ErrorResponse(BaseModel):
    error: str
    error_type: str

# Load model as BentoML model
MODEL_REF = "kingabzpro/whisper-large-v3-turbo-urdu"
model = bentoml.transformers.get(MODEL_REF) or bentoml.transformers.import_model(
    MODEL_REF,
    model_name="whisper_urdu",
    signatures={"generate": {"batchable": True, "batch_dim": 0}}
)

@bentoml.service(
    name="urdu-asr",
    resources={"memory": "2Gi", "gpu": 1 if torch.cuda.is_available() else 0},
    traffic={"timeout": 30, "concurrency": 4}
)
class UrduASRService:
    """Urdu ASR service using fine-tuned Whisper."""
    
    def __init__(self):
        # Load processor and model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = WhisperProcessor.from_pretrained(MODEL_REF)
        self.model = WhisperForConditionalGeneration.from_pretrained(MODEL_REF).to(self.device)
        self.model.eval()

    @bentoml.api(
        input=bentoml.io.File(mime_type="audio/*"),
        output=bentoml.io.JSON(),
    )
    @bentoml.monitor("transcription_requests", "Number of transcription requests")
    def transcribe(self, audio_file: Annotated[bytes, bentoml.io.File()]) -> TranscriptionResponse | ErrorResponse:
        """Transcribe audio to Urdu text."""
        start_time = time.time()
        
        try:
            # Validate file size
            if len(audio_file) > 25 * 1024 * 1024:
                bentoml.monitor.log_counter("transcription_errors", {"type": "file_too_large"})
                return ErrorResponse(error="File too large (max 25MB)", error_type="validation_error")
            
            # Process audio
            audio_stream = io.BytesIO(audio_file)
            data, sample_rate = sf.read(audio_stream, dtype="float32")
            
            # Calculate audio duration
            audio_duration = len(data) / sample_rate
            bentoml.monitor.log_histogram("audio_duration_seconds", audio_duration)
            
            # Ensure mono
            if len(data.shape) > 1:
                data = data.mean(axis=1)
            
            # Transcribe
            inputs = self.processor.feature_extractor(
                data, sampling_rate=sample_rate, return_tensors="pt"
            )
            
            with torch.no_grad():
                predicted_ids = self.model.generate(
                    inputs.input_features.to(self.device),
                    language="urdu",
                    task="transcribe",
                    max_new_tokens=448
                )
            
            transcript = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            processing_time = time.time() - start_time
            
            # Log metrics
            bentoml.monitor.log_histogram("processing_time_seconds", processing_time)
            bentoml.monitor.log_counter("transcription_success")
            
            return TranscriptionResponse(
                transcript=transcript,
                processing_time=processing_time,
                audio_duration=audio_duration
            )
            
        except Exception as e:
            bentoml.monitor.log_counter("transcription_errors", {"type": "processing_error"})
            return ErrorResponse(error=str(e), error_type="processing_error")
    
    @bentoml.api()
    def health(self) -> dict:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "model": MODEL_REF,
            "device": self.device
        }
